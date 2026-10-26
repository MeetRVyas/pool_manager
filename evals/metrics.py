"""Per-run metrics, and statistics across seeds (bootstrap confidence intervals, paired differences)."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from evals.provider import CallCtx

SLOW_CALL_S = 30.0


@dataclass
class CallResult:
    i: int
    t_arrival: float
    latency: float
    outcome: str       # ok | exhausted | attempts_failed | leaked_429 | leaked_5xx | leaked_auth | other
    ctx: CallCtx


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in 0..100), like numpy's default."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def jain(xs: Sequence[float]) -> Optional[float]:
    """Jain's fairness index: 1.0 = perfectly even, 1/n = all load on one."""
    xs = list(xs)
    if not xs or sum(xs) == 0:
        return None
    return sum(xs) ** 2 / (len(xs) * sum(x * x for x in xs))


def summarize(results: list[CallResult], counters: dict, max_accepted: float) -> dict:
    """`max_accepted`: the most requests any implementation could have had accepted (Scenario.max_accepted)."""
    n = len(results)
    by = Counter(r.outcome for r in results)
    ok = [r for r in results if r.outcome == "ok"]
    failed = [r for r in results if r.outcome != "ok"]
    requests = sum(r.ctx.requests for r in results)
    rejected = sum(r.ctx.r429 + r.ctx.s5xx + r.ctx.auth for r in results)
    accepted = sum(c["accepted"] for c in counters.values())
    per_key: Counter = Counter()
    for ep_id, c in counters.items():
        per_key[ep_id.rsplit(":", 1)[-1]] += c["accepted"]

    def rate(k: str) -> float:
        return by[k] / n if n else 0.0

    lat_ok = [r.latency for r in ok]
    return {
        "n_calls": n,
        # ---- the headline: did the caller get a response?
        "success_rate": len(ok) / n if n else 0.0,
        "exhausted_rate": rate("exhausted"),                      # AllEndpointsExhausted
        "attempts_failed_429_rate": rate("attempts_failed"),      # AllAttemptsFailed, every attempt rate-limited
        "leaked_429_rate": rate("leaked_429"),                    # a raw rate-limit error reached the caller
        # headline: calls that failed BECAUSE of rate limits, by any route
        "ratelimit_failure_rate": rate("exhausted") + rate("attempts_failed") + rate("leaked_429"),
        "attempts_failed_other_rate": rate("attempts_failed_other"),   # all attempts hit outages / dead keys
        "leaked_5xx_rate": rate("leaked_5xx"),
        "other_fail_rate": rate("leaked_auth") + rate("other"),
        # ---- efficiency: how much provider traffic was wasted on rejections
        "clean_success_rate": sum(1 for r in ok if r.ctx.r429 + r.ctx.s5xx + r.ctx.auth == 0) / n if n else 0.0,
        "provider_429_per_100_calls": 100.0 * sum(r.ctx.r429 for r in results) / n if n else 0.0,
        "wasted_request_pct": 100.0 * rejected / requests if requests else 0.0,
        "requests_per_call": requests / n if n else 0.0,
        # ---- waiting ("long waiting")
        "lat_p50": percentile(lat_ok, 50),
        "lat_p95": percentile(lat_ok, 95),
        "lat_p99": percentile(lat_ok, 99),
        "lat_max": max(lat_ok) if lat_ok else None,
        "fail_latency_mean": (sum(r.latency for r in failed) / len(failed)) if failed else None,
        "slow_call_rate": sum(1 for r in results if r.latency > SLOW_CALL_S) / n if n else 0.0,
        "wasted_wait_s_per_failure": (sum(r.latency for r in failed) / len(failed)) if failed else 0.0,
        # ---- capacity use: how close to the best ANY implementation could possibly do
        "ceiling_success": min(1.0, max_accepted / n) if n else 0.0,
        "efficiency": min(1.0, (len(ok) / n) / min(1.0, max_accepted / n)) if n and max_accepted > 0 else 0.0,
        "accepted_requests": accepted,
        "keys_used": sum(1 for v in per_key.values() if v > 0),
        "key_jain": jain(list(per_key.values())),
    }


# ----------------------------------------------------------------------- statistics
def bootstrap_ci(values: Sequence[Optional[float]], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 12345):
    """(mean, lo, hi). Percentile bootstrap over seeds; None values (undefined for a seed) are dropped."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None, None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return mean, boots[int(n_boot * alpha / 2)], boots[int(n_boot * (1 - alpha / 2)) - 1]


def paired_diff_ci(a: Sequence[Optional[float]], b: Sequence[Optional[float]], **kw):
    """CI of mean(a_i - b_i) over seeds where both are defined (paired: same seed = same arrivals)."""
    return bootstrap_ci([x - y for x, y in zip(a, b) if x is not None and y is not None], **kw)
