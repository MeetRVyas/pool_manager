"""
Measure the pool against the REAL provider with YOUR keys.

This is the only script in this package whose numbers may be described as "measured on live
API traffic". Unit tests use injected fake clients (see tests/test_live_eval.py).
Read the dry-run output first and start with a small --calls.

    # 1. check config, keys and limits; makes no API call
    python -m evals.live_eval --config configs/llm_config.yaml --dry-run

    # 2. the pool alone: 60 tiny calls at ~90 calls/min
    python -m evals.live_eval --config configs/llm_config.yaml --mode pool --calls 60 --rate-per-min 90 --yes

    # 3. pool vs. "just use the first key" on the same arrival schedule (waits --pause s in between
    #    so the per-minute windows of the first run have expired)
    python -m evals.live_eval --config configs/llm_config.yaml --mode both --calls 60 --rate-per-min 90 --yes \
        --out live_results.json

Modes
    pool           the merged pool WITH configs/model_limits.yaml
    pool-reactive  the merged pool WITHOUT a limits file (reactive 429 handling only)
    single-key     the first model with the first key and the SDK's default retries
    both           single-key, then pool

What it reports (per mode)
    success %, how the failures looked (AllEndpointsExhausted / AllAttemptsFailed / raw 429 / 5xx),
    p50/p95/p99/max latency of successful calls, mean time spent on a failed call, calls slower
    than 30 s, which keys received traffic, and (pool modes) the rate-limit responses the pool saw.
    The single-key baseline cannot report 429s: the SDK retries them out of sight.

Cost control: every call is a one-line prompt asking for a one-word answer, `--max-calls` is a
hard cap (default 200), and nothing is sent without --yes.

Caveats that no script can remove
    * One run is one sample of a quota state you do not control (time of day, other users of the
      project, whether your keys share a project). Repeat at different times before quoting a number.
    * The interval printed beside a success rate is a Wilson 95% interval that treats calls as
      independent. They are not (they share quota windows), so read it as a rough guide only.
    * Keys that belong to ONE project share ONE quota; the pool then cannot add capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage

from evals.metrics import jain, summarize
from evals.sim import _drive

MODES = ("pool", "pool-reactive", "single-key", "both")
DEFAULT_MAX_CALLS = 200
LIVE_KEYS = (  # the subset of metrics.summarize that is meaningful without provider-side counters
    "n_calls", "success_rate", "exhausted_rate", "attempts_failed_429_rate", "leaked_429_rate",
    "ratelimit_failure_rate", "attempts_failed_other_rate", "leaked_5xx_rate", "other_fail_rate",
    "lat_p50", "lat_p95", "lat_p99", "lat_max", "fail_latency_mean", "slow_call_rate",
)


@dataclass(frozen=True)
class Plan:
    calls: int = 60
    rate_per_min: float = 90.0
    pattern: str = "poisson"          # poisson | burst
    seed: int = 1
    prompt: str = "Reply with exactly one word: ok"
    max_wait: float = 60.0            # the pool's max_wait_seconds
    request_timeout: float = 60.0
    limits_path: Optional[str] = "auto"


def arrivals(plan: Plan) -> list[float]:
    """Seconds after start at which each call is issued. Same (seed, plan) -> same schedule for every mode."""
    rng = random.Random(f"live|{plan.pattern}|{plan.seed}")
    if plan.pattern == "burst":
        return sorted(rng.uniform(0.0, 1.0) for _ in range(plan.calls))
    if plan.pattern != "poisson":
        raise ValueError(f"unknown pattern {plan.pattern!r}")
    rate = plan.rate_per_min / 60.0
    t, out = 0.0, []
    for _ in range(plan.calls):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (assumes independent calls)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------------------------ targets
@dataclass
class Target:
    name: str
    call: Callable[[int], Awaitable[Any]]
    stats: Callable[[], list[dict]]


def build_target(mode: str, config: str | Path, plan: Plan, *, clock: Optional[Callable[[], float]] = None) -> Target:
    """`clock` is only for tests that run under a virtual clock; real runs leave it None."""
    from pool_manager import llm as pm  # imported late so tests can patch the client classes on it

    messages = [HumanMessage(plan.prompt)]

    if mode in ("pool", "pool-reactive"):
        limits = plan.limits_path if mode == "pool" else None
        pool = pm.get_llm(config, max_wait_seconds=plan.max_wait, limits_path=limits, request_timeout=plan.request_timeout)
        if clock is not None:
            pool.scheduler.clock = clock

        async def call(i: int) -> Any:
            return await pool.ainvoke(messages)

        return Target(mode, call, pool.stats)

    if mode == "single-key":
        first = pm.build_endpoints(config, limits_path=None, request_timeout=plan.request_timeout)[0]
        key_name = first.id.rsplit(":", 1)[-1]
        client_cls = pm.ChatGoogleGenerativeAI if first.provider.lower() in ("google", "gemini") else pm.ChatOpenAI
        # Library-default retries on purpose: this is "what you get by just calling the model".
        client = client_cls(model=first.model, api_key=os.environ[key_name], timeout=plan.request_timeout)

        async def call(i: int) -> Any:
            return await client.ainvoke(messages)

        return Target(f"single-key ({first.model}, {key_name})", call, lambda: [])

    raise ValueError(f"unknown mode {mode!r}")


# ------------------------------------------------------------------------------------ running
def summarize_live(results: list, stats: list[dict]) -> dict:
    full = summarize(results, {}, 0.0)
    out: dict[str, Any] = {k: full[k] for k in LIVE_KEYS}
    ok = sum(1 for r in results if r.outcome == "ok")
    lo, hi = wilson(ok, len(results))
    out["success_ci95_wilson"] = [lo, hi]
    out["outcomes"] = dict(Counter(r.outcome for r in results))
    if stats:
        per_key: Counter = Counter()
        for s in stats:
            per_key[s["id"].rsplit(":", 1)[-1]] += s["calls"]
        out["keys_used"] = sum(1 for v in per_key.values() if v > 0)
        out["keys_total"] = len(per_key)
        out["key_jain"] = jain(list(per_key.values()))
        out["pool_rate_limit_responses"] = sum(s["rate_limited"] + s["quota_hits"] for s in stats)
        out["pool_server_errors"] = sum(s["server_errors"] for s in stats)
        out["pool_dead_key_errors"] = sum(s["dead_errors"] for s in stats)
        out["per_endpoint"] = [
            {k: s[k] for k in ("id", "calls", "successes", "rate_limited", "quota_hits", "server_errors", "dead_errors")} for s in stats
        ]
    else:
        out["note"] = "429s are retried inside the SDK and are invisible here."
    return out


async def run_mode(mode: str, config: str | Path, plan: Plan, *, clock: Optional[Callable[[], float]] = None) -> dict:
    target = build_target(mode, config, plan, clock=clock)
    results = await _drive(target.call, arrivals(plan))
    out = summarize_live(results, target.stats())
    out["target"] = target.name
    return out


async def run_modes(modes: list[str], config: str | Path, plan: Plan, *, pause_s: float = 70.0,
                    clock: Optional[Callable[[], float]] = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, mode in enumerate(modes):
        if i:
            await asyncio.sleep(pause_s)           # let the previous run's per-minute windows expire
        out[mode] = await run_mode(mode, config, plan, clock=clock)
    return out


# ------------------------------------------------------------------------------------ CLI
def _fmt(m: dict) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    sec = lambda x: "-" if x is None else f"{x:.1f}s"
    lo, hi = m["success_ci95_wilson"]
    lines = [
        f"  target               {m['target']}",
        f"  calls                {m['n_calls']}",
        f"  success              {pct(m['success_rate'])}   (Wilson 95% ~ {pct(lo)}-{pct(hi)}, rough)",
        f"  rate-limit failures  {pct(m['ratelimit_failure_rate'])}   (exhausted {pct(m['exhausted_rate'])}, "
        f"all-attempts-429 {pct(m['attempts_failed_429_rate'])}, raw 429 {pct(m['leaked_429_rate'])})",
        f"  other failures       {pct(m['attempts_failed_other_rate'] + m['leaked_5xx_rate'] + m['other_fail_rate'])}",
        f"  latency (ok calls)   p50 {sec(m['lat_p50'])}  p95 {sec(m['lat_p95'])}  p99 {sec(m['lat_p99'])}  max {sec(m['lat_max'])}",
        f"  failed calls         mean {sec(m['fail_latency_mean'])} spent each; {pct(m['slow_call_rate'])} of all calls took > 30s",
    ]
    if "keys_used" in m:
        lines.append(f"  keys used            {m['keys_used']}/{m['keys_total']}  (evenness {m['key_jain']:.2f})   "
                     f"429s seen by the pool: {m['pool_rate_limit_responses']}")
    else:
        lines.append(f"  note                 {m['note']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llm_config.yaml")
    ap.add_argument("--env-file", default=".env", help="dotenv file to load without overriding existing environment variables")
    ap.add_argument("--mode", choices=MODES, default="pool")
    ap.add_argument("--calls", type=int, default=60)
    ap.add_argument("--rate-per-min", type=float, default=90.0, help="offered load; pick a value ABOVE one key's limit to see a difference")
    ap.add_argument("--pattern", choices=("poisson", "burst"), default="poisson")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-wait", type=float, default=60.0, help="the pool's max_wait_seconds")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request timeout")
    ap.add_argument("--limits", default="auto", help="model-limits YAML ('auto' = configs/model_limits.yaml)")
    ap.add_argument("--pause", type=float, default=70.0, help="seconds between modes in --mode both")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="hard cap on --calls")
    ap.add_argument("--out", help="write the full results as JSON")
    ap.add_argument("--dry-run", action="store_true", help="check config/keys/limits and print the plan; sends nothing")
    ap.add_argument("--yes", action="store_true", help="confirm that real API calls may be made")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)

    if args.calls < 1 or args.calls > args.max_calls:
        ap.error(f"--calls must be between 1 and --max-calls ({args.max_calls})")
    plan = Plan(calls=args.calls, rate_per_min=args.rate_per_min, pattern=args.pattern, seed=args.seed,
                max_wait=args.max_wait, request_timeout=args.timeout, limits_path=None if args.limits.lower() == "none" else args.limits)
    modes = ["single-key", "pool"] if args.mode == "both" else [args.mode]

    from pool_manager import llm as pm
    try:
        endpoints = pm.build_endpoints(args.config, limits_path=plan.limits_path, request_timeout=plan.request_timeout)
    except Exception as exc:  # missing env var, bad YAML, ...
        print(f"config problem: {exc}", file=sys.stderr)
        return 2
    print(pm.describe_endpoints(endpoints))
    span = plan.calls / (plan.rate_per_min / 60.0) if plan.pattern == "poisson" else 1.0
    print(f"\nplan: {plan.calls} calls, {plan.pattern}, ~{plan.rate_per_min:.0f}/min (about {span:.0f}s), modes: {', '.join(modes)}")
    if args.dry_run:
        print("dry run: nothing was sent.")
        return 0
    if not args.yes:
        print("refusing to call the API without --yes (use --dry-run to inspect the plan).", file=sys.stderr)
        return 2

    started = time.time()
    results = asyncio.run(run_modes(modes, args.config, plan, pause_s=args.pause))
    for mode, m in results.items():
        print(f"\n[{mode}]\n{_fmt(m)}")
    print("\nOne run is one sample; repeat at different times of day before quoting a number.")
    if args.out:
        Path(args.out).write_text(json.dumps({"plan": asdict(plan), "wall_seconds": round(time.time() - started, 1),
                                              "results": results}, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
