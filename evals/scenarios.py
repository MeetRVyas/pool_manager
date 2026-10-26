"""
Evaluation scenarios.

Topology used by every scenario unless stated otherwise (ILLUSTRATIVE numbers; the metrics
that matter are relative to this capacity, and a load sweep shows how they move with it):

    2 models x 3 API keys, each key in its own project
      flash : 10 requests/min per key    (primary, listed first)
      lite  : 15 requests/min per key    (fallback)
    nominal capacity = 3*10 + 3*15 = 75 requests/min = 1.25 requests/s

Load is expressed as rho = offered load / nominal capacity. rho > 1 is unservable by ANY
implementation, so success there is bounded by 1/rho and efficiency is what is compared.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from evals.provider import Spec

MODELS = ("flash", "lite")
TRUE_RPM = {"flash": 10, "lite": 15}
N_KEYS = 3
DURATION_S = 600.0


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    rho: float = 0.7
    duration_s: float = DURATION_S
    arrivals: str = "poisson"                 # poisson | bursty
    burst_every_s: float = 120.0
    enforcement: str = "sliding"              # how the provider counts a minute: sliding | fixed
    shared_project: bool = False              # all keys belong to ONE project (quota not additive)
    true_rpd: Optional[dict] = None           # {model: per-endpoint daily limit}
    yaml_scale: float = 1.0                   # what the limits file claims, as a multiple of the truth
    outage: Optional[tuple] = None            # (model, key_index, t0, t1)
    revoked: Optional[tuple] = None           # (key_index, from_t)
    p_server_error: float = 0.0
    group: str = "scenario"

    # ---- topology
    def specs(self) -> list[Spec]:
        out = []
        for model in MODELS:
            for k in range(1, N_KEYS + 1):
                outages = ()
                if self.outage and self.outage[0] == model and self.outage[1] == k:
                    outages = ((self.outage[2], self.outage[3]),)
                out.append(
                    Spec(
                        model=model,
                        key_name=f"KEY_{k}",
                        api_key=f"sk-eval-{k}",
                        project="P" if self.shared_project else f"P{k}",
                        true_rpm=TRUE_RPM[model],
                        true_rpd=(self.true_rpd or {}).get(model),
                        outages=outages,
                        revoked_from=(self.revoked[1] if self.revoked and self.revoked[0] == k else None),
                        p_server_error=self.p_server_error,
                    )
                )
        return out

    def capacity_rps(self) -> float:
        """Nominal throughput ceiling: one quota bucket per (project, model)."""
        buckets = {(s.project, s.model): s.true_rpm for s in self.specs()}
        return sum(buckets.values()) / 60.0

    def max_accepted(self, horizon_s: float) -> float:
        """
        Upper bound on requests ANY implementation could get the provider to accept within
        `horizon_s`. A sliding-window limit lets a quota be used in full at the start and again
        every 60 s, so a bucket accepts at most rpm * (horizon/60 + 1), not rpm * horizon/60.
        Outage and revocation time is subtracted; a daily cap, if any, applies on top.
        """
        if self.shared_project:                                  # one bucket per (project, model); no faults modelled
            buckets = {(s.project, s.model): (s.true_rpm, s.true_rpd) for s in self.specs()}
            return sum(min(rpm * (horizon_s / 60.0 + 1), rpd if rpd is not None else float("inf")) for rpm, rpd in buckets.values())
        total = 0.0
        for s in self.specs():
            usable = horizon_s
            for a, b in s.outages:
                usable -= max(0.0, min(b, horizon_s) - min(a, horizon_s))
            if s.revoked_from is not None:
                usable -= max(0.0, horizon_s - s.revoked_from)
            if usable <= 0:
                continue
            cap = s.true_rpm * (usable / 60.0 + 1)
            if s.true_rpd is not None:
                cap = min(cap, s.true_rpd)
            total += cap
        return total

    # ---- what the operator wrote down (the limits file)
    def yaml_limits(self) -> dict:
        lim = {}
        for model in MODELS:
            entry = {"rpm": max(1, round(TRUE_RPM[model] * self.yaml_scale))}
            if self.true_rpd and model in self.true_rpd:
                entry["rpd"] = max(1, round(self.true_rpd[model] * self.yaml_scale))
            lim[model] = entry
        return lim

    # ---- offered load
    def arrival_times(self, rng: random.Random) -> list[float]:
        rate = self.rho * self.capacity_rps()
        times: list[float] = []
        if self.arrivals == "poisson":
            t = rng.expovariate(rate)
            while t < self.duration_s:
                times.append(t)
                t += rng.expovariate(rate)
        elif self.arrivals == "bursty":
            size = max(1, round(rate * self.burst_every_s))      # same average rate as poisson
            t0 = 5.0
            while t0 < self.duration_s:
                times.extend(t0 + rng.uniform(0, 1.0) for _ in range(size))
                t0 += self.burst_every_s
        else:  # pragma: no cover
            raise ValueError(self.arrivals)
        return sorted(times)


def core_scenarios() -> list[Scenario]:
    return [
        Scenario("steady_50", "Light steady load (50% of capacity), healthy provider.", rho=0.5),
        Scenario("steady_90", "Heavy steady load (90% of capacity), healthy provider.", rho=0.9),
        Scenario("steady_90_fixedwin", "Same, but the provider counts fixed clock-minutes, not a sliding minute.",
                 rho=0.9, enforcement="fixed"),
        Scenario("bursty_60", "Pipeline-style fan-out: 60% average load arriving in synchronized bursts every 2 min.",
                 rho=0.6, arrivals="bursty"),
        Scenario("overload_150", "150% of capacity: unservable by anyone; success is capped at ~67%.", rho=1.5),
        Scenario("chaos_70", "70% load with a 3-minute flash outage on key 2, 1% random 5xx, and key 3 revoked mid-run.",
                 rho=0.7, outage=("flash", 2, 150.0, 330.0), revoked=(3, 300.0), p_server_error=0.01),
        Scenario("yaml_optimistic", "Limits file claims 1.5x the real limits (stale/wrong YAML), 80% load.",
                 rho=0.8, yaml_scale=1.5),
        Scenario("yaml_pessimistic", "Limits file claims 0.6x the real limits (over-cautious YAML), 80% load.",
                 rho=0.8, yaml_scale=0.6),
        Scenario("shared_project", "All 3 keys belong to ONE Google project, so quota is NOT additive; 70% of the true capacity.",
                 rho=0.7, shared_project=True),
        Scenario("daily_quota", "Daily quotas (30/day flash, 50/day lite per key) run out mid-run at 60% rpm load.",
                 rho=0.6, true_rpd={"flash": 30, "lite": 50}),
    ]


def sweep_scenarios() -> list[Scenario]:
    return [
        Scenario(f"load_{int(r * 100):03d}", f"Load sweep point: {int(r * 100)}% of capacity.", rho=r, group="sweep")
        for r in (0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0)
    ]


def all_scenarios() -> dict[str, Scenario]:
    return {s.name: s for s in core_scenarios() + sweep_scenarios()}
