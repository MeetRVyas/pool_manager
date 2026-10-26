"""Run one (scenario, implementation, seed) simulation and return its metrics."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from pool_manager.errors import Kind, classify
from evals import vtime
from evals.impls import build
from evals.metrics import CallResult, summarize
from evals.provider import CALL, CallCtx, World
from evals.scenarios import Scenario


def outcome_of(exc: BaseException) -> str:
    """What the CALLER experienced. The pools' own exception names win; otherwise classify the provider error."""
    name = type(exc).__name__
    if name == "AllEndpointsExhausted":
        return "exhausted"
    if name == "AllAttemptsFailed":
        # Every attempt failed. If the cause was rate limiting it counts as a rate-limit failure
        # (it is "all endpoints exhausted" by another route); otherwise it was an outage.
        cause = exc.__cause__
        rate_limited = cause is not None and classify(cause).kind in (Kind.RATE_LIMIT, Kind.QUOTA)
        return "attempts_failed" if rate_limited else "attempts_failed_other"
    kind = classify(exc).kind
    return {
        Kind.RATE_LIMIT: "leaked_429",
        Kind.QUOTA: "leaked_429",
        Kind.TRANSIENT: "leaked_5xx",
        Kind.DEAD: "leaked_auth",
    }.get(kind, "other")


async def _drive(call, arrivals: list[float]) -> list[CallResult]:
    loop = asyncio.get_running_loop()
    results: list[CallResult] = []

    async def one(i: int, t_arrival: float) -> None:
        await asyncio.sleep(max(0.0, t_arrival - loop.time()))
        ctx = CallCtx()
        CALL.set(ctx)               # each task owns its context copy: provider events are attributed to this call
        started = loop.time()
        outcome = "ok"
        try:
            await call(i)
        except Exception as exc:    # anything reaching here is what a real caller would have to handle
            outcome = outcome_of(exc)
        results.append(CallResult(i, t_arrival, loop.time() - started, outcome, ctx))

    await asyncio.gather(*(one(i, t) for i, t in enumerate(arrivals)))
    return results


def simulate(scn: Scenario, impl: str, seed: int) -> dict[str, Any]:
    """Deterministic: identical (scn, impl, seed) -> identical metrics. Arrivals depend on (scn, seed) only,
    so every implementation faces exactly the same offered load for a given seed (paired comparison)."""
    arrivals = scn.arrival_times(random.Random(f"arrivals|{scn.name}|{seed}"))

    async def main() -> dict[str, Any]:
        world = World(scn.specs(), rng=random.Random(f"world|{scn.name}|{seed}"), enforcement=scn.enforcement)
        call = build(impl, scn, world)
        results = await _drive(call, arrivals)
        end = max(vtime.now(), scn.duration_s)       # the provider can accept nothing after the last call finished,
        out = summarize(results, world.counters, scn.max_accepted(end))   # so this is a valid (and tight) capacity bound
        out["virtual_seconds"] = vtime.now()
        return out

    return vtime.run_virtual(main)
