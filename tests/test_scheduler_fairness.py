"""Deterministic (virtual-time, seeded) fairness checks for LLMScheduler.acquire.

The previous polling admission let later arrivals overtake older waiters. These
regressions require FIFO admission using the actual scheduler and endpoint state.

Offline only: real LLMScheduler + EndpointState on evals.vtime's virtual loop, no clients, no network.
"""
import asyncio

import pytest

from evals import vtime
from evals.live_eval import Plan, arrivals
from pool_manager.endpoint_state import EndpointState
from pool_manager.scheduler import AllEndpointsExhausted, LLMScheduler


def serve(rate_per_min, calls=1000, *, seed=1, rpm=15, keys=2, max_wait=70.0, service_s=0.5):
    """Return ([(served_at, id, arrival)], [(id, arrival)]) for Poisson arrivals against `keys` x `rpm` RPM."""
    offsets = arrivals(Plan(calls=calls, rate_per_min=rate_per_min, pattern="poisson", seed=seed))

    async def main():
        endpoints = [EndpointState(id=f"k{i}", provider="g", model="m", client=None, rpm_limit=rpm)
                     for i in range(keys)]
        scheduler = LLMScheduler(endpoints, max_wait_seconds=max_wait, clock=vtime.now)
        served, expired = [], []

        async def caller(i, arrival):
            await asyncio.sleep(arrival)
            try:
                endpoint, _ = await scheduler.acquire()
            except AllEndpointsExhausted:
                expired.append((i, arrival))
                return
            served.append((vtime.now(), i, arrival))
            await asyncio.sleep(service_s)
            endpoint.release()

        await asyncio.gather(*(caller(i, t) for i, t in enumerate(offsets)))
        return served, expired

    return vtime.run_virtual(main)


def max_overtakers(served, tolerance_s=1.0):
    """Largest number of callers served BEFORE one caller although they arrived more than `tolerance_s` later."""
    order = sorted(served)
    return max((sum(1 for _, _, a2 in order[:pos] if a2 > arrival + tolerance_s)
                for pos, (_, _, arrival) in enumerate(order)), default=0)


def test_no_local_waiting_at_80_percent_load_matches_live_evidence():
    """Guards the *expected* behaviour: the 24/min evidence schedule never waits for quota (observed: 0.0175 s total)."""
    offsets = arrivals(Plan(calls=100, rate_per_min=24, pattern="poisson", seed=1))
    served, expired = serve(24, calls=len(offsets))
    assert not expired
    assert max(t - a for t, _, a in served) < 0.01


def test_saturation_produces_long_but_bounded_waiting_not_failure_of_the_scheduler():
    """At rho = 1.0 a queue has no stable wait time; p95 of tens of seconds is expected backpressure."""
    served, expired = serve(30)
    waits = sorted(t - a for t, _, a in served)
    assert waits[int(0.95 * len(waits))] > 25.0           # the observed 42 s scheduler p95 is in this regime


def test_service_order_follows_arrival_order_at_80_percent_load():
    served, expired = serve(24)
    assert not expired
    assert max_overtakers(served) == 0                    # pre-FIFO: 9 on seed 1


def test_no_caller_expires_that_fifo_would_have_served():
    served, expired = serve(30)
    assert not expired                                    # pre-FIFO: 19/1000 on seed 1
