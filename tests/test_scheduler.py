"""Selection (all keys used, model priority), bounded waiting, and thread safety."""
import asyncio
import collections
import threading
import time

import pytest

from pool_manager.scheduler import AllEndpointsExhausted
from tests.fakes import FakeTime, make_ep, make_scheduler, run

KEYS = ("KEY_1", "KEY_2", "KEY_3")


def grid(models=("flash", "lite"), **limits):
    """models x keys, model-major, priority = model order (what build_endpoints produces)."""
    return [make_ep(f"google:{m}:{k}", priority=i, **limits) for i, m in enumerate(models) for k in KEYS]


def take(sched, n):
    counts = collections.Counter()
    for _ in range(n):
        with sched.allocate_endpoint_sync() as ep:
            counts[ep.id] += 1
    return counts


# --------------------------------------------------------------- all keys get used
def test_sequential_traffic_rotates_over_every_key_of_the_active_model():
    # Baseline v3 sent ALL 30 sequential calls to KEY_1 (1 of 6 endpoints used).
    counts = take(make_scheduler(grid()), 30)
    assert counts == {"google:flash:KEY_1": 10, "google:flash:KEY_2": 10, "google:flash:KEY_3": 10}


def test_balanced_strategy_uses_every_model_and_every_key():
    sched = make_scheduler(grid(), strategy="balanced")
    counts = take(sched, 30)
    assert len(counts) == 6 and set(counts.values()) == {5}
    assert sched.unused_endpoints() == []


def test_concurrent_burst_also_spreads_over_all_endpoints():
    sched = make_scheduler(grid(), strategy="balanced")

    async def burst():
        async def one():
            async with sched.allocate_endpoint() as ep:
                await sched._asleep(0.0)
                return ep.id
        return await asyncio.gather(*[one() for _ in range(12)])

    assert len(set(run(burst()))) == 6


# ------------------------------------------------- model priority + pre-stop switch
def test_switches_to_the_fallback_model_exactly_when_the_primary_hits_its_limit():
    eps = grid(rpm_limit=2)                                    # 3 keys x 2 rpm = 6 requests/min per model
    sched = make_scheduler(eps)
    counts = take(sched, 12)                                   # no time passes: one minute window
    primary = sum(v for k, v in counts.items() if ":flash:" in k)
    fallback = sum(v for k, v in counts.items() if ":lite:" in k)
    assert (primary, fallback) == (6, 6)
    assert all(v == 2 for v in counts.values())                # nobody was driven past its limit
    assert [ep.rate_limited for ep in eps] == [0] * 6          # ...and no 429 was ever needed


def test_primary_model_is_preferred_again_once_its_window_reopens():
    ft = FakeTime()
    sched = make_scheduler(grid(rpm_limit=1), ft)
    take(sched, 6)                                             # everything used once
    ft.t += 61
    counts = take(sched, 3)
    assert set(counts) == {f"google:flash:{k}" for k in KEYS}


def test_a_cooling_down_primary_is_skipped_without_waiting():
    ft = FakeTime()
    eps = grid()
    sched = make_scheduler(eps, ft)
    for ep in eps[:3]:
        ep.mark_rate_limited(ft.now(), retry_after=30)
    counts = take(sched, 3)
    assert all(":lite:" in k for k in counts) and ft.sleeps == []


# --------------------------------------------------------------- bounded waiting
def test_waits_exactly_until_the_earliest_endpoint_frees_up_when_that_fits_the_budget():
    ft = FakeTime()
    eps = grid(models=("flash",), rpm_limit=1)
    sched = make_scheduler(eps, ft, max_wait_seconds=60)
    take(sched, 3)                                             # all three keys spent at t0
    ft.sleeps.clear()
    ep, waited = sched.acquire_sync()
    assert ep is not None
    assert waited == pytest.approx(60.0, abs=0.01)
    assert ft.total_slept == pytest.approx(60.0, abs=0.01)     # slept until it was free, not a fixed poll


def test_fails_fast_when_every_endpoint_is_out_of_daily_quota():
    # v3 slept the full 120 s here and THEN raised, for a quota that resets hours later.
    ft = FakeTime()
    eps = grid()
    for ep in eps:
        ep.mark_rate_limited(ft.now(), quota_exhausted=True)   # cooldown 3600 s
    sched = make_scheduler(eps, ft, max_wait_seconds=60)
    with pytest.raises(AllEndpointsExhausted) as err:
        sched.acquire_sync()
    assert ft.total_slept == 0.0                               # no waiting at all
    assert err.value.retry_after == pytest.approx(3600.0)
    assert "free in" in str(err.value)


def test_fails_fast_when_the_wait_would_overshoot_the_budget():
    ft = FakeTime()
    eps = grid(models=("flash",))
    for ep in eps:
        ep.mark_rate_limited(ft.now(), retry_after=89)         # free in 90 s, budget is 60 s
    with pytest.raises(AllEndpointsExhausted):
        make_scheduler(eps, ft, max_wait_seconds=60).acquire_sync()
    assert ft.total_slept == 0.0


def test_async_acquire_has_the_same_budget_rules():
    ft = FakeTime()
    eps = grid(models=("flash",))
    for ep in eps:
        ep.mark_rate_limited(ft.now(), retry_after=89)
    sched = make_scheduler(eps, ft, max_wait_seconds=60)
    with pytest.raises(AllEndpointsExhausted):
        run(sched.acquire())
    assert ft.total_slept == 0.0


def test_zero_budget_never_sleeps():
    ft = FakeTime()
    eps = grid(models=("flash",), rpm_limit=1)
    sched = make_scheduler(eps, ft)
    take(sched, 3)
    with pytest.raises(AllEndpointsExhausted):
        sched.acquire_sync(0)
    assert ft.total_slept == 0.0


def test_no_endpoints_is_reported_clearly():
    with pytest.raises(AllEndpointsExhausted, match="No endpoints"):
        make_scheduler([]).acquire_sync()


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        make_scheduler(grid(), strategy="fastest")


# -------------------------------------------------------------------- thread safety
def test_threads_can_never_push_an_endpoint_past_its_limit():
    eps = [make_ep(f"google:m:KEY_{i}", rpm_limit=100) for i in range(2)]
    sched = make_scheduler(eps)
    sched.clock = time.monotonic                                # real clock: this test races real threads
    granted = []

    def worker():
        for _ in range(25):
            try:
                ep, _ = sched.acquire_sync(0)
                granted.append(ep.id)
                ep.release()
            except AllEndpointsExhausted:
                pass

    threads = [threading.Thread(target=worker) for _ in range(16)]   # 400 attempts for 200 slots
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(granted) == 200
    assert [ep.calls for ep in eps] == [100, 100]


# ------------------------------------------------------------------- observability
def test_usage_report_lists_every_endpoint_and_names_the_unused_ones():
    eps = grid(models=("flash",))
    sched = make_scheduler(eps)
    take(sched, 1)
    report = sched.usage_report()
    assert all(k in report for k in KEYS)
    assert "1/3 endpoints used" in report and "never used" in report
