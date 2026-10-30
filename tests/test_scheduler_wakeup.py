"""Admission notifications and quota timers; real scheduler, no provider traffic."""
import asyncio
import threading

import pytest

from evals import vtime
from pool_manager.scheduler import AllEndpointsExhausted, LLMScheduler
from tests.fakes import make_ep


@pytest.mark.parametrize("calls,served,expired", [(48, 48, 0), (96, 96, 0), (100, 96, 4)])
def test_burst_uses_every_available_slot_without_overtaking(calls, served, expired):
    async def scenario():
        eps = [make_ep(f"g:m:K{i}", rpm_limit=12) for i in range(4)]
        sched = LLMScheduler(eps, clock=vtime.now, max_wait_seconds=70)
        accepted, rejected = [], []

        async def caller(i):
            try:
                ep, _ = await sched.acquire()
            except AllEndpointsExhausted:
                rejected.append(i)
                return
            accepted.append((i, vtime.now(), ep.id))
            ep.release()

        await asyncio.gather(*(caller(i) for i in range(calls)))
        assert len(accepted) == served
        assert len(rejected) == expired
        assert [i for i, _, _ in accepted] == list(range(served))
        assert all(t == 0 for _, t, _ in accepted[:48])
        assert all(60 <= t < 60.001 for _, t, _ in accepted[48:])
        assert all(ep.calls <= 24 for ep in eps)
        assert sched._waiters == []

    vtime.run_virtual(scenario)


def test_async_head_advance_wakes_already_armed_follower_without_time_passing():
    async def scenario():
        ep = make_ep("g:m:K")
        sched = LLMScheduler([ep], clock=vtime.now, max_wait_seconds=70)
        head = sched._join(70)  # Hold first place while the follower actually waits.
        follower = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        assert len(sched._waiters) == 2
        chosen, _ = sched._admit(head, vtime.now(), None)
        assert chosen is ep
        chosen.release()
        chosen, waited = await follower
        assert chosen is ep and waited == 0
        chosen.release()
        assert sched._waiters == []

    vtime.run_virtual(scenario)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_notification_between_check_and_sleep_is_not_lost(mode):
    ep = make_ep("g:m:K")
    sched = LLMScheduler([ep], max_wait_seconds=70)

    async def scenario():
        loop = asyncio.get_running_loop() if mode == "async" else None
        head = sched._join(sched.clock() + 70)
        follower = sched._join(sched.clock() + 70, loop)
        result, delay = sched._admit(follower, sched.clock(), None)
        assert result is None
        # Signal arrives before entering the actual blocking wait.
        chosen, _ = sched._admit(head, sched.clock(), None)
        chosen.release()
        if mode == "async":
            await asyncio.wait_for(sched._wait_async(follower, delay), 2)
        else:
            assert follower.signal.is_set()
            sched._wait_sync(follower, delay)
        chosen, _ = sched._admit(follower, sched.clock(), None)
        assert chosen is ep
        chosen.release()
        assert not sched._waiters

    asyncio.run(scenario())


def test_sync_head_notifies_async_follower_on_its_own_loop():
    async def scenario():
        ep = make_ep("g:m:K")
        sched = LLMScheduler([ep])
        head = sched._join(sched.clock() + 60)
        follower = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        failures = []

        def release_head():
            try:
                chosen, _ = sched._admit(head, sched.clock(), None)
                chosen.release()
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=release_head)
        thread.start()
        chosen, _ = await asyncio.wait_for(follower, 2)
        chosen.release()
        thread.join(2)
        assert not failures and not thread.is_alive() and not sched._waiters

    asyncio.run(scenario())


def test_async_head_notifies_sync_follower():
    ep = make_ep("g:m:K")
    sched = LLMScheduler([ep])
    head = sched._join(sched.clock() + 60)
    waiting = threading.Event()
    original_wait = sched._wait_sync
    accepted, failures = [], []

    def observed_wait(waiter, delay):
        waiting.set()
        original_wait(waiter, delay)

    sched._wait_sync = observed_wait

    def caller():
        try:
            chosen, _ = sched.acquire_sync()
            accepted.append(chosen.id)
            chosen.release()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=caller, daemon=True)
    thread.start()
    try:
        assert waiting.wait(2)

        async def release_head():
            chosen, _ = sched._admit(head, sched.clock(), None)
            chosen.release()

        asyncio.run(release_head())
        thread.join(2)
        assert not thread.is_alive() and accepted == [ep.id] and not failures
        assert not sched._waiters
    finally:
        sched._leave(head)
        sched.notify_changed()
        thread.join(2)


def test_cancelled_head_wakes_follower_after_eligibility_changes():
    async def scenario():
        ep = make_ep("g:m:K")
        sched = LLMScheduler([ep], clock=vtime.now, max_wait_seconds=70)
        ep.cooldown_until = 60
        first = asyncio.create_task(sched.acquire())
        second = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        assert len(sched._waiters) == 2
        ep.cooldown_until = 0
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        chosen, waited = await second
        assert waited == 0 and chosen is ep
        chosen.release()
        assert not sched._waiters

    vtime.run_virtual(scenario)


def test_external_state_notification_interrupts_known_reopening_timer():
    async def scenario():
        ep = make_ep("g:m:K")
        sched = LLMScheduler([ep], clock=vtime.now, max_wait_seconds=70)
        ep.cooldown_until = 60
        caller = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        ep.cooldown_until = 0
        sched.notify_changed()
        chosen, waited = await caller
        assert chosen is ep and waited == 0
        chosen.release()
        assert not sched._waiters

    vtime.run_virtual(scenario)


def test_uncontended_admission_allocates_no_notification_objects_or_sleeps():
    async def scenario():
        ep = make_ep("g:m:K")
        sched = LLMScheduler([ep])
        joined = []
        original_join = sched._join

        def observed_join(*args, **kwargs):
            waiter = original_join(*args, **kwargs)
            joined.append(waiter)
            return waiter

        sched._join = observed_join
        for _ in range(3):
            chosen, _ = await sched.acquire()
            chosen.release()
            chosen, _ = sched.acquire_sync()
            chosen.release()
        assert all(w.ready is None and w.signal is None for w in joined)

    asyncio.run(scenario())
