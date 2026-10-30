"""FIFO queue cleanup and sync/async sharing, independent of provider traffic."""
import asyncio
import threading

import pytest

from pool_manager.scheduler import AllEndpointsExhausted, LLMScheduler
from tests.fakes import FakeTime, make_ep


def test_cancelled_oldest_waiter_leaves_no_blocker():
    async def scenario():
        ft = FakeTime(start=0)
        gate = asyncio.Event()
        async def blocked_sleep(delay):
            await gate.wait()
        ep = make_ep("google:m:KEY", rpm_limit=1)
        assert ep.try_reserve(0)
        ep.release()
        sched = LLMScheduler([ep], clock=ft.now, asleep=blocked_sleep, max_wait_seconds=70)
        oldest = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        later = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        assert len(sched._waiters) == 2
        oldest.cancel()
        with pytest.raises(asyncio.CancelledError):
            await oldest
        ft.t = 60
        gate.set()
        chosen, _ = await later
        assert chosen is ep
        chosen.release()
        assert sched._waiters == []
    asyncio.run(scenario())


def test_expired_head_does_not_block_a_later_caller():
    async def scenario():
        ft = FakeTime(start=0)
        gate = asyncio.Event()
        async def blocked_sleep(delay):
            await gate.wait()
        ep = make_ep("google:m:KEY")
        ep.cooldown_until = 0.5
        sched = LLMScheduler([ep], clock=ft.now, asleep=blocked_sleep, max_wait_seconds=1)
        oldest = asyncio.create_task(sched.acquire())
        await asyncio.sleep(0)
        ft.t = 2
        chosen, _ = await sched.acquire()
        assert chosen is ep
        chosen.release()
        gate.set()
        with pytest.raises(AllEndpointsExhausted):
            await oldest
        assert sched._waiters == []
    asyncio.run(scenario())


def test_sync_and_async_callers_share_one_admission_order():
    ft = FakeTime(start=0)
    sync_waiting = threading.Event()
    release_sync = threading.Event()
    ep = make_ep("google:m:KEY", rpm_limit=1)
    assert ep.try_reserve(0)
    ep.release()
    served = []
    failures = []
    def sync_sleep(delay):
        sync_waiting.set()
        assert release_sync.wait(5)
    async def async_sleep(delay):
        # A later async waiter cannot take the slot while sync head is sleeping.
        assert served == []
        release_sync.set()
        await asyncio.to_thread(thread.join, 5)
        ft.t = 120
    sched = LLMScheduler([ep], clock=ft.now, sleep=sync_sleep, asleep=async_sleep, max_wait_seconds=130)
    def worker():
        try:
            chosen, _ = sched.acquire_sync()
            served.append("sync")
            chosen.release()
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    assert sync_waiting.wait(5)
    ft.t = 60
    async def scenario():
        chosen, _ = await sched.acquire()
        served.append("async")
        chosen.release()
    try:
        asyncio.run(scenario())
    finally:
        release_sync.set()
        thread.join(5)
    assert not failures and not thread.is_alive()
    assert served == ["sync", "async"]
    assert sched._waiters == []
