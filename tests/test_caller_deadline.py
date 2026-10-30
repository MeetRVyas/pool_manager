"""Caller deadline behavior with real pool/scheduler and fake providers only."""
import asyncio
import threading
import time

import pytest

from pool_manager.deadline import CallerDeadlineExceeded
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler
from tests.fakes import FakeClient, make_ep, gemini_server_error


def selected(pool, path):
    if path == "tools":
        return pool.bind_tools([{"name": "lookup"}])
    if path == "structured":
        return pool.with_structured_output({"title": "answer", "type": "object"})
    return pool


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("path", ["plain", "tools", "structured"])
def test_deadline_preserves_success_and_retry_paths(mode, path):
    bad = make_ep("google:primary:KEY", FakeClient("bad", [gemini_server_error(503, "UNAVAILABLE", "retry")]))
    good = make_ep("google:fallback:KEY", FakeClient("good"), priority=1)
    pool = MultiProviderChatLLM(scheduler=LLMScheduler([bad, good]), caller_timeout=2)
    runnable = selected(pool, path)
    result = runnable.invoke("test") if mode == "sync" else asyncio.run(runnable.ainvoke("test"))
    assert result is not None
    assert bad.calls == good.calls == 1
    assert bad.inflight == good.inflight == 0
    assert good.successes == 1 and bad.server_errors == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("path", ["plain", "tools", "structured"])
def test_deadline_cancels_async_or_returns_without_releasing_active_sync_sdk_work(mode, path):
    entered, finish, released = threading.Event(), threading.Event(), threading.Event()
    class Blocked(FakeClient):
        def _next(self):
            self.calls += 1
            entered.set()
            assert finish.wait(5)
        async def ainvoke(self, *args, **kwargs):
            self.calls += 1
            entered.set()
            await asyncio.Event().wait()
        def bind_tools(self, *args, **kwargs):
            return self
        def with_structured_output(self, *args, **kwargs):
            return self
    client = Blocked("blocked")
    ep = make_ep("google:m:KEY", client)
    release = ep.release
    releases = []
    def counted_release():
        releases.append(1)
        release()
        released.set()
    ep.release = counted_release
    healthy = make_ep("google:fallback:KEY", FakeClient("unused"), priority=1)
    pool = MultiProviderChatLLM(scheduler=LLMScheduler([ep, healthy]), caller_timeout=0.08)
    runnable = selected(pool, path)
    try:
        with pytest.raises(CallerDeadlineExceeded):
            if mode == "sync":
                runnable.invoke("test")
            else:
                asyncio.run(runnable.ainvoke("test"))
        assert entered.is_set()
        assert healthy.calls == 0  # caller expiry is not an endpoint error or a retry
        assert ep.server_errors == 0 and ep.successes == 0
        if mode == "sync":
            assert ep.inflight == 1 and releases == []
        else:
            assert ep.inflight == 0 and len(releases) == 1
    finally:
        finish.set()
        assert released.wait(5)
    assert ep.inflight == 0 and len(releases) == 1
    assert healthy.calls == 0


def test_deadline_cancels_queued_acquisition_without_reserving_quota():
    async def scenario():
        sleeping = asyncio.Event()
        gate = asyncio.Event()
        async def wait(delay):
            sleeping.set()
            await gate.wait()
        ep = make_ep("google:m:KEY")
        ep.cooldown_until = time.monotonic() + 0.01
        sched = LLMScheduler([ep], asleep=wait)
        oldest = asyncio.create_task(sched.acquire())
        await sleeping.wait()
        ep.cooldown_until = 0
        pool = MultiProviderChatLLM(scheduler=sched, caller_timeout=0.03)
        try:
            with pytest.raises(CallerDeadlineExceeded):
                await pool.ainvoke("queued")
            assert ep.calls == 0 and ep.inflight == 0
            assert len(sched._waiters) == 1
        finally:
            oldest.cancel()
            with pytest.raises(asyncio.CancelledError):
                await oldest
        assert sched._waiters == []
    asyncio.run(scenario())


def test_external_async_cancellation_keeps_its_identity_and_releases_once():
    async def scenario():
        entered = asyncio.Event()
        class Blocked(FakeClient):
            async def ainvoke(self, *args, **kwargs):
                entered.set()
                await asyncio.Event().wait()
        ep = make_ep("google:m:KEY", Blocked("waiting"))
        pool = MultiProviderChatLLM(scheduler=LLMScheduler([ep]), caller_timeout=2)
        task = asyncio.create_task(pool.ainvoke("test"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ep.inflight == 0 and ep.server_errors == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadlines_rejected(timeout):
    with pytest.raises(ValueError):
        MultiProviderChatLLM(scheduler=LLMScheduler([]), caller_timeout=timeout)


def test_deadline_outcome_is_not_misreported_as_a_provider_5xx():
    from evals.sim import outcome_of
    assert outcome_of(CallerDeadlineExceeded("expired")) == "deadline_exceeded"
