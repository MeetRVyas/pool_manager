"""Offline diagnostic lifecycle, HTTP tracing and injected loop-blocking tests."""
import asyncio
import json
import logging
import time

import httpx
import pytest

from evals.loop_diagnostics import CALL, LoopDiagnostics, timing_summary


def test_timing_boundaries_do_not_hide_dispatch_lateness():
    result = timing_summary([dict(outcome="ok", latency_s=2, scheduled_to_result_s=9,
                                  dispatch_lateness_s=7, overlaps_loop_stall=None)])
    assert result["invocation_success"]["p95_s"] == 2
    assert result["scheduled_arrival_success"]["p95_s"] == 9
    assert result["dispatch_lateness_all"]["p95_s"] == 7
    assert result["loop_stall_measurement_enabled"] is False


def test_mock_http_tracing_restores_hooks_logger_dns_and_debug(tmp_path):
    prior_phases = []
    async def previous_trace(name, info):
        prior_phases.append(name)
    async def respond(request):
        assert "x-diag-rid" not in request.headers
        assert "authorization" not in request.headers
        trace = request.extensions["trace"]
        await trace("http11.receive_response_headers.started", {"secret": "private-key"})
        await trace("http11.receive_response_headers.complete", {})
        return httpx.Response(200, headers={"server-timing": "test;dur=1"}, text="ok")
    async def scenario():
        loop = asyncio.get_running_loop()
        log = logging.getLogger("asyncio")
        before = (loop.getaddrinfo, loop.get_debug(), loop.slow_callback_duration,
                  list(log.handlers), log.level, log.propagate)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            old_hooks = {k:list(v) for k,v in client.event_hooks.items()}
            diag = LoopDiagnostics(tmp_path, secrets=["private-key"])
            await diag.start()
            try:
                diag.attach_httpx(client, "fake-endpoint")
                token = CALL.set(42)
                try:
                    reply = await client.get("https://offline.invalid", extensions={"trace": previous_trace})
                    assert reply.status_code == 200
                finally:
                    CALL.reset(token)
                diag.emit("test", detail="private-key")
            finally:
                await diag.aclose()
            assert client.event_hooks == old_hooks
            assert not diag._thread.is_alive()
        after = (loop.getaddrinfo, loop.get_debug(), loop.slow_callback_duration,
                 list(log.handlers), log.level, log.propagate)
        assert after == before
        await diag.aclose()  # idempotent, no cancellation of unrelated dump scheduling
    asyncio.run(scenario())
    text = (tmp_path / "loop_diag.jsonl").read_text(encoding="utf-8")
    assert "private-key" not in text and "[REDACTED]" in text
    records = [json.loads(line) for line in text.splitlines()]
    assert any(r.get("call") == 42 and r.get("status") == 200 for r in records)
    assert len(prior_phases) == 2
    summary = json.loads((tmp_path / "diagnostics.summary.json").read_text())
    assert summary["requests"][0]["call"] == 42
    assert "http11.receive_response_headers" in summary["requests"][0]["phases_s"]


def test_injected_block_is_detected_without_completion_spread_threshold(tmp_path):
    async def scenario():
        diag = LoopDiagnostics(tmp_path, beat_s=0.005, lag_s=0.025, sample_s=0.01)
        await diag.start()
        try:
            await asyncio.sleep(0.03)  # heartbeat armed before injecting the fault
            start = time.monotonic() - diag.origin
            time.sleep(0.15)  # intentional loop-blocking syscall, no provider/network
            await asyncio.sleep(0.04)
            end = time.monotonic() - diag.origin
            assert diag.overlaps(start, end)
        finally:
            await diag.aclose()
    asyncio.run(scenario())
    records = [json.loads(line) for line in (tmp_path / "loop_diag.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "loop_lag" and r["late_s"] >= 0.1 for r in records)
    assert any(r["kind"] == "stall" for r in records)


def test_diagnostics_cleanup_on_caller_cancellation(tmp_path):
    async def scenario():
        diag = LoopDiagnostics(tmp_path)
        await diag.start()
        async def cancelled():
            try:
                await asyncio.Event().wait()
            finally:
                await diag.aclose()
        task = asyncio.create_task(cancelled())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not diag._thread.is_alive()
        assert diag._task.done()
    asyncio.run(scenario())


def test_c_tracebacks_require_explicit_opt_in_and_release_ownership(tmp_path, monkeypatch):
    import faulthandler
    from evals.loop_diagnostics import _DUMP_OWNER
    actions = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: actions.append("armed"))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: actions.append("cancelled"))
    async def scenario():
        disabled = LoopDiagnostics(tmp_path / "disabled", debug=False)
        await disabled.start()
        await disabled.aclose()
        assert actions == []
        enabled = LoopDiagnostics(tmp_path / "enabled", beat_s=0.005, debug=False, stack_dumps=True)
        await enabled.start()
        await asyncio.sleep(0.03)
        await enabled.aclose()
        assert enabled._dump_file.closed
        assert _DUMP_OWNER.acquire(blocking=False)
        _DUMP_OWNER.release()
    asyncio.run(scenario())
    assert "armed" in actions and actions[-1] == "cancelled"
