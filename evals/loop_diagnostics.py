"""Opt-in loop/HTTP diagnostics. No provider calls; event-file writes use a worker.

Adapted from the reviewed files.zip diagnostic. No outbound headers are added.
Debug/logging/DNS/hooks are restored on exit. C traceback scheduling is process
global and enabled only with an explicit stack_dumps=True opt-in.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
import contextvars
import faulthandler
import inspect
import itertools
import json
import logging
from pathlib import Path
import sys
import threading
import time
import traceback

CALL = contextvars.ContextVar("diagnostic_call", default=None)
_DUMP_OWNER = threading.Lock()
_TIMING_HEADERS = ("server-timing", "x-envoy-upstream-service-time", "x-gemini-service-tier")


def backend_metadata(client):
    """Inspect already-constructed clients without creating clients or requests."""
    sdk = getattr(client, "client", None)
    api = getattr(sdk, "_api_client", None)
    aiohttp_active = bool(api._use_aiohttp()) if callable(getattr(api, "_use_aiohttp", None)) else False
    backend = getattr(api, "_async_httpx_client", None)
    if aiohttp_active:
        backend = None
    if backend is None:
        backend = getattr(getattr(client, "async_client", None), "_client", None)
    return backend, {"provider_client": f"{type(client).__module__}.{type(client).__qualname__}",
                     "aiohttp_active": aiohttp_active,
                     "http_backend": None if backend is None else
                     f"{type(backend).__module__}.{type(backend).__qualname__}"}


class _QueuedLog(logging.Handler):
    def __init__(self, emit):
        super().__init__()
        self.emit_event = emit

    def emit(self, record):
        self.emit_event("asyncio_slow", message=record.getMessage())


class LoopDiagnostics:
    def __init__(self, folder, *, origin=None, beat_s=0.05, lag_s=0.25,
                 sample_s=0.5, debug=True, stack_dumps=False, dump_s=2.0, secrets=()):
        if min(beat_s, lag_s, sample_s, dump_s) <= 0:
            raise ValueError("Diagnostic intervals must be positive")
        self.folder = Path(folder)
        self.origin = time.monotonic() if origin is None else origin
        self.beat_s, self.lag_s, self.sample_s = beat_s, lag_s, sample_s
        self.debug, self.stack_dumps, self.dump_s = debug, stack_dumps, dump_s
        self.secrets = tuple(secrets)
        self.events = deque()
        self.lag_windows = []
        self._rid = itertools.count(1)
        self._stop = threading.Event()
        self._hooks = []
        self._task = self._thread = self._dump_file = None
        self._owns_dump = False
        self._started = False
        self.writer_error = None

    def emit(self, kind, **fields):
        self.events.append(dict(t=time.monotonic() - self.origin, kind=kind, call=CALL.get(), **fields))

    async def start(self):
        if self._started:
            raise RuntimeError("Diagnostics already started")
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._last_beat = time.monotonic()
        self._old_dns = self._loop.getaddrinfo
        self._old_debug = self._loop.get_debug()
        self._old_slow = self._loop.slow_callback_duration
        log = logging.getLogger("asyncio")
        self._old_log = (list(log.handlers), log.level, log.propagate)
        self._started = True
        try:
            await asyncio.to_thread(self.folder.mkdir, parents=True, exist_ok=True)
            if self.stack_dumps:
                if not _DUMP_OWNER.acquire(blocking=False):
                    raise RuntimeError("Another diagnostic owns process-global traceback scheduling")
                self._owns_dump = True
                self._dump_file = await asyncio.to_thread(
                    (self.folder / "faulthandler.log").open, "a", encoding="utf-8")
            self._loop.getaddrinfo = self._timed_dns
            if self.debug:
                self._loop.set_debug(True)
                self._loop.slow_callback_duration = self.lag_s
                log.handlers = [_QueuedLog(self.emit)]
                log.setLevel(logging.WARNING)
                log.propagate = False
            self._thread = threading.Thread(target=self._watch, daemon=True, name="pool-loop-diagnostics")
            self._thread.start()
            self._task = self._loop.create_task(self._beat())
            self.emit("start", loop_type=type(self._loop).__name__, beat_s=self.beat_s,
                      lag_s=self.lag_s, debug=self.debug, stack_dumps=self.stack_dumps)
        except BaseException:
            await self.aclose()
            raise
        return self

    async def aclose(self):
        if not self._started:
            return
        self._started = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        for client, req, resp in self._hooks:
            for kind, hook in (("request", req), ("response", resp)):
                if hook in client.event_hooks[kind]:
                    client.event_hooks[kind].remove(hook)
        self._hooks.clear()
        self._loop.getaddrinfo = self._old_dns
        self._loop.set_debug(self._old_debug)
        self._loop.slow_callback_duration = self._old_slow
        log = logging.getLogger("asyncio")
        if self.debug:
            for handler in log.handlers:
                if isinstance(handler, _QueuedLog):
                    handler.close()
            log.handlers, level, log.propagate = self._old_log
            log.setLevel(level)
        if self._owns_dump:
            faulthandler.cancel_dump_traceback_later()
            if self._dump_file is not None:
                await asyncio.to_thread(self._dump_file.close)
            _DUMP_OWNER.release()
            self._owns_dump = False
        self.emit("stop")
        self._stop.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join)
        if self.writer_error:
            raise RuntimeError("Diagnostic writer failed") from self.writer_error
        path = self.folder / "loop_diag.jsonl"
        if path.exists():
            await asyncio.to_thread(write_diagnostic_summary, path)

    def attach_httpx(self, client, tag):
        async def on_request(request):
            rid = next(self._rid)
            request.extensions["pool_diag_rid"] = rid
            previous_trace = request.extensions.get("trace")

            async def trace(name, info):
                # Never serialize info: it may contain headers or credentials.
                self.emit("http", rid=rid, tag=tag, phase=name)
                if previous_trace is not None:
                    result = previous_trace(name, info)
                    if inspect.isawaitable(result):
                        await result

            request.extensions["trace"] = trace
            self.emit("http", rid=rid, tag=tag, phase="request_hook", host=request.url.host)

        async def on_response(response):
            self.emit("http", rid=response.request.extensions.get("pool_diag_rid"), tag=tag,
                      phase="response_hook", status=response.status_code,
                      timing={k: response.headers[k] for k in _TIMING_HEADERS if k in response.headers})

        client.event_hooks["request"].append(on_request)
        client.event_hooks["response"].append(on_response)
        self._hooks.append((client, on_request, on_response))

    async def _timed_dns(self, *args, **kwargs):
        before = time.monotonic()
        try:
            return await self._old_dns(*args, **kwargs)
        finally:
            self.emit("dns", duration_s=time.monotonic() - before)

    async def _beat(self):
        expected = time.monotonic() + self.beat_s
        armed = 0.0
        while True:
            await asyncio.sleep(self.beat_s)
            now = time.monotonic()
            late = now - expected
            if late >= self.lag_s:
                self.lag_windows.append((expected - self.origin, now - self.origin))
                self.emit("loop_lag", late_s=late)
            self._last_beat = now
            expected = now + self.beat_s
            if self._owns_dump and now - armed >= min(0.5, self.dump_s / 2):
                faulthandler.dump_traceback_later(self.dump_s, file=self._dump_file)
                armed = now

    def overlaps(self, start, end):
        return any(a < end and b > start for a, b in self.lag_windows)

    def _watch(self):
        tick = max(0.005, min(self.sample_s / 5, self.lag_s / 4))
        previous = time.monotonic()
        wall0, mono0, drift0 = time.time(), previous, 0.0
        sampled = previous
        try:
            with (self.folder / "loop_diag.jsonl").open("a", encoding="utf-8") as output:
                while not self._stop.wait(tick):
                    now = time.monotonic()
                    if now - previous - tick >= self.lag_s:
                        self.emit("watchdog_late", late_s=now - previous - tick)
                    previous = now
                    drift = time.time() - wall0 - (now - mono0)
                    if abs(drift - drift0) >= 0.5:
                        self.emit("clock_jump", wall_minus_monotonic_s=drift - drift0)
                    drift0 = drift
                    if now - self._last_beat >= self.lag_s + self.beat_s and now - sampled >= self.sample_s:
                        frame = sys._current_frames().get(self._loop_thread)
                        stack = traceback.extract_stack(frame)[-8:] if frame is not None else []
                        self.emit("stall", stale_s=now - self._last_beat,
                                  stack=[f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in stack])
                        sampled = now
                    self._drain(output)
                self._drain(output)
        except BaseException as exc:
            self.writer_error = exc

    def _drain(self, output):
        while self.events:
            line = json.dumps(self.events.popleft(), ensure_ascii=True)
            for secret in self.secrets:
                line = line.replace(secret, "[REDACTED]")
            output.write(line + "\n")
        output.flush()


def timing_summary(rows):
    """Preserve invocation metrics; additionally report successful arrival latency."""
    from evals.metrics import percentile
    ok = [r for r in rows if r["outcome"] == "ok"]
    def quantiles(values):
        return {f"p{p}_s": percentile(values, p) for p in (50, 95, 99)}
    return {
        "invocation_success": quantiles([r["latency_s"] for r in ok]),
        "scheduled_arrival_success": quantiles([r["scheduled_to_result_s"] for r in ok]),
        "dispatch_lateness_all": quantiles([r["dispatch_lateness_s"] for r in rows]),
        "loop_stall_overlap_calls": sum(r.get("overlaps_loop_stall") is True for r in rows),
        "loop_stall_measurement_enabled": any(r.get("overlaps_loop_stall") is not None for r in rows),
    }


def write_diagnostic_summary(path):
    """Run off-loop after the writer exits; accept HTTP/1.1 and HTTP/2 phase names."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    requests = {}
    windows = [(r["t"] - r["late_s"], r["t"]) for r in rows if r["kind"] == "loop_lag"]
    for row in rows:
        if row["kind"] != "http":
            continue
        rid = row["rid"]
        item = requests.setdefault(rid, {"rid": rid, "call": row.get("call"), "endpoint": row.get("tag"),
                                         "start_s": row["t"], "phases_s": {}, "events": []})
        item["events"].append(row["phase"])
        item["end_s"] = row["t"]
        if row["phase"] == "response_hook":
            item.update(status=row["status"], timing_headers=row["timing"])
    for rid, item in requests.items():
        started = {}
        for row in rows:
            if row["kind"] != "http" or row.get("rid") != rid:
                continue
            phase, _, state = row["phase"].rpartition(".")
            if state == "started":
                started[phase] = row["t"]
            elif state in ("complete", "failed") and phase in started:
                item["phases_s"][phase] = row["t"] - started[phase]
        item["overlaps_loop_stall"] = any(a < item["end_s"] and b > item["start_s"] for a,b in windows)
    result = {"loop_lag": [r for r in rows if r["kind"] == "loop_lag"],
              "watchdog_late": [r for r in rows if r["kind"] == "watchdog_late"],
              "clock_jumps": [r for r in rows if r["kind"] == "clock_jump"],
              "stall_samples": [r for r in rows if r["kind"] == "stall"],
              "requests": list(requests.values())}
    path.with_name("diagnostics.summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
