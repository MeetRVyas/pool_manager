"""
Chooses which endpoint serves the next request and bounds how long a caller may wait.

Selection (`strategy="priority"`, the default)
    Endpoints are ordered by their position in llm_config (model priority). The
    best-ranked model that still has headroom is used; among its API keys the
    least-recently-used one is picked, so every key shares the load. A model that is
    at its (YAML) limit, cooling down or failing is skipped, which is what lets the
    pool switch to the next model *before* a rate limit is hit.
    `strategy="balanced"` ignores model order and just spreads load over everything.
    Retries can supply attempted endpoint IDs: untried eligible endpoints then take
    precedence, with the configured strategy ordering endpoints within each group.

Waiting
    Sync and async callers share FIFO admission. Only the oldest live waiter may
    reserve an endpoint. Queue advancement signals the next waiter immediately;
    the head otherwise sleeps until known quota/cooldown reopening. Waiting still
    respects the per-acquisition budget, and impossible quota waits fail early.
    A retry joins at the back as a new acquisition, retaining endpoint history.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import threading
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from typing import Awaitable, Callable, Optional

from pool_manager.endpoint_state import EndpointState

logger = logging.getLogger("pool_manager.scheduler")

STRATEGIES = ("priority", "balanced")


@dataclass(eq=False)
class _Waiter:
    deadline: float
    checked: bool = False
    loop: Optional[asyncio.AbstractEventLoop] = None
    signal: Optional[threading.Event] = field(default=None, repr=False)
    ready: Optional[asyncio.Future] = field(default=None, repr=False)


class AllEndpointsExhausted(Exception):
    """No endpoint can serve the request within the wait budget.

    `retry_after` is how long (seconds) until the earliest endpoint is expected to be
    usable again, or None if unknown -- callers can use it to decide whether to wait.
    """

    def __init__(self, message: str = "All endpoints are exhausted", *, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AllAttemptsFailed(AllEndpointsExhausted):
    """Every failover attempt for one call failed with a retryable error.

    Subclasses AllEndpointsExhausted so existing `except AllEndpointsExhausted` handlers
    keep working. `__cause__` is the last underlying provider exception.
    """


class LLMScheduler:
    def __init__(
        self,
        endpoints: list[EndpointState],
        max_wait_seconds: float = 60.0,
        poll_interval: float = 1.0,
        strategy: str = "priority",
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        self.endpoints = list(endpoints)
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval = poll_interval
        self.strategy = strategy
        self.clock = clock
        self._sleep = sleep
        self._asleep = asleep
        # One queue for sync threads and async tasks. Never hold this lock across
        # provider calls, sleeps or awaits. Endpoint preference is unchanged.
        self._admission_lock = threading.Lock()
        self._waiters: list[_Waiter] = []

    def _join(self, deadline: float, loop=None) -> _Waiter:
        waiter = _Waiter(deadline, loop=loop)
        with self._admission_lock:
            self._waiters.append(waiter)
        return waiter

    def _leave(self, waiter: _Waiter) -> None:
        with self._admission_lock:
            if waiter in self._waiters:
                was_head = self._waiters[0] is waiter
                self._waiters.remove(waiter)
                if was_head:
                    self._notify_head_locked()

    @staticmethod
    def _resolve(ready: asyncio.Future) -> None:
        if not ready.done():
            ready.set_result(None)

    def _notify_head_locked(self) -> None:
        if not self._waiters:
            return
        head = self._waiters[0]
        if head.signal is not None:
            head.signal.set()
        # Capture this generation's future, not whatever the waiter creates later.
        # Always deliver on its owning loop, including sync-to-async handoffs.
        if head.loop is not None and head.ready is not None:
            try:
                head.loop.call_soon_threadsafe(self._resolve, head.ready)
            except RuntimeError:
                if not head.loop.is_closed():
                    raise

    def notify_changed(self) -> None:
        """Recheck admission after endpoint state changes outside acquisition.

        Pool outcome/release paths call this. Integrations changing quotas or
        endpoint state directly should do so too; it never grants quota itself.
        """
        with self._admission_lock:
            self._notify_head_locked()

    def _arm_locked(self, waiter: _Waiter) -> None:
        # Checking eligibility and arming share the notification lock: a change
        # between the check and actual sleep cannot be lost. No objects are
        # allocated on the uncontended successful admission path.
        if waiter.loop is not None:
            waiter.ready = waiter.loop.create_future()
        else:
            if waiter.signal is None:
                waiter.signal = threading.Event()
            waiter.signal.clear()

    async def _wait_async(self, waiter: _Waiter, delay: float) -> None:
        ready = waiter.ready
        sleeper = asyncio.ensure_future(self._asleep(delay))
        try:
            done, _ = await asyncio.wait((ready, sleeper), return_when=asyncio.FIRST_COMPLETED)
            if sleeper in done:
                sleeper.result()  # Do not hide errors from injected clocks/sleep hooks.
        finally:
            if ready is not None:
                ready.cancel()
            sleeper.cancel()
            await asyncio.gather(sleeper, return_exceptions=True)

    def _wait_sync(self, waiter: _Waiter, delay: float) -> None:
        if self._sleep is time.sleep:
            waiter.signal.wait(delay)
        else:
            # Retain injectable virtual-clock sleeps. They cannot be interrupted
            # like Event.wait, so keep the compatibility polling bound for them.
            self._sleep(min(delay, self.poll_interval))

    def _admit(self, waiter: _Waiter, now: float, attempted):
        with self._admission_lock:
            # A delayed/cancelled task must not retain an expired queue position.
            old_head = self._waiters[0] if self._waiters else None
            self._waiters[:] = [w for w in self._waiters if not w.checked or w.deadline + 1e-9 >= now]
            if self._waiters and self._waiters[0] is not old_head:
                self._notify_head_locked()
            if waiter not in self._waiters:
                wait = min((ep.available_in(now) for ep in self.endpoints), default=math.inf)
                raise self._exhausted(now, wait, waiter.deadline)
            waiter.checked = True
            if self._waiters[0] is waiter:
                endpoint, delay = self._poll(now, waiter.deadline, attempted)
                if endpoint is not None:
                    self._waiters.remove(waiter)
                    self._notify_head_locked()
                else:
                    self._arm_locked(waiter)
                return endpoint, delay
            wait = min((ep.available_in(now) for ep in self.endpoints), default=math.inf)
            remaining = waiter.deadline - now
            if remaining <= 0 or wait == math.inf or now + wait > waiter.deadline + 1e-9:
                raise self._exhausted(now, wait, waiter.deadline)
            # A follower needs no polling timer; head advancement signals it.
            # Its own deadline still bounds the wait if the head never resumes.
            self._arm_locked(waiter)
            return None, remaining

    # ----------------------------------------------------------------- selection
    def _order_key(self, ep: EndpointState):
        # Lower sorts first. use_seq (a strict counter) makes ties rotate through the keys (LRU).
        if self.strategy == "priority":
            return (ep.priority, ep.inflight, ep.use_seq)
        return (ep.inflight, ep.use_seq, ep.priority)

    def _try_acquire(self, now: float, attempted: Optional[set[str]] = None) -> tuple[Optional[EndpointState], float]:
        """One non-blocking pass. Returns (endpoint, 0) or (None, seconds until one may free up)."""
        ready: list[EndpointState] = []
        soonest = math.inf
        for ep in self.endpoints:
            wait = ep.available_in(now)
            if wait <= 0.0:
                ready.append(ep)
            elif wait < soonest:
                soonest = wait
        attempted = attempted or set()
        for ep in sorted(ready, key=lambda e: (e.id in attempted, self._order_key(e))):
            if ep.try_reserve(now):  # re-checks atomically; another thread may have won
                return ep, 0.0
        return None, (0.0 if ready else soonest)

    def _poll(self, now: float, deadline: float, attempted: Optional[set[str]] = None) -> tuple[Optional[EndpointState], float]:
        ep, wait = self._try_acquire(now, attempted)
        if ep is not None:
            return ep, 0.0
        if wait == math.inf or now + wait > deadline + 1e-9:
            raise self._exhausted(now, wait, deadline)
        return None, max(0.001, wait)

    def _exhausted(self, now: float, wait: float, deadline: float) -> AllEndpointsExhausted:
        if not self.endpoints:
            return AllEndpointsExhausted("No endpoints are configured.")
        budget = max(0.0, deadline - now)
        blockers = []
        for ep in sorted(self.endpoints, key=lambda e: e.available_in(now))[:8]:
            why = ep.cooldown_kind if ep.cooldown_remaining(now) > 0 else "at configured limit"
            blockers.append(f"{ep.id} (free in {ep.available_in(now):.0f}s, {why})")
        if wait == math.inf:
            head = "No endpoint will become available."
        else:
            head = (
                f"No endpoint available within the remaining {budget:.0f}s wait budget "
                f"(earliest is free in {wait:.0f}s)."
            )
        return AllEndpointsExhausted(
            f"{head} {len(self.endpoints)} endpoints configured; soonest: " + "; ".join(blockers),
            retry_after=None if wait == math.inf else wait,
        )

    # ------------------------------------------------------------------- acquire
    async def acquire(self, max_wait: Optional[float] = None, *, attempted: Optional[set[str]] = None) -> tuple[EndpointState, float]:
        """Reserve an endpoint, preferring untried eligible IDs. Returns (endpoint, seconds waited)."""
        budget = self.max_wait_seconds if max_wait is None else max(0.0, max_wait)
        start = self.clock()
        deadline = start + budget
        waiter = self._join(deadline, asyncio.get_running_loop())
        try:
            while True:
                ep, delay = self._admit(waiter, self.clock(), attempted)
                if ep is not None:
                    return ep, self.clock() - start
                await self._wait_async(waiter, delay)
        finally:
            self._leave(waiter)

    def acquire_sync(self, max_wait: Optional[float] = None, *, attempted: Optional[set[str]] = None) -> tuple[EndpointState, float]:
        budget = self.max_wait_seconds if max_wait is None else max(0.0, max_wait)
        start = self.clock()
        deadline = start + budget
        waiter = self._join(deadline)
        try:
            while True:
                ep, delay = self._admit(waiter, self.clock(), attempted)
                if ep is not None:
                    return ep, self.clock() - start
                self._wait_sync(waiter, delay)
        finally:
            self._leave(waiter)

    @asynccontextmanager
    async def allocate_endpoint(self, max_wait: Optional[float] = None):
        endpoint, _ = await self.acquire(max_wait)
        try:
            yield endpoint
        finally:
            endpoint.release()
            self.notify_changed()

    @contextmanager
    def allocate_endpoint_sync(self, max_wait: Optional[float] = None):
        endpoint, _ = self.acquire_sync(max_wait)
        try:
            yield endpoint
        finally:
            endpoint.release()
            self.notify_changed()

    # ------------------------------------------------------------- observability
    def stats(self) -> list[dict]:
        now = self.clock()
        return [ep.snapshot(now) for ep in self.endpoints]

    def unused_endpoints(self) -> list[str]:
        """Ids of endpoints that have never been reserved (handy for checking all keys get used)."""
        return [ep.id for ep in self.endpoints if ep.calls == 0]

    def usage_report(self) -> str:
        rows = self.stats()
        if not rows:
            return "(no endpoints)"
        head = f"{'endpoint':<46} {'calls':>5} {'ok':>5} {'429':>4} {'quota':>5} {'5xx/net':>7} {'dead':>4}  {'rpm':>9} {'rpd':>11}  state"
        lines = [head, "-" * len(head)]
        for r in rows:
            rpm = f"{r['rpm_used']}/{r['rpm_limit']}" if r["rpm_limit"] else f"{r['rpm_used']}/-"
            rpd = f"{r['rpd_used']}/{r['rpd_limit']}" if r["rpd_limit"] else "-"
            state = (
                f"{r['cooldown_kind']} {r['cooldown_remaining_s']:.0f}s"
                if r["cooldown_kind"]
                else ("limit-wait %.0fs" % r["available_in_s"] if r["available_in_s"] > 0 else "ready")
            )
            lines.append(
                f"{r['id']:<46} {r['calls']:>5} {r['successes']:>5} {r['rate_limited']:>4} "
                f"{r['quota_hits']:>5} {r['server_errors']:>7} {r['dead_errors']:>4}  {rpm:>9} {rpd:>11}  {state}"
            )
        unused = self.unused_endpoints()
        lines.append(
            f"\n{len(rows) - len(unused)}/{len(rows)} endpoints used"
            + (f"; never used: {', '.join(unused)}" if unused else "")
        )
        return "\n".join(lines)
