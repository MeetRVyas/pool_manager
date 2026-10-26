"""
Chooses which endpoint serves the next request and bounds how long a caller may wait.

Selection (`strategy="priority"`, the default)
    Endpoints are ordered by their position in llm_config (model priority). The
    best-ranked model that still has headroom is used; among its API keys the
    least-recently-used one is picked, so every key shares the load. A model that is
    at its (YAML) limit, cooling down or failing is skipped, which is what lets the
    pool switch to the next model *before* a rate limit is hit.
    `strategy="balanced"` ignores model order and just spreads load over everything.

Waiting
    If nothing is available the scheduler sleeps exactly until the earliest endpoint
    frees up -- but only if that fits in the wait budget. If the earliest availability
    is later than the budget (e.g. every key is out of daily quota) it raises
    immediately instead of sleeping out the budget for nothing.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Awaitable, Callable, Optional

from pool_manager.endpoint_state import EndpointState

logger = logging.getLogger("pool_manager.scheduler")

STRATEGIES = ("priority", "balanced")


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

    # ----------------------------------------------------------------- selection
    def _order_key(self, ep: EndpointState):
        # Lower sorts first. use_seq (a strict counter) makes ties rotate through the keys (LRU).
        if self.strategy == "priority":
            return (ep.priority, ep.inflight, ep.use_seq)
        return (ep.inflight, ep.use_seq, ep.priority)

    def _try_acquire(self, now: float) -> tuple[Optional[EndpointState], float]:
        """One non-blocking pass. Returns (endpoint, 0) or (None, seconds until one may free up)."""
        ready: list[EndpointState] = []
        soonest = math.inf
        for ep in self.endpoints:
            wait = ep.available_in(now)
            if wait <= 0.0:
                ready.append(ep)
            elif wait < soonest:
                soonest = wait
        for ep in sorted(ready, key=self._order_key):
            if ep.try_reserve(now):  # re-checks atomically; another thread may have won
                return ep, 0.0
        return None, (0.0 if ready else soonest)

    def _poll(self, now: float, deadline: float) -> tuple[Optional[EndpointState], float]:
        ep, wait = self._try_acquire(now)
        if ep is not None:
            return ep, 0.0
        if wait == math.inf or now + wait > deadline + 1e-9:
            raise self._exhausted(now, wait, deadline)
        return None, max(0.001, min(wait, self.poll_interval))

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
    async def acquire(self, max_wait: Optional[float] = None) -> tuple[EndpointState, float]:
        """Reserve an endpoint. Returns (endpoint, seconds waited). Raises AllEndpointsExhausted."""
        budget = self.max_wait_seconds if max_wait is None else max(0.0, max_wait)
        start = self.clock()
        deadline = start + budget
        while True:
            ep, delay = self._poll(self.clock(), deadline)
            if ep is not None:
                return ep, self.clock() - start
            await self._asleep(delay)

    def acquire_sync(self, max_wait: Optional[float] = None) -> tuple[EndpointState, float]:
        budget = self.max_wait_seconds if max_wait is None else max(0.0, max_wait)
        start = self.clock()
        deadline = start + budget
        while True:
            ep, delay = self._poll(self.clock(), deadline)
            if ep is not None:
                return ep, self.clock() - start
            self._sleep(delay)

    @asynccontextmanager
    async def allocate_endpoint(self, max_wait: Optional[float] = None):
        endpoint, _ = await self.acquire(max_wait)
        try:
            yield endpoint
        finally:
            endpoint.release()

    @contextmanager
    def allocate_endpoint_sync(self, max_wait: Optional[float] = None):
        endpoint, _ = self.acquire_sync(max_wait)
        try:
            yield endpoint
        finally:
            endpoint.release()

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
