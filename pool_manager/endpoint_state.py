"""
State of one (provider, model, API key) endpoint: how much of its rate limit is
used, whether it is cooling down, and counters for observability.

All time arguments (`now`) are monotonic seconds supplied by the scheduler, which
keeps this class deterministic and testable. Every public method is thread-safe.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Optional

from pool_manager.limits import seconds_until_daily_reset

MINUTE = 60.0
DAY = 86_400.0
_MIN_WAIT = 1e-9  # smallest wait a blocked window may report (see _SlidingWindow.time_until_room)

# Strictly increasing ticket used to rotate through keys (least-recently-used first).
# A timestamp would not do: coarse clocks (e.g. ~15 ms on Windows) give equal values for
# back-to-back reservations, and ties would always go to the first endpoint in the list.
_USE_SEQ = itertools.count()


class _SlidingWindow:
    """Weighted events inside the last `span` seconds (requests: weight 1, tokens: weight n)."""

    __slots__ = ("span", "_events", "_total")

    def __init__(self, span: float) -> None:
        self.span = span
        self._events: Deque[tuple[float, int]] = deque()
        self._total = 0

    def _expire(self, now: float) -> None:
        cutoff = now - self.span
        while self._events and self._events[0][0] <= cutoff:
            self._total -= self._events.popleft()[1]

    def used(self, now: float) -> int:
        self._expire(now)
        return self._total

    def add(self, now: float, weight: int = 1) -> None:
        self._events.append((now, weight))
        self._total += weight

    def time_until_room(self, now: float, limit: int) -> float:
        """Seconds until usage drops below `limit` (0 if there is room right now)."""
        self._expire(now)
        if self._total < limit:
            return 0.0
        total = self._total
        for ts, weight in self._events:
            total -= weight
            if total < limit:
                # We only get here when _expire() just decided this entry is still inside the
                # window, so the answer must be strictly positive. `ts + span - now` is NOT
                # float-equivalent to `_expire`'s `ts <= now - span` and can round to 0.0,
                # which would admit a request into a still-full window.
                return max(ts + self.span - now, _MIN_WAIT)
        return self.span  # unreachable for sane inputs; be conservative


@dataclass(frozen=True)
class CooldownPolicy:
    """How long an endpoint sits out after each kind of failure (seconds)."""

    rate_limit_default: float = 60.0   # 429 with no Retry-After: one full per-minute window
    rate_limit_min: float = 1.0
    rate_limit_max: float = 3600.0
    retry_after_pad: float = 1.0       # added to a server-provided Retry-After
    quota_default: float = 3600.0      # daily quota hit and no calendar reset known: probe hourly
    server_error_base: float = 5.0     # doubles per consecutive failure ...
    server_error_max: float = 120.0    # ... up to this
    quarantine: float = 3600.0         # 401/403/404/invalid key: retry in an hour, in case it was fixed


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(eq=False)  # identity semantics: hashable, and the lock is never compared
class EndpointState:
    id: str
    provider: str
    model: str
    client: Any  # a LangChain chat model in production, any stub in tests

    # Limits for THIS key. None = no proactive limit on that dimension.
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    tpm_limit: Optional[int] = None
    daily_reset: str = "rolling"          # rolling | midnight_pt | midnight_utc
    limits_source: str = "none (reactive handling only)"

    priority: int = 0                      # lower = preferred (position in llm_config)
    cooldowns: CooldownPolicy = field(default_factory=CooldownPolicy)
    wall_clock: Callable[[], datetime] = field(default=_utcnow, repr=False)

    # ---- runtime state
    inflight: int = 0
    use_seq: int = -1                      # ticket of the last reservation (LRU balancing across keys)
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    cooldown_kind: str = ""                # why we are (or last were) cooling down

    # ---- counters for stats()/usage_report()
    calls: int = 0
    successes: int = 0
    rate_limited: int = 0
    quota_hits: int = 0
    server_errors: int = 0
    dead_errors: int = 0
    last_error: str = ""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._minute = _SlidingWindow(MINUTE)
        self._tokens = _SlidingWindow(MINUTE)
        self._daily_rolling = _SlidingWindow(DAY)   # used when daily_reset == "rolling"
        self._daily_count = 0                       # used for calendar resets
        self._daily_reset_at: Optional[float] = None

    # ------------------------------------------------------------ daily window
    def _roll_calendar_day(self, now: float) -> None:
        if self.daily_reset == "rolling":
            return
        if self._daily_reset_at is None or now >= self._daily_reset_at:
            if self._daily_reset_at is not None:
                self._daily_count = 0
            secs = seconds_until_daily_reset(self.daily_reset, self.wall_clock())
            self._daily_reset_at = now + (secs if secs is not None else DAY)

    def _daily_used(self, now: float) -> int:
        if self.daily_reset == "rolling":
            return self._daily_rolling.used(now)
        self._roll_calendar_day(now)
        return self._daily_count

    def _daily_wait(self, now: float) -> float:
        if self.rpd_limit is None:
            return 0.0
        if self.daily_reset == "rolling":
            return self._daily_rolling.time_until_room(now, self.rpd_limit)
        self._roll_calendar_day(now)
        if self._daily_count < self.rpd_limit:
            return 0.0
        return max(0.0, (self._daily_reset_at or now) - now)

    def _seconds_to_calendar_reset(self, now: float) -> Optional[float]:
        if self.daily_reset == "rolling":
            return None
        self._roll_calendar_day(now)
        return max(0.0, (self._daily_reset_at or now) - now)

    # ------------------------------------------------------------ availability
    def _available_in(self, now: float) -> float:
        waits = [self.cooldown_until - now]
        if self.rpm_limit is not None:
            waits.append(self._minute.time_until_room(now, self.rpm_limit))
        if self.tpm_limit is not None:
            waits.append(self._tokens.time_until_room(now, self.tpm_limit))
        waits.append(self._daily_wait(now))
        return max(0.0, *waits)

    def available_in(self, now: float) -> float:
        """Seconds until this endpoint may take a request (0.0 = available now)."""
        with self._lock:
            return self._available_in(now)

    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self.cooldown_until - now)

    def try_reserve(self, now: float) -> bool:
        """Atomically check limits and, if there is room, count one request against them."""
        with self._lock:
            if self._available_in(now) > 0.0:
                return False
            self.inflight += 1
            self.calls += 1
            self.use_seq = next(_USE_SEQ)
            self._minute.add(now)
            if self.daily_reset == "rolling":
                if self.rpd_limit is not None:
                    self._daily_rolling.add(now)
            else:
                self._roll_calendar_day(now)
                self._daily_count += 1
            return True

    def release(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)

    # --------------------------------------------------------------- outcomes
    def mark_success(self, now: float, tokens: Optional[int] = None) -> None:
        with self._lock:
            self.successes += 1
            self.consecutive_failures = 0
            if tokens and self.tpm_limit is not None:
                self._tokens.add(now, int(tokens))

    def _cool(self, now: float, seconds: float, kind: str, detail: str) -> float:
        self.cooldown_until = max(self.cooldown_until, now + seconds)
        self.cooldown_kind = kind
        if detail:
            self.last_error = f"{kind}: {detail}"
        return seconds

    def mark_rate_limited(
        self,
        now: Optional[float] = None,
        retry_after: Optional[float] = None,
        *,
        quota_exhausted: bool = False,
        detail: str = "",
    ) -> float:
        """The provider answered 429. Returns the cooldown applied (seconds)."""
        now = time.monotonic() if now is None else now
        cp = self.cooldowns
        with self._lock:
            if quota_exhausted:
                self.quota_hits += 1
                until_reset = self._seconds_to_calendar_reset(now)
                if until_reset is not None:
                    secs = until_reset + cp.retry_after_pad
                else:
                    secs = max(retry_after or 0.0, cp.quota_default)
                return self._cool(now, secs, "quota", detail)
            self.rate_limited += 1
            if retry_after is not None:
                secs = retry_after + cp.retry_after_pad
            else:
                secs = cp.rate_limit_default
            secs = min(max(secs, cp.rate_limit_min), cp.rate_limit_max)
            return self._cool(now, secs, "rate_limit", detail)

    def mark_server_error(
        self, now: float, retry_after: Optional[float] = None, *, detail: str = ""
    ) -> float:
        """5xx / timeout / network failure: exponential backoff, reset by the next success."""
        cp = self.cooldowns
        with self._lock:
            self.server_errors += 1
            self.consecutive_failures += 1
            secs = min(cp.server_error_max, cp.server_error_base * 2 ** (self.consecutive_failures - 1))
            if retry_after is not None:
                secs = min(cp.server_error_max, max(secs, retry_after))
            return self._cool(now, secs, "server_error", detail)

    def mark_dead(self, now: float, *, detail: str = "") -> float:
        """401/403/404 or invalid key: this key/model combination is unusable for a while."""
        with self._lock:
            self.dead_errors += 1
            return self._cool(now, self.cooldowns.quarantine, "dead", detail)

    # -------------------------------------------------------------- reporting
    def snapshot(self, now: float) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "provider": self.provider,
                "model": self.model,
                "priority": self.priority,
                "calls": self.calls,
                "successes": self.successes,
                "rate_limited": self.rate_limited,
                "quota_hits": self.quota_hits,
                "server_errors": self.server_errors,
                "dead_errors": self.dead_errors,
                "inflight": self.inflight,
                "rpm_used": self._minute.used(now),
                "rpm_limit": self.rpm_limit,
                "tpm_used": self._tokens.used(now),
                "tpm_limit": self.tpm_limit,
                "rpd_used": self._daily_used(now),
                "rpd_limit": self.rpd_limit,
                "cooldown_remaining_s": round(max(0.0, self.cooldown_until - now), 1),
                "cooldown_kind": self.cooldown_kind if self.cooldown_until > now else "",
                "available_in_s": round(self._available_in(now), 1),
                "limits_source": self.limits_source,
                "last_error": self.last_error,
            }
