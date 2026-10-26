"""
Virtual time for deterministic load simulation.

`VirtualLoop` is an asyncio event loop whose clock only moves when everything is idle:
instead of really sleeping until the next timer it jumps straight to it. A ten-minute
rate-limit scenario (60 s windows, 33 s SDK back-offs, 120 s waits) therefore runs in
milliseconds, while keeping REAL asyncio concurrency semantics: tasks interleave exactly
as they would in production, only the waiting is free.

Everything that waits via `asyncio.sleep` (the pool, the fake provider, the SDK-retry
model) is virtualised automatically. Code that reads a clock must be given `now()`:
the merged pool takes `scheduler.clock`, and the baseline v3 pool reads `datetime.now()`,
which `virtual_datetime()` redirects to this clock while the baseline is imported.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
from typing import Any, Awaitable, Callable, Iterator, Optional

_ACTIVE: Optional["VirtualLoop"] = None
_EPOCH = _dt.datetime(2026, 1, 1)


class VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._vt = 0.0
        real_select = self._selector.select

        def select(timeout: Optional[float] = None):
            if timeout is None:
                # Nothing scheduled and nothing ready: in a closed simulation that is a deadlock.
                raise RuntimeError("virtual loop deadlock: no timers and no ready callbacks")
            if timeout > 0:
                self._vt += timeout          # jump to the next timer instead of sleeping
            return real_select(0)

        self._selector.select = select  # type: ignore[method-assign]

    def time(self) -> float:
        return self._vt


def now() -> float:
    """Current virtual time in seconds (0.0 outside a simulation)."""
    return _ACTIVE.time() if _ACTIVE is not None else 0.0


def run_virtual(main: Callable[[], Awaitable[Any]]) -> Any:
    """Run `main()` to completion on a fresh virtual-time loop."""
    global _ACTIVE
    loop = VirtualLoop()
    _ACTIVE = loop
    try:
        return loop.run_until_complete(main())
    finally:
        _ACTIVE = None
        loop.close()


class _VirtualDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz: Any = None):  # noqa: D401 - mimics datetime.now
        return _EPOCH + _dt.timedelta(seconds=now())


@contextlib.contextmanager
def virtual_datetime() -> Iterator[None]:
    """
    While active, `from datetime import datetime` yields a class whose `now()` reads virtual
    time. Import the (unmodified) v3 baseline inside this block: it binds `datetime.now` in
    dataclass `default_factory`s at import time, so it must be imported while patched.
    """
    real = _dt.datetime
    _dt.datetime = _VirtualDatetime  # type: ignore[misc]
    try:
        yield
    finally:
        _dt.datetime = real  # type: ignore[misc]
