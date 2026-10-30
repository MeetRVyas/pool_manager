"""Optional caller budget. Sync SDK work cannot be killed; its lease stays held."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
import contextvars
import math
import threading
import time


class CallerDeadlineExceeded(TimeoutError):
    """The caller's total configured budget expired, independently of provider health."""


def validate_timeout(value):
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError("caller_timeout must be a positive finite number or None")
    return value


class Deadline:
    def __init__(self, seconds):
        self.expires = time.monotonic() + validate_timeout(seconds)
        self.cancelled = threading.Event()

    def remaining(self):
        return max(0.0, self.expires - time.monotonic())

    def check(self):
        if self.cancelled.is_set() or self.remaining() <= 0:
            raise CallerDeadlineExceeded("Overall caller deadline exceeded")


def run_sync(call, seconds):
    if seconds is None:
        return call(None)
    deadline = Deadline(seconds)
    result = Future()
    context = contextvars.copy_context()

    def worker():
        try:
            result.set_result(context.run(call, deadline))
        except BaseException as exc:
            result.set_exception(exc)

    # A timed-out SDK call retains its endpoint lease until the worker exits.
    # Daemon avoids interpreter shutdown hanging on an uninterruptible SDK call.
    threading.Thread(target=worker, name="pool-sync-deadline", daemon=True).start()
    try:
        value = result.result(timeout=deadline.remaining())
    except FutureTimeout:
        # Provider TimeoutError can also be stored in Future. Distinguish it from
        # waiting for the worker timing out without consuming the caller budget.
        if result.done() and not deadline.cancelled.is_set() and deadline.remaining() > 0:
            return result.result()
        deadline.cancelled.set()
        raise CallerDeadlineExceeded("Overall caller deadline exceeded") from None
    deadline.check()
    return value


async def run_async(call, seconds):
    if seconds is None:
        return await call(None)
    deadline = Deadline(seconds)
    timer = asyncio.timeout(deadline.remaining())
    try:
        async with timer:
            result = await call(deadline)
            deadline.check()
            return result
    except TimeoutError:
        if timer.expired():
            deadline.cancelled.set()
            raise CallerDeadlineExceeded("Overall caller deadline exceeded") from None
        raise
