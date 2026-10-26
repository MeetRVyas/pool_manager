"""
Decide what to do about an exception raised by a provider client.

Every provider/SDK version wraps errors differently (e.g. langchain-google-genai
re-raises a Gemini 429 as `GoogleRateLimitError` whose own `.code` is None and
keeps the real `google.genai.errors.ClientError` as `__cause__`). So instead of
isinstance-checking one class, we walk the whole exception chain, look for an
HTTP status, and only then fall back to type names / message text.

Outcome (`Kind`):

    RATE_LIMIT  429 per-minute style      -> cool this endpoint down (honouring Retry-After), fail over
    QUOTA       429 per-day / no credit   -> cool down until the daily reset, fail over
    TRANSIENT   5xx, 408, timeouts, network errors -> short exponential backoff, fail over
    DEAD        401/403/404, invalid key  -> quarantine this endpoint for a long time, fail over
    FATAL       everything else (400 bad request, parse errors, bugs) -> raise to the caller unchanged
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional


class Kind(str, Enum):
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    TRANSIENT = "transient"
    DEAD = "dead"
    FATAL = "fatal"


@dataclass(frozen=True)
class Classified:
    kind: Kind
    status: Optional[int] = None
    retry_after: Optional[float] = None  # seconds, if the server told us
    detail: str = ""

    @property
    def failover(self) -> bool:
        """True if trying a different endpoint can help."""
        return self.kind is not Kind.FATAL


# --------------------------------------------------------------------------- helpers
_MAX_CHAIN = 8
_TEXT_LIMIT = 4000

# langchain-core >= 1.x ships typed model errors; older versions don't -> optional.
try:  # pragma: no cover - depends on installed langchain-core
    from langchain_core import exceptions as _lc

    _LC_STATUS_HINTS: tuple[tuple[type, int], ...] = tuple(
        (getattr(_lc, name), status)
        for name, status in (
            ("ModelRateLimitError", 429),
            ("ModelAuthenticationError", 401),
            ("ModelPermissionDeniedError", 403),
            ("ModelNotFoundError", 404),
            ("ModelAPIError", 500),
        )
        if hasattr(_lc, name)
    )
    _LC_TRANSIENT_TYPES: tuple[type, ...] = tuple(
        getattr(_lc, name) for name in ("ModelTimeoutError", "ModelConnectionError") if hasattr(_lc, name)
    )
except Exception:  # pragma: no cover
    _LC_STATUS_HINTS = ()
    _LC_TRANSIENT_TYPES = ()

# Matched by class name anywhere in the MRO so we don't need httpx/requests/aiohttp imports.
_TRANSIENT_TYPE_NAMES = frozenset(
    {
        "TimeoutError", "Timeout", "TimeoutException", "ConnectTimeout", "ReadTimeout",
        "WriteTimeout", "PoolTimeout", "APITimeoutError", "DeadlineExceeded",
        "ConnectionError", "ConnectError", "APIConnectionError", "ReadError", "WriteError",
        "RemoteProtocolError", "ProtocolError", "NetworkError", "TransportError",
        "ServerDisconnectedError", "ClientConnectionError", "ChunkedEncodingError",
        "SSLError", "IncompleteRead",
    }
)

# Deliberately strict: Gemini's ordinary per-minute 429 text says "check your plan and
# billing details", so words like "billing"/"daily" must NOT count as a daily-quota signal.
# Genuine daily exhaustion names the quota id (GenerateRequestsPerDay...) or, for OpenAI
# style APIs, the `insufficient_quota` error code.
_DAILY_QUOTA_RE = re.compile(r"per[\s_-]?day|insufficient_quota", re.I)
_INVALID_KEY_RE = re.compile(
    r"api[\s_-]?key[\s_-]?(?:not[\s_-]?valid|invalid|expired|not[\s_-]?found)|API_KEY_INVALID|"
    r"invalid[\s_-]?api[\s_-]?key|incorrect api key",
    re.I,
)

_DURATION_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
# `(?![a-z])` instead of `\b`: in "6m0s" there is no word boundary between "m" and "0".
_DURATION_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)(?![a-z])", re.I)
_RETRY_TEXT_RES = (
    # google: "Please retry in 34.07s."   openai: "Please try again in 6m0s" / "in 20ms"
    re.compile(r"(?:retry|try again)\s+(?:after|in)\s+((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h)\s*)+)", re.I),
    # google RetryInfo inside the error details:  'retryDelay': '34s'
    re.compile(r"retryDelay['\"]?\s*:\s*['\"]?((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h)\s*)+)", re.I),
)


def parse_duration(text: str) -> Optional[float]:
    """'34.07s' -> 34.07, '6m0s' -> 360, '20ms' -> 0.02, '1h2m' -> 3720. None if unparsable."""
    tokens = _DURATION_TOKEN_RE.findall(text or "")
    if not tokens:
        return None
    return sum(float(num) * _DURATION_UNIT[unit.lower()] for num, unit in tokens)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    stack = [exc]
    while stack and len(seen) < _MAX_CHAIN:
        cur = stack.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        stack.extend([cur.__cause__, cur.__context__])


def _status_of(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "http_status", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    for lc_type, status in _LC_STATUS_HINTS:
        if isinstance(exc, lc_type):
            return status
    return None


def _headers_of(exc: BaseException) -> Any:
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None)


def _retry_after_from_headers(headers: Any) -> Optional[float]:
    if not headers:
        return None
    try:
        getter = headers.get
        ms = getter("retry-after-ms") or getter("Retry-After-Ms")
        if ms is not None:
            return max(0.0, float(ms) / 1000.0)
        value = getter("retry-after") or getter("Retry-After")
    except Exception:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:  # HTTP-date form
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        when = parsedate_to_datetime(str(value))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _retry_after(chain: list[BaseException], text: str) -> Optional[float]:
    for exc in chain:
        hint = _retry_after_from_headers(_headers_of(exc))
        if hint is not None:
            return hint
    for regex in _RETRY_TEXT_RES:
        match = regex.search(text)
        if match:
            seconds = parse_duration(match.group(1))
            if seconds is not None:
                return seconds
    return None


def _is_transient_type(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if _LC_TRANSIENT_TYPES and isinstance(exc, _LC_TRANSIENT_TYPES):
        return True
    return any(cls.__name__ in _TRANSIENT_TYPE_NAMES for cls in type(exc).__mro__)


def _describe(chain: list[BaseException]) -> str:
    inner = chain[-1] if len(chain) > 1 else chain[0]
    msg = str(inner).replace("\n", " ")
    return f"{type(inner).__name__}: {msg[:200]}"


# ------------------------------------------------------------------------ classifier
def classify(exc: BaseException) -> Classified:
    chain = list(_chain(exc))
    text = " | ".join(str(e) for e in chain)[:_TEXT_LIMIT]
    detail = _describe(chain)

    status = next((s for s in (_status_of(e) for e in chain) if s is not None), None)

    if status is not None:
        retry_after = _retry_after(chain, text)
        if status == 429:
            kind = Kind.QUOTA if _DAILY_QUOTA_RE.search(text) else Kind.RATE_LIMIT
            return Classified(kind, status, retry_after, detail)
        if status in (401, 403, 404):
            return Classified(Kind.DEAD, status, None, detail)
        if status == 408 or 500 <= status <= 599:
            return Classified(Kind.TRANSIENT, status, retry_after, detail)
        if status in (400, 422) and _INVALID_KEY_RE.search(text):
            # Google answers a bad key with HTTP 400 / API_KEY_INVALID.
            return Classified(Kind.DEAD, status, None, detail)
        return Classified(Kind.FATAL, status, None, detail)

    if any(_is_transient_type(e) for e in chain):
        return Classified(Kind.TRANSIENT, None, None, detail)
    return Classified(Kind.FATAL, None, None, detail)
