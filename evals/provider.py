"""
A fake LLM provider that ENFORCES rate limits the way the real thing does, so any pool
can be measured against it. It raises the REAL exception objects (google.genai errors
wrapped exactly as langchain-google-genai 4.x wraps them), not mocks of them.

What it models
  * per (project, model) quota: per-minute (sliding or fixed window) and per-day,
    shared by every key that belongs to the same project (Google limits are per PROJECT)
  * 429 bodies shaped like Gemini's: quota id (PerMinute / PerDay) and a RetryInfo hint
  * outages (503/500), random background server errors, and revoked keys (HTTP 400 API_KEY_INVALID)
  * the SDK client's own retry loop (google-genai default: up to `max_retries` attempts on
    408/429/5xx, wait = min(60, 2^(n-1) + U(0,1)) s, server retry hint ignored) -- taken
    from the installed google-genai source; the production builder passes max_retries=0
    for the merged pool and the sim honours whatever the builder actually passed.

Assumptions (deliberate, stated so they can be challenged)
  * a request rejected with 429 does not consume quota
  * 429/5xx responses return after `reject_latency`; successes take lognormal latency
"""

from __future__ import annotations

import asyncio
import contextvars
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import langchain_google_genai.chat_models as gm
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage

from evals import vtime

# ----------------------------------------------------------------- per-call attribution
@dataclass
class CallCtx:
    """What the provider did to ONE caller-level call (across every internal retry)."""

    requests: int = 0
    r429: int = 0
    s5xx: int = 0
    auth: int = 0


CALL: contextvars.ContextVar[Optional[CallCtx]] = contextvars.ContextVar("call_ctx", default=None)


# -------------------------------------------------------------------------- topology
@dataclass
class Spec:
    model: str
    key_name: str          # env-var style name, e.g. KEY_1
    api_key: str           # the (fake) secret the client is constructed with
    project: str           # keys of one project share quota
    true_rpm: int
    true_rpd: Optional[int] = None
    outages: tuple = ()    # ((t_start, t_end), ...) virtual seconds
    revoked_from: Optional[float] = None
    p_server_error: float = 0.0

    @property
    def id(self) -> str:
        return f"google:{self.model}:{self.key_name}"


@dataclass
class Attempt:
    status: int
    kind: str                      # ok | rate | quota_day | server | auth
    retry_s: Optional[float] = None
    message: str = ""
    status_name: str = ""


class _Bucket:
    __slots__ = ("accepted", "day_count", "phase", "fixed_idx", "fixed_count")

    def __init__(self, phase: float) -> None:
        self.accepted: deque[float] = deque()
        self.day_count = 0
        self.phase = phase
        self.fixed_idx: Optional[int] = None
        self.fixed_count = 0


_SERVER_STYLES = (
    (503, "UNAVAILABLE", "This model is currently experiencing high demand. Spikes in demand are usually temporary."),
    (503, "UNAVAILABLE", "The model is overloaded. Please try again later."),
    (500, "INTERNAL", "An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting"),
)


class World:
    def __init__(
        self,
        specs: list[Spec],
        *,
        rng: random.Random,
        enforcement: str = "sliding",
        latency_median: float = 1.5,
        latency_sigma: float = 0.5,
        reject_latency: float = 0.15,
    ) -> None:
        if enforcement not in ("sliding", "fixed"):
            raise ValueError(enforcement)
        self.specs = specs
        self.by_id = {s.id: s for s in specs}
        self.by_cred = {(s.model, s.api_key): s for s in specs}
        self.rng = rng
        self.enforcement = enforcement
        self.latency_median = latency_median
        self.latency_sigma = latency_sigma
        self.reject_latency = reject_latency
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self.counters: dict[str, Counter] = {s.id: Counter() for s in specs}

    # ---- quota
    def _bucket(self, spec: Spec) -> _Bucket:
        key = (spec.project, spec.model)
        if key not in self._buckets:
            self._buckets[key] = _Bucket(phase=self.rng.uniform(0, 60))
        return self._buckets[key]

    def _admit(self, spec: Spec, t: float) -> Optional[tuple[float, bool]]:
        """None = accepted (and counted). Otherwise (retry_after_seconds, is_daily)."""
        b = self._bucket(spec)
        if spec.true_rpd is not None and b.day_count >= spec.true_rpd:
            return 30.0, True            # Gemini's hint is unhelpfully short even for daily quota
        if self.enforcement == "sliding":
            while b.accepted and b.accepted[0] <= t - 60.0:
                b.accepted.popleft()
            if len(b.accepted) >= spec.true_rpm:
                return float(math.ceil(b.accepted[0] + 60.0 - t)), False
            b.accepted.append(t)
        else:
            idx = int((t + b.phase) // 60)
            if b.fixed_idx != idx:
                b.fixed_idx, b.fixed_count = idx, 0
            if b.fixed_count >= spec.true_rpm:
                return float(math.ceil((idx + 1) * 60 - (t + b.phase))), False
            b.fixed_count += 1
        b.day_count += 1
        return None

    def _latency(self) -> float:
        return min(30.0, max(0.2, self.latency_median * math.exp(self.rng.gauss(0.0, self.latency_sigma))))

    # ---- one HTTP-level request
    async def attempt(self, spec: Spec) -> Attempt:
        ctx = CALL.get()
        t = vtime.now()
        counts = self.counters[spec.id]
        if ctx:
            ctx.requests += 1

        if spec.revoked_from is not None and t >= spec.revoked_from:
            await asyncio.sleep(self.reject_latency)
            counts["auth"] += 1
            if ctx:
                ctx.auth += 1
            return Attempt(400, "auth", None, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")

        if any(a <= t < b for a, b in spec.outages) or (spec.p_server_error and self.rng.random() < spec.p_server_error):
            code, name, msg = self.rng.choice(_SERVER_STYLES)
            await asyncio.sleep(self.reject_latency * 3)
            counts["5xx"] += 1
            if ctx:
                ctx.s5xx += 1
            return Attempt(code, "server", None, msg, name)

        verdict = self._admit(spec, t)
        if verdict is not None:
            retry_s, daily = verdict
            await asyncio.sleep(self.reject_latency)
            counts["429_day" if daily else "429_min"] += 1
            if ctx:
                ctx.r429 += 1
            return Attempt(429, "quota_day" if daily else "rate", retry_s, "", "RESOURCE_EXHAUSTED")

        counts["accepted"] += 1
        await asyncio.sleep(self._latency())
        return Attempt(200, "ok")


# --------------------------------------------------------------- exceptions, as raised
def build_exception(att: Attempt, model: str, *, legacy: bool = False) -> BaseException:
    """
    The exception a caller sees for a failed attempt.
      legacy=False: as langchain-google-genai 4.x raises it (typed wrapper, `.code` None on 429).
      legacy=True : the raw google.genai ClientError/ServerError that the v3 classifier was written for.
    """
    if att.kind in ("rate", "quota_day"):
        quota = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if att.kind == "quota_day" else "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
        delay = f"{int(att.retry_s or 30)}s"
        body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                          "message": f"You exceeded your current quota, please check your plan and billing details. Please retry in {delay}.",
                          "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota}]},
                                      {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}
        raw: BaseException = ClientError(429, body)
        handler = gm._handle_client_error
    elif att.kind == "server":
        body = {"error": {"code": att.status, "status": att.status_name, "message": att.message}}
        raw = ServerError(att.status, body)
        handler = None
    elif att.kind == "auth":
        body = {"error": {"code": 400, "status": att.status_name, "message": att.message}}
        raw = ClientError(400, body)
        handler = gm._handle_client_error
    else:  # pragma: no cover
        raise ValueError(att.kind)
    if legacy:
        return raw
    try:
        if handler is gm._handle_client_error:
            gm._handle_client_error(raw, {"model": model})
        else:
            gm._handle_server_error(raw)
    except Exception as wrapped:
        return wrapped
    raise AssertionError("wrapper did not raise")  # pragma: no cover


# ------------------------------------------------------------------------- the client
_SDK_RETRY_CODES = (408, 429, 500, 502, 503, 504)


class FakeGemini:
    """
    Drop-in for ChatGoogleGenerativeAI in the pools under test. Constructed with the same
    keyword arguments the production builder passes (model, api_key, max_retries, timeout...).
    `max_retries` follows the real field semantics: attempts = max(1, max_retries); the real
    default (when a builder passes nothing) is 6.
    """

    def __init__(self, *, world: World, model: str, api_key: str, max_retries: int = 6,
                 legacy_errors: bool = False, **ignored: Any) -> None:
        self.world = world
        self.spec = world.by_cred[(model, api_key)]
        self.model = model
        self.max_retries = max_retries
        self.sdk_attempts = max(1, max_retries)
        self.legacy_errors = legacy_errors

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        att: Optional[Attempt] = None
        for n in range(1, self.sdk_attempts + 1):
            att = await self.world.attempt(self.spec)
            if att.kind == "ok":
                return AIMessage(content="ok", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
            if n == self.sdk_attempts or att.status not in _SDK_RETRY_CODES:
                break
            # google-genai: tenacity.wait_exponential_jitter(initial=1, max=60, exp_base=2, jitter=1)
            await asyncio.sleep(min(60.0, 1.0 * 2 ** (n - 1) + self.world.rng.uniform(0, 1.0)))
        assert att is not None
        raise build_exception(att, self.model, legacy=self.legacy_errors)
