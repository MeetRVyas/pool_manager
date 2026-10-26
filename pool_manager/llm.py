from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Type, TypeVar, Union

import yaml
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from pool_manager.benchmark import CallRecord, get_tracker
from pool_manager.endpoint_state import CooldownPolicy, EndpointState
from pool_manager.errors import Classified, Kind, classify
from pool_manager.limits import LimitsCatalogue, resolve_limits
from pool_manager.scheduler import (
    AllAttemptsFailed,
    AllEndpointsExhausted,
    LLMScheduler,
)

__all__ = [
    "AllAttemptsFailed",
    "AllEndpointsExhausted",
    "MultiProviderChatLLM",
    "build_endpoints",
    "describe_endpoints",
    "get_llm",
]

logger = logging.getLogger("pool_manager")

T = TypeVar("T")

DEFAULT_LIMITS_PATH = "configs/model_limits.yaml"
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_MAX_ATTEMPTS = 6


# ============================================================================ build
def _load_catalogue(limits_path: Union[str, Path, None, LimitsCatalogue]) -> Optional[LimitsCatalogue]:
    if isinstance(limits_path, LimitsCatalogue):
        return limits_path
    if limits_path is None:  # explicitly disabled
        return None
    if limits_path == "auto":
        path = Path(DEFAULT_LIMITS_PATH)
        if not path.exists():
            logger.warning(
                "No model limits file at %s: running WITHOUT proactive rate limits "
                "(only reactive handling of 429s). Pass limits_path=... to set one.",
                path,
            )
            return None
        return LimitsCatalogue.from_file(path)
    return LimitsCatalogue.from_file(limits_path)  # explicit path: must exist


def _as_list(value: Any, provider: str, field: str) -> list:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"llm_config['{provider}'] needs a non-empty list for '{field}'")
    return list(value)


def build_endpoints(
    config: Union[str, Path],
    temperature: float = 0.0,
    *,
    limits_path: Union[str, Path, None, LimitsCatalogue] = "auto",
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    safety_margin: float = 0.0,
    cooldowns: Optional[CooldownPolicy] = None,
) -> list[EndpointState]:
    """
    One endpoint per (provider, model, API key). `config` is a YAML file, per provider:

        google:                       # provider label; "google"/"gemini" use the Gemini client,
          models: [m1, m2]            #   anything else is treated as OpenAI-compatible
          keys: [ENV_VAR_1, ENV_VAR_2]  # names of ENV VARS holding the keys (not the keys)
          base_url: ...               # optional
          timeout: 120                # optional, per-request seconds (default: request_timeout)
          rpm_limit / rpd_limit / tpm_limit: int   # optional provider-wide defaults

    Per-model limits come from the catalogue (`limits_path`, see limits.py) and win over
    the provider-wide defaults. A model with no limits anywhere is not pre-limited.

    Endpoints are ordered by preference: the order of providers/models in the file, so
    `models: [primary, fallback]` means "use primary while it has headroom".
    """
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"{config}: expected a mapping of provider -> settings")

    catalogue = _load_catalogue(limits_path)
    cooldowns = cooldowns or CooldownPolicy()
    endpoints: list[EndpointState] = []
    priority = 0

    for provider, params in cfg.items():
        if not isinstance(params, dict):
            raise ValueError(f"llm_config['{provider}'] must be a mapping")
        models = list(dict.fromkeys(_as_list(params.get("models"), provider, "models")))
        key_names = _as_list(params.get("keys"), provider, "keys")
        is_gemini = str(provider).lower() in ("google", "gemini")
        timeout = params.get("timeout", request_timeout)

        # Resolve every key up front; fail loudly if one is missing, and use identical key
        # values once (two env vars holding the same key add no quota but would double-count).
        resolved: list[tuple[str, str]] = []
        seen: dict[str, str] = {}
        for key_name in key_names:
            api_key = os.environ.get(key_name)
            if not api_key:
                raise ValueError(f"Environment variable '{key_name}' is not set")
            if api_key in seen:
                logger.warning(
                    "llm_config['%s']: env vars %s and %s hold the same API key; using it once.",
                    provider, seen[api_key], key_name,
                )
                continue
            seen[api_key] = key_name
            resolved.append((key_name, api_key))

        for model in models:
            limits = resolve_limits(provider, model, catalogue, params).with_margin(safety_margin)
            for key_name, api_key in resolved:
                client_kwargs: Dict[str, Any] = {
                    "model": model,
                    "api_key": api_key,
                    # The pool does the retrying/failover. The SDKs' own retry loops (Gemini
                    # defaults to 6 attempts with exponential backoff, OpenAI to 2) would sit
                    # on a 429 for tens of seconds and hide it from the pool.
                    "max_retries": 0,
                    "timeout": timeout,
                }
                if params.get("base_url"):
                    client_kwargs["base_url"] = params["base_url"]
                if not is_gemini:
                    # Gemini is deliberately left at its own default temperature (as in v3).
                    client_kwargs["temperature"] = temperature

                endpoints.append(
                    EndpointState(
                        id=f"{provider}:{model}:{key_name}",
                        provider=provider,
                        model=model,
                        client=ChatGoogleGenerativeAI(**client_kwargs) if is_gemini else ChatOpenAI(**client_kwargs),
                        rpm_limit=limits.rpm,
                        rpd_limit=limits.rpd,
                        tpm_limit=limits.tpm,
                        daily_reset=limits.daily_reset,
                        limits_source=limits.source,
                        priority=priority,
                        cooldowns=cooldowns,
                    )
                )
            priority += 1

    if not endpoints:
        raise ValueError(f"{config}: no endpoints were built")
    return endpoints


def describe_endpoints(endpoints: list[EndpointState]) -> str:
    """Human-readable table: which limits each endpoint was given and where they came from."""
    lines = [f"{'#':>2}  {'endpoint':<46} limits  (source)"]
    for ep in endpoints:
        lim = " ".join(
            part
            for part in (
                f"rpm={ep.rpm_limit}" if ep.rpm_limit else "",
                f"tpm={ep.tpm_limit}" if ep.tpm_limit else "",
                f"rpd={ep.rpd_limit}({ep.daily_reset})" if ep.rpd_limit else "",
            )
            if part
        ) or "none -> reactive only"
        lines.append(f"{ep.priority:>2}  {ep.id:<46} {lim}  ({ep.limits_source})")
    return "\n".join(lines)


# ======================================================================== failover
def _phase_from_config(config: Optional[RunnableConfig]) -> str:
    return ((config or {}).get("metadata") or {}).get("phase", "unknown")


def _phase_from_run_manager(run_manager: Any) -> str:
    return (getattr(run_manager, "metadata", None) or {}).get("phase", "unknown")


def _usage_of_message(message: Any) -> dict:
    return getattr(message, "usage_metadata", None) or {}


def _usage_of_structured(result: Any) -> dict:
    # include_raw=True results look like {"raw": AIMessage, "parsed": ..., "parsing_error": ...}
    if isinstance(result, dict):
        return getattr(result.get("raw"), "usage_metadata", None) or {}
    return {}


def _record_success(ep: EndpointState, phase: str, latency_s: float, usage: dict, now: float) -> None:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total = usage.get("total_tokens") or ((input_tokens or 0) + (output_tokens or 0))
    ep.mark_success(now, total or None)
    try:  # telemetry must never break a call
        get_tracker().record(
            CallRecord(
                phase=phase,
                provider=ep.provider,
                model=ep.model,
                latency_s=latency_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                endpoint_id=ep.id,
            )
        )
    except Exception:  # pragma: no cover
        logger.debug("benchmark tracking failed", exc_info=True)


def _apply_failure(ep: EndpointState, info: Classified, now: float) -> str:
    """Update the endpoint for a retryable failure; returns a short summary for the attempt log."""
    if info.kind is Kind.RATE_LIMIT:
        ep.mark_rate_limited(now, info.retry_after, detail=info.detail)
    elif info.kind is Kind.QUOTA:
        ep.mark_rate_limited(now, info.retry_after, quota_exhausted=True, detail=info.detail)
    elif info.kind is Kind.TRANSIENT:
        ep.mark_server_error(now, info.retry_after, detail=info.detail)
    else:  # Kind.DEAD
        ep.mark_dead(now, detail=info.detail)
    status = f" HTTP {info.status}" if info.status else ""
    logger.warning(
        "%s: %s%s -> sitting out %.0fs, failing over", ep.id, info.kind.value, status, ep.cooldown_remaining(now)
    )
    return f"{ep.id}: {info.kind.value}{status}"


def _failed(tried: list[str], max_attempts: int) -> AllAttemptsFailed:
    return AllAttemptsFailed(f"All {max_attempts} attempts failed -> " + "; ".join(tried))


def run_with_failover_sync(
    scheduler: LLMScheduler,
    call: Callable[[EndpointState], T],
    *,
    max_attempts: int,
    phase: str,
    usage_of: Callable[[Any], dict],
) -> T:
    """
    Run `call(endpoint)`; on a retryable failure (429, quota, 5xx, timeout, dead key) mark
    the endpoint and retry on another one. Non-retryable errors (bad request, parse errors)
    are raised untouched. Bounded by `max_attempts` and by the scheduler's wait budget.
    """
    max_attempts = max(1, max_attempts)
    last_exc: Optional[BaseException] = None
    tried: list[str] = []
    waited_total = 0.0
    for _ in range(max_attempts):
        try:
            ep, waited = scheduler.acquire_sync(max(0.0, scheduler.max_wait_seconds - waited_total))
        except AllEndpointsExhausted as exc:
            if last_exc is not None:
                raise exc from last_exc  # keep the provider error that led here
            raise
        waited_total += waited
        started = time.monotonic()
        try:
            result = call(ep)
        except Exception as exc:
            info = classify(exc)
            if not info.failover:
                raise
            tried.append(_apply_failure(ep, info, scheduler.clock()))
            last_exc = exc
            continue
        else:
            _record_success(ep, phase, time.monotonic() - started, _safe_usage(usage_of, result), scheduler.clock())
            return result
        finally:
            ep.release()
    raise _failed(tried, max_attempts) from last_exc


async def run_with_failover_async(
    scheduler: LLMScheduler,
    call: Callable[[EndpointState], Awaitable[T]],
    *,
    max_attempts: int,
    phase: str,
    usage_of: Callable[[Any], dict],
) -> T:
    """Async twin of run_with_failover_sync."""
    max_attempts = max(1, max_attempts)
    last_exc: Optional[BaseException] = None
    tried: list[str] = []
    waited_total = 0.0
    for _ in range(max_attempts):
        try:
            ep, waited = await scheduler.acquire(max(0.0, scheduler.max_wait_seconds - waited_total))
        except AllEndpointsExhausted as exc:
            if last_exc is not None:
                raise exc from last_exc
            raise
        waited_total += waited
        started = time.monotonic()
        try:
            result = await call(ep)
        except Exception as exc:
            info = classify(exc)
            if not info.failover:
                raise
            tried.append(_apply_failure(ep, info, scheduler.clock()))
            last_exc = exc
            continue
        else:
            _record_success(ep, phase, time.monotonic() - started, _safe_usage(usage_of, result), scheduler.clock())
            return result
        finally:
            ep.release()
    raise _failed(tried, max_attempts) from last_exc


def _safe_usage(usage_of: Callable[[Any], dict], result: Any) -> dict:
    try:
        return usage_of(result) or {}
    except Exception:  # pragma: no cover
        return {}


# ======================================================================== runnables
class _MultiProviderToolBoundRunnable(Runnable[LanguageModelInput, BaseMessage]):
    """
    Runnable returned by MultiProviderChatLLM.bind_tools().

    Tool binding is deferred until invocation because MultiProviderChatLLM does not
    represent one concrete provider/model. Each invocation selects an endpoint and lets
    that endpoint's client perform its native tool binding.
    """

    def __init__(
        self,
        scheduler: LLMScheduler,
        tools: list[Any],
        *,
        tool_choice: Any = None,
        bind_kwargs: Optional[Dict[str, Any]] = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.scheduler = scheduler
        self.tools = tools
        self.tool_choice = tool_choice
        self.bind_kwargs = bind_kwargs or {}
        self.max_attempts = max_attempts

    def _bind_endpoint(self, ep: EndpointState):
        kwargs = dict(self.bind_kwargs)
        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice
        return ep.client.bind_tools(self.tools, **kwargs)

    def invoke(self, input: LanguageModelInput, config: Optional[RunnableConfig] = None, **kwargs: Any) -> BaseMessage:
        return run_with_failover_sync(
            self.scheduler,
            lambda ep: self._bind_endpoint(ep).invoke(input, config=config, **kwargs),
            max_attempts=self.max_attempts,
            phase=_phase_from_config(config),
            usage_of=_usage_of_message,
        )

    async def ainvoke(
        self, input: LanguageModelInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> BaseMessage:
        async def call(ep: EndpointState):
            return await self._bind_endpoint(ep).ainvoke(input, config=config, **kwargs)

        return await run_with_failover_async(
            self.scheduler,
            call,
            max_attempts=self.max_attempts,
            phase=_phase_from_config(config),
            usage_of=_usage_of_message,
        )


class _MultiProviderStructuredRunnable(Runnable[LanguageModelInput, Union[Dict, BaseModel]]):
    """
    Runnable returned by MultiProviderChatLLM.with_structured_output().

    MultiProviderChatLLM picks a (provider, model, key) endpoint fresh on every call, so
    structured output cannot be resolved once at bind time. Each call allocates an
    endpoint and defers to *that endpoint's* own `.with_structured_output(...)`, so each
    provider uses its native mechanism; this class does no schema translation.

    include_raw=True is always requested from the endpoint so the raw message's
    usage_metadata is available for benchmarking; the result is then unwrapped to
    exactly the shape the caller asked for.

    Caveat: if your pool mixes providers, `**kwargs` passed to with_structured_output
    (e.g. `method="json_mode"`) must be valid for every provider in the pool.
    """

    def __init__(
        self,
        scheduler: LLMScheduler,
        schema: Optional[Union[Dict, Type[BaseModel]]],
        *,
        include_raw: bool = False,
        structured_kwargs: Optional[Dict[str, Any]] = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.scheduler = scheduler
        self.schema = schema
        self.include_raw = include_raw
        self.structured_kwargs = structured_kwargs or {}
        self.max_attempts = max_attempts

    def _structured(self, ep: EndpointState):
        return ep.client.with_structured_output(self.schema, include_raw=True, **self.structured_kwargs)

    def _unwrap(self, raw_result: Any) -> Any:
        if self.include_raw:
            return raw_result
        if raw_result.get("parsing_error") is not None:
            raise raw_result["parsing_error"]  # a parse failure is not an endpoint failure
        return raw_result["parsed"]

    def invoke(
        self, input: LanguageModelInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Union[Dict, BaseModel]:
        raw_result = run_with_failover_sync(
            self.scheduler,
            lambda ep: self._structured(ep).invoke(input, config=config, **kwargs),
            max_attempts=self.max_attempts,
            phase=_phase_from_config(config),
            usage_of=_usage_of_structured,
        )
        return self._unwrap(raw_result)

    async def ainvoke(
        self, input: LanguageModelInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Union[Dict, BaseModel]:
        async def call(ep: EndpointState):
            return await self._structured(ep).ainvoke(input, config=config, **kwargs)

        raw_result = await run_with_failover_async(
            self.scheduler,
            call,
            max_attempts=self.max_attempts,
            phase=_phase_from_config(config),
            usage_of=_usage_of_structured,
        )
        return self._unwrap(raw_result)


class MultiProviderChatLLM(BaseChatModel):
    scheduler: LLMScheduler
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "multi_provider_chat"

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        call_kwargs = dict(kwargs)
        if stop is not None:
            call_kwargs["stop"] = stop

        async def call(ep: EndpointState):
            return await ep.client.ainvoke(messages, **call_kwargs)

        response = await run_with_failover_async(
            self.scheduler,
            call,
            max_attempts=self.max_attempts,
            phase=_phase_from_run_manager(run_manager),
            usage_of=_usage_of_message,
        )
        # Return the endpoint's message as-is so tool_calls / usage / metadata survive.
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        call_kwargs = dict(kwargs)
        if stop is not None:
            call_kwargs["stop"] = stop

        response = run_with_failover_sync(
            self.scheduler,
            lambda ep: ep.client.invoke(messages, **call_kwargs),
            max_attempts=self.max_attempts,
            phase=_phase_from_run_manager(run_manager),
            usage_of=_usage_of_message,
        )
        return ChatResult(generations=[ChatGeneration(message=response)])

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Structured output matching `schema` (Pydantic model, TypedDict or JSON schema),
        same contract as LangChain's with_structured_output. Supports invoke()/batch()
        and ainvoke()/abatch(). Each call independently allocates an endpoint, so calls
        (and failover retries within a call) may land on different providers/models/keys.
        """
        return _MultiProviderStructuredRunnable(
            scheduler=self.scheduler,
            schema=schema,
            include_raw=include_raw,
            structured_kwargs=kwargs,
            max_attempts=self.max_attempts,
        )

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools dynamically: the endpoint is chosen at invocation time, not here."""
        return _MultiProviderToolBoundRunnable(
            scheduler=self.scheduler,
            tools=tools,
            tool_choice=tool_choice,
            bind_kwargs=kwargs,
            max_attempts=self.max_attempts,
        )

    # ---- observability
    def stats(self) -> list[dict]:
        """Per-endpoint counters (calls, 429s, errors, limit usage, cooldowns)."""
        return self.scheduler.stats()

    def usage_report(self) -> str:
        """Text table of how every (model, key) endpoint was used -- see which keys got traffic."""
        return self.scheduler.usage_report()


def get_llm(
    config: Union[str, Path],
    temperature: float = 0.0,
    max_wait_seconds: float = 60.0,
    *,
    limits_path: Union[str, Path, None, LimitsCatalogue] = "auto",
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    strategy: str = "priority",
    safety_margin: float = 0.0,
    cooldowns: Optional[CooldownPolicy] = None,
) -> MultiProviderChatLLM:
    """
    Build the pool from a YAML llm_config.

    max_wait_seconds  Most time one call may spend *waiting for an endpoint to free up*
                      (across all its retries). If the earliest endpoint is free later than
                      that, the call fails immediately with AllEndpointsExhausted.
    request_timeout   Per-request HTTP timeout, so a hung connection can't stall a call.
    max_attempts      Most endpoints one call will try before giving up (AllAttemptsFailed).
    limits_path       Model-limits YAML ("auto" = configs/model_limits.yaml if present,
                      None = disabled, or a path / LimitsCatalogue).
    strategy          "priority" (model order in config, keys balanced) or "balanced".
    safety_margin     0.1 = pre-stop at 90% of every configured limit.
    """
    endpoints = build_endpoints(
        config,
        temperature=temperature,
        limits_path=limits_path,
        request_timeout=request_timeout,
        safety_margin=safety_margin,
        cooldowns=cooldowns,
    )
    scheduler = LLMScheduler(endpoints, max_wait_seconds=max_wait_seconds, strategy=strategy)
    logger.info("LLM pool ready: %d endpoints\n%s", len(endpoints), describe_endpoints(endpoints))
    return MultiProviderChatLLM(scheduler=scheduler, max_attempts=max_attempts)
