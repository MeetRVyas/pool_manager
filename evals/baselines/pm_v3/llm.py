from __future__ import annotations
from typing import Any, Dict, Optional, Type, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from openai import RateLimitError
from pydantic import BaseModel


try:
    from google.genai.errors import ClientError as GoogleGenAIClientError, ServerError as GoogleGenAIServerError
except ImportError:  # pragma: no cover
    class ChatGoogleGenerativeAIError(Exception):  # type: ignore[no-redef]
        pass
    class GoogleGenAIClientError(Exception):  # type: ignore[no-redef]
        pass
    class GoogleGenAIServerError(Exception):  # type: ignore[no-redef]
        pass

from evals.baselines.pm_v3.endpoint_state import EndpointState
from evals.baselines.pm_v3.scheduler import AllEndpointsExhausted, LLMScheduler

from pathlib import Path
import yaml
import os


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    True if `exc` represents a 429 / rate-limit response from any provider
    client this pool might use.
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, (GoogleGenAIClientError, ChatGoogleGenerativeAIError)):
        return getattr(exc, "code", None) == 429
    if isinstance(exc, GoogleGenAIServerError):
        return getattr(exc, "code", None) == 503 and "high demand" in getattr(exc, "message", "")
    return False


def build_endpoints(config: str | Path, temperature: float = 0.0) -> list[EndpointState]:
    """
    config shape, per provider:
        {
            "models": [...],
            "keys": [...],
            "base_url": "...",        # optional
            "rpm_limit": int,         # required — requests/minute for this provider's models
            "rpd_limit": int,         # required — requests/day for this provider's models
        }
    """
    endpoints: list[EndpointState] = []

    with open(config) as f:
        config = yaml.safe_load(f)

    for provider, params in config.items():
        is_gemini = provider.lower() in ("google", "gemini")
        if "rpm_limit" not in params or "rpd_limit" not in params:
            raise ValueError(
                f"llm_config['{provider}'] is missing rpm_limit/rpd_limit — "
                "EndpointState requires both. Add the provider's published "
                "per-key rate limits."
            )

        rpm_limit = params["rpm_limit"]
        rpd_limit = params["rpd_limit"]

        for key in params["keys"]:
            api_key = os.environ.get(key)
            if not api_key:
                raise ValueError(
                    f"Environment variable '{key}' is not set"
                )
            for model in params["models"]:
                client_kwargs: Dict[str, Any] = {
                    "model": model,
                    "api_key": api_key,
                }
                if params.get("base_url"):
                    client_kwargs["base_url"] = params["base_url"]
                if not is_gemini:
                    client_kwargs["temperature"] = temperature

                endpoints.append(
                    EndpointState(
                        id=f"{provider}:{model}:{key}",
                        provider=provider,
                        model=model,
                        rpm_limit=rpm_limit,
                        rpd_limit=rpd_limit,
                        client=ChatGoogleGenerativeAI(**client_kwargs) if is_gemini else ChatOpenAI(**client_kwargs),
                    )
                )

    return endpoints


class _MultiProviderToolBoundRunnable(
    Runnable[LanguageModelInput, BaseMessage]
):
    """
    Runnable returned by MultiProviderChatLLM.bind_tools().

    Tool binding is deferred until invocation because MultiProviderChatLLM
    does not represent one concrete provider/model. Each invocation selects
    an endpoint and lets that endpoint perform its native tool binding.
    """

    def __init__(
        self,
        scheduler: LLMScheduler,
        tools: list[Any],
        *,
        tool_choice: Any = None,
        bind_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scheduler = scheduler
        self.tools = tools
        self.tool_choice = tool_choice
        self.bind_kwargs = bind_kwargs or {}

    def _bind_endpoint(self, ep: EndpointState):
        kwargs = dict(self.bind_kwargs)

        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice

        return ep.client.bind_tools(
            self.tools,
            **kwargs,
        )

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseMessage:

        while True:
            with self.scheduler.allocate_endpoint_sync() as ep:
                try:
                    bound_client = self._bind_endpoint(ep)

                    return bound_client.invoke(
                        input,
                        config=config,
                        **kwargs,
                    )

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue

                    raise

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseMessage:

        while True:
            async with self.scheduler.allocate_endpoint() as ep:
                try:
                    bound_client = self._bind_endpoint(ep)

                    return await bound_client.ainvoke(
                        input,
                        config=config,
                        **kwargs,
                    )

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue

                    raise


class _MultiProviderStructuredRunnable(Runnable[LanguageModelInput, Union[Dict, BaseModel]]):
    """
    Runnable returned by MultiProviderChatLLM.with_structured_output().

    MultiProviderChatLLM doesn't hold a single bound client — it picks a
    (provider, model, key) endpoint from the scheduler fresh on every call.
    So structured output can't be resolved once at bind time the way a
    normal single-provider ChatModel would.

    Instead, each ainvoke() here allocates an endpoint exactly like
    MultiProviderChatLLM._agenerate does, then defers to *that endpoint's*
    own `.with_structured_output(...)`. That means each provider uses its
    native mechanism under the hood (OpenAI tool calling, Gemini function
    calling / JSON mode, etc.) — this class does no schema translation of
    its own.

    Caveat: if your pool mixes providers, `**kwargs` passed here (e.g. a
    `method="json_mode"` override) must be valid for every provider in the
    pool, since the same kwargs are forwarded regardless of which endpoint
    gets picked for a given call.
    """

    def __init__(
        self,
        scheduler: LLMScheduler,
        schema: Optional[Union[Dict, Type[BaseModel]]],
        *,
        include_raw: bool = False,
        structured_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scheduler = scheduler
        self.schema = schema
        self.include_raw = include_raw
        self.structured_kwargs = structured_kwargs or {}

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Union[Dict, BaseModel]:
        while True:
            with self.scheduler.allocate_endpoint_sync() as ep:
                try:
                    structured_client = ep.client.with_structured_output(
                        self.schema,
                        include_raw=self.include_raw,
                        **self.structured_kwargs,
                    )
                    return structured_client.invoke(input, config=config, **kwargs)

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue
                    raise

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Union[Dict, BaseModel]:
        while True:
            async with self.scheduler.allocate_endpoint() as ep:
                try:
                    structured_client = ep.client.with_structured_output(
                        self.schema,
                        include_raw=self.include_raw,
                        **self.structured_kwargs,
                    )
                    return await structured_client.ainvoke(input, config=config, **kwargs)

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue
                    raise


class MultiProviderChatLLM(BaseChatModel):
    scheduler: LLMScheduler

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
        while True:
            async with self.scheduler.allocate_endpoint() as ep:
                try:
                    call_kwargs = dict(kwargs)
                    if stop is not None:
                        call_kwargs["stop"] = stop

                    response = await ep.client.ainvoke(messages, **call_kwargs)

                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=response,
                            )
                        ]
                    )

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue
                    raise

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        while True:
            with self.scheduler.allocate_endpoint_sync() as ep:
                try:
                    call_kwargs = dict(kwargs)
                    if stop is not None:
                        call_kwargs["stop"] = stop

                    response = ep.client.invoke(messages, **call_kwargs)

                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=response,
                            )
                        ]
                    )

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue
                    raise

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Returns a Runnable that produces structured output matching `schema`
        (a Pydantic model, TypedDict, or JSON schema dict — same as the
        standard LangChain with_structured_output API).

        Both invoke()/batch() and ainvoke()/abatch() are supported on the
        returned runnable, matching the rest of this class. Each call
        independently allocates an endpoint from the scheduler, so different
        calls (or retries within a call) may land on different
        providers/models/keys — see _MultiProviderStructuredRunnable for
        details and caveats around mixed-provider pools.
        """
        return _MultiProviderStructuredRunnable(
            scheduler=self.scheduler,
            schema=schema,
            include_raw=include_raw,
            structured_kwargs=kwargs,
        )

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        """
        Bind tools dynamically.

        The actual endpoint is selected at invocation time, not when
        bind_tools() is called.
        """
        return _MultiProviderToolBoundRunnable(
            scheduler=self.scheduler,
            tools=tools,
            tool_choice=tool_choice,
            bind_kwargs=kwargs,
        )


def get_llm(
    config: str | Path,
    temperature: float = 0.0,
    max_wait_seconds: float = 120.0,
) -> MultiProviderChatLLM:
    endpoints = build_endpoints(config, temperature=temperature)
    scheduler = LLMScheduler(endpoints, max_wait_seconds=max_wait_seconds)
    return MultiProviderChatLLM(scheduler=scheduler)