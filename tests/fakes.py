"""Test doubles: a controllable clock, scripted fake chat clients, and REAL SDK exceptions."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
import openai
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage
import langchain_google_genai.chat_models as gm

from pool_manager.endpoint_state import CooldownPolicy, EndpointState
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------------------ time
class FakeTime:
    """Monotonic clock that only moves when something sleeps -- tests never really wait."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    async def asleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
        await asyncio.sleep(0)

    @property
    def total_slept(self) -> float:
        return sum(self.sleeps)


# ---------------------------------------------------------------------- fake clients
class ParseFail:
    """Script marker: the structured-output call returns parsing_error instead of a value."""


class FakeClient:
    """
    Stands in for a LangChain chat model. `script` is consumed one item per call:
    an Exception instance is raised, anything else (or an exhausted script) is a success.
    """

    def __init__(self, name: str, script: Optional[list] = None, tokens: tuple[int, int] = (10, 5)) -> None:
        self.name = name
        self.script = list(script or [])
        self.tokens = tokens
        self.calls = 0

    def _next(self) -> None:
        self.calls += 1
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return None

    def _message(self, **extra: Any) -> AIMessage:
        i, o = self.tokens
        return AIMessage(
            content=f"from {self.name}",
            usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o},
            **extra,
        )

    # plain chat
    def invoke(self, messages, **kwargs):
        self._next()
        return self._message()

    async def ainvoke(self, messages, **kwargs):
        self._next()
        return self._message()

    # tools
    def bind_tools(self, tools, **kwargs):
        return _Bound(self)

    # structured output
    def with_structured_output(self, schema, include_raw=False, **kwargs):
        assert include_raw is True, "the pool must always request include_raw=True from endpoints"
        return _Structured(self)


class _Bound:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    def _msg(self):
        return self.client._message(tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "call_1"}])

    def invoke(self, input, config=None, **kw):
        self.client._next()
        return self._msg()

    async def ainvoke(self, input, config=None, **kw):
        self.client._next()
        return self._msg()


class _Structured:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    def _result(self, item):
        if isinstance(item, ParseFail):
            return {"raw": self.client._message(), "parsed": None, "parsing_error": ValueError("bad json")}
        return {"raw": self.client._message(), "parsed": {"answer": self.client.name}, "parsing_error": None}

    def invoke(self, input, config=None, **kw):
        return self._result(self.client._next())

    async def ainvoke(self, input, config=None, **kw):
        return self._result(self.client._next())


# --------------------------------------------------------------------- build helpers
def make_ep(
    ep_id: str,
    client: Optional[FakeClient] = None,
    *,
    priority: int = 0,
    cooldowns: Optional[CooldownPolicy] = None,
    **limits: Any,
) -> EndpointState:
    return EndpointState(
        id=ep_id,
        provider=ep_id.split(":")[0],
        model=ep_id.split(":")[1],
        client=client or FakeClient(ep_id),
        priority=priority,
        cooldowns=cooldowns or CooldownPolicy(),
        **limits,
    )


def make_scheduler(eps, ft: Optional[FakeTime] = None, **kwargs) -> LLMScheduler:
    ft = ft or FakeTime()
    kwargs.setdefault("max_wait_seconds", 60.0)
    return LLMScheduler(eps, clock=ft.now, sleep=ft.sleep, asleep=ft.asleep, **kwargs)


def make_llm(eps, ft: Optional[FakeTime] = None, max_attempts: int = 6, **sched_kwargs) -> MultiProviderChatLLM:
    return MultiProviderChatLLM(scheduler=make_scheduler(eps, ft, **sched_kwargs), max_attempts=max_attempts)


# ------------------------------------------------------- REAL SDK exceptions (as raised)
def gemini_client_error(code: int, status: str, message: str, details: Optional[list] = None):
    """A Gemini 4xx exactly as langchain-google-genai 4.x surfaces it (wrapper with __cause__)."""
    raw = ClientError(code, {"error": {"code": code, "message": message, "status": status, "details": details or []}})
    try:
        gm._handle_client_error(raw, {"model": "gemini-2.5-flash"})
    except Exception as wrapped:
        return wrapped
    raise AssertionError("expected the wrapper to raise")


def gemini_server_error(code: int, status: str, message: str):
    raw = ServerError(code, {"error": {"code": code, "message": message, "status": status}})
    try:
        gm._handle_server_error(raw)
    except Exception as wrapped:
        return wrapped
    raise AssertionError("expected the wrapper to raise")


def gemini_429(retry: str = "34s", per_day: bool = False):
    quota_id = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if per_day else "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    return gemini_client_error(
        429,
        "RESOURCE_EXHAUSTED",
        # the real text includes "billing details" even for plain per-minute limits
        f"You exceeded your current quota, please check your plan and billing details. Please retry in {retry}.",
        details=[
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry},
        ],
    )


_REQ = httpx.Request("POST", "https://example.test/v1/chat/completions")


def openai_status_error(cls, status: int, headers: Optional[dict] = None, body: Optional[dict] = None):
    response = httpx.Response(status, request=_REQ, headers=headers or {})
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def openai_429(headers: Optional[dict] = None, body: Optional[dict] = None):
    return openai_status_error(openai.RateLimitError, 429, headers, body)


def openai_500():
    return openai_status_error(openai.InternalServerError, 500)
