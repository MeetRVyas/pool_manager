"""Low-volume latency diagnosis; run explicitly with --yes to send live calls."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from pool_manager.endpoint_state import EndpointState
from pool_manager.errors import classify
from pool_manager.llm import MultiProviderChatLLM, get_llm
from pool_manager.scheduler import LLMScheduler

PROMPT = "Reply with exactly one word: ok"
CURRENT = contextvars.ContextVar("latency_trace", default=None)


def event(name, **fields):
    row = CURRENT.get()
    if row is not None:
        row["events"].append({"event": name, "at_s": time.perf_counter() - row["origin"], **fields})


async def transport_trace(name, info):
    # Deliberately do not serialize info: it can contain request headers.
    event(name)


async def request_hook(request):
    request.extensions["trace"] = transport_trace
    event("http_request", transport=type(request).__module__)


async def response_hook(response):
    timing = {k: response.headers[k] for k in ("server-timing", "x-envoy-upstream-service-time") if k in response.headers}
    event("http_response_headers", status=response.status_code, timing_headers=timing)


class ObservedClient:
    def __init__(self, endpoint):
        self.client, self.endpoint_id = endpoint.client, endpoint.id

    async def ainvoke(self, *args, **kwargs):
        event("provider_start", endpoint=self.endpoint_id)
        try:
            return await self.client.ainvoke(*args, **kwargs)
        finally:
            event("provider_end", endpoint=self.endpoint_id)


async def measure(label, call):
    row = {"case": label, "events": [], "origin": time.perf_counter()}
    token = CURRENT.set(row)
    print("START", label, flush=True)
    try:
        result = await call()
        row["ok"] = True
        row["usage"] = getattr(result, "usage_metadata", None)
    except Exception as exc:
        info = classify(exc)
        row.update(ok=False, error_type=type(exc).__name__, status=info.status, kind=info.kind.value)
    finally:
        row["total_s"] = time.perf_counter() - row.pop("origin")
        CURRENT.reset(token)
    print("DONE", label, "ok=", row["ok"], "seconds=", round(row["total_s"], 4), flush=True)
    for item in row["events"]:
        if item["event"] in ("scheduler_acquired", "scheduler_sleep", "http_request", "http_response_headers", "first_text"):
            print(json.dumps(item), flush=True)
    return row


async def run():
    load_dotenv(".env", override=True)
    setup = time.perf_counter()
    pool = get_llm("configs/llm_config.yaml", request_timeout=60, max_wait_seconds=70)
    originals = {ep.id: ep.client for ep in pool.scheduler.endpoints}
    backend_types = {}
    for ep in pool.scheduler.endpoints:
        backend = ep.client.client._api_client._async_httpx_client
        backend_types[ep.id] = type(backend).__module__ + "." + type(backend).__name__
        backend.event_hooks["request"].append(request_hook)
        backend.event_hooks["response"].append(response_hook)
        ep.client = ObservedClient(ep)
    setup_s = time.perf_counter() - setup
    acquire, asleep = pool.scheduler.acquire, pool.scheduler._asleep

    async def observed_acquire(*args, **kwargs):
        before = time.perf_counter()
        ep, waited = await acquire(*args, **kwargs)
        event("scheduler_acquired", endpoint=ep.id, reported_wait_s=waited,
              elapsed_s=time.perf_counter() - before)
        return ep, waited

    async def observed_sleep(seconds):
        event("scheduler_sleep", seconds=seconds)
        await asleep(seconds)

    pool.scheduler.acquire = observed_acquire
    pool.scheduler._asleep = observed_sleep
    rows = []

    class ImmediateClient:
        async def ainvoke(self, *args, **kwargs):
            return AIMessage(content="ok")

    fake = MultiProviderChatLLM(scheduler=LLMScheduler([
        EndpointState("local:fake:KEY", "local", "fake", ImmediateClient())]))

    async def local_calls():
        for _ in range(100):
            await fake.ainvoke(PROMPT)

    rows.append(await measure("local_pool_100_immediate_responses", local_calls))
    for i in range(2):
        rows.append(await measure(f"pool_sequential_{i + 1}", lambda: pool.ainvoke(PROMPT)))

    lite31 = next(c for eid, c in originals.items() if ":gemini-3.1-flash-lite:" in eid)
    rows.append(await measure("direct_langchain_3.1_lite", lambda: lite31.ainvoke(PROMPT)))

    async with httpx.AsyncClient(timeout=60, event_hooks={"request": [request_hook], "response": [response_hook]}) as raw:
        for model in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
            async def call(model=model):
                response = await raw.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                    json={"contents": [{"role": "user", "parts": [{"text": PROMPT}]}]},
                )
                event("raw_body_complete", status=response.status_code)
                CURRENT.get()["raw_status"] = response.status_code
                CURRENT.get()["raw_usage"] = response.json().get("usageMetadata")
                response.raise_for_status()
            rows.append(await measure("raw_http_" + model, call))

    lite35 = next(c for eid, c in originals.items() if ":gemini-3.5-flash-lite:" in eid)

    async def stream():
        first = True
        combined = None
        async for chunk in lite35.astream(PROMPT):
            combined = chunk if combined is None else combined + chunk
            if first and chunk.content:
                event("first_text")
                first = False
        if combined is not None:
            CURRENT.get()["stream_usage"] = combined.usage_metadata

    rows.append(await measure("direct_langchain_3.5_lite_stream", stream))
    return {"client_setup_s": setup_s, "sdk_http_backends": backend_types,
            "proxy_environment_present": {k: bool(os.environ.get(k)) for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")},
            "prompt": PROMPT, "sdk_max_retries": {eid: c.max_retries for eid, c in originals.items()},
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--out", default="evals/results/latency_trace_2026-10-26.json")
    args = parser.parse_args()
    if not args.yes:
        parser.error("--yes is required to send six live calls (pool retries may add attempts)")
    result = asyncio.run(run())
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
