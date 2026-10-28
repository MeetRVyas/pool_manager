"""Offline wrapper audit with a mocked HTTP transport and a dummy key. No live API calls."""
import asyncio
import contextvars
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

from evals.metrics import percentile
from pool_manager.endpoint_state import EndpointState
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler

ROOT = Path(__file__).resolve().parents[1]
LABEL = contextvars.ContextVar("audit_path")
MODEL = "gemini-3.5-flash-lite"
PROMPT = "Reply with exactly one word: ok"
CONTENTS = [{"role": "user", "parts": [{"text": PROMPT}]}]
BODY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1, "totalTokenCount": 9},
        "modelVersion": MODEL}


async def audit(iterations=300, warmup=5):
    counts, payloads = {}, {}

    async def handler(request):
        label = LABEL.get()
        counts[label] = counts.get(label, 0) + 1
        payloads.setdefault(label, json.loads(request.content))
        if label == "langchain_503_probe":
            return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE", "message": "offline mock"}})
        return httpx.Response(200, json=BODY)

    # This guard makes an accidental escape from MockTransport fail instead of networking.
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in offline audit")), \
         patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden in offline audit")):
        setup = time.perf_counter()
        model = ChatGoogleGenerativeAI(model=MODEL, api_key="offline-dummy-key", max_retries=0, timeout=60)
        api = model.client._api_client
        original_backend = api._async_httpx_client
        runtime_aiohttp = api._use_aiohttp()
        if runtime_aiohttp:
            raise AssertionError("Runtime uses aiohttp; adapt audit transport explicitly before running")
        backend = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api._async_httpx_client = backend
        await original_backend.aclose()
        pool = MultiProviderChatLLM(scheduler=LLMScheduler([
            EndpointState(id="google:" + MODEL + ":OFFLINE", provider="google", model=MODEL, client=model)]))
        setup_s = time.perf_counter() - setup

        async def raw():
            return await backend.post("https://generativelanguage.googleapis.com/v1beta/models/" + MODEL + ":generateContent",
                                      json={"contents": CONTENTS})

        async def sdk():
            return await model.async_client.models.generate_content(model=MODEL, contents=CONTENTS)

        async def langchain():
            return await model.ainvoke(PROMPT)

        async def pooled():
            return await pool.ainvoke(PROMPT)

        paths = {"raw_http": raw, "google_sdk": sdk, "langchain": langchain, "pool_plus_langchain": pooled}
        timings = {name: [] for name in paths}
        names = list(paths)
        error_probe = {}
        try:
            for i in range(warmup + iterations):
                # Interleave and rotate order to reduce ordering/warmup bias.
                for name in names[i % len(names):] + names[:i % len(names)]:
                    token = LABEL.set(name)
                    before = time.perf_counter()
                    try:
                        await paths[name]()
                    finally:
                        duration = time.perf_counter() - before
                        LABEL.reset(token)
                    if i >= warmup:
                        timings[name].append(duration)
            token = LABEL.set("langchain_503_probe")
            before = time.perf_counter()
            try:
                await model.ainvoke(PROMPT)
            except Exception as exc:
                error_probe.update(error_type=type(exc).__name__, duration_ms=(time.perf_counter() - before) * 1000)
            finally:
                LABEL.reset(token)
            error_probe["mock_http_requests"] = counts.get("langchain_503_probe", 0)
            assert error_probe["mock_http_requests"] == 1, error_probe
            assert "error_type" in error_probe
        finally:
            await model.aclose()
            model.client.close()
        assert all(counts[name] == iterations + warmup for name in paths), counts
        return {"network_requests": 0, "iterations_per_path": iterations, "warmup_per_path": warmup,
                "client_setup_s": setup_s, "runtime_aiohttp": runtime_aiohttp,
                "transport": "httpx.MockTransport, immediate canned 200 response", "mock_requests": counts,
                "units": "milliseconds", "timings": {
                    name: {f"p{q}": percentile([t * 1000 for t in ts], q) for q in (50, 95, 99)}
                    for name, ts in timings.items()},
                "payloads_without_headers": payloads, "max_retries_zero_503_probe": error_probe,
                "caveat": "Measures local wrapper work under a mocked immediate response, not real network "
                          "behavior, concurrent event-loop contention or production tail latency. Client setup excluded."}


def main():
    result = asyncio.run(audit())
    path = ROOT / "evals/results/WRAPPER_OVERHEAD_2026-10-27.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
