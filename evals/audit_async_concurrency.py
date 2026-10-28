"""Offline async overlap and event-loop lag audit; no live API calls or pool edits."""
import asyncio
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

from evals.audit_wrapper_overhead import BODY, MODEL, PROMPT
from evals.metrics import percentile
from pool_manager.endpoint_state import EndpointState
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler

ROOT = Path(__file__).resolve().parents[1]


def ms_stats(values):
    return {**{f"p{q}": percentile([v * 1000 for v in values], q) for q in (50, 95, 99)},
            "max": max(values, default=0) * 1000}


async def audit(concurrencies=(1, 8, 32), rounds=3, delay_s=0.1):
    active, peak, requests = 0, 0, 0

    async def handler(request):
        nonlocal active, peak, requests
        active += 1
        peak = max(peak, active)
        requests += 1
        try:
            await asyncio.sleep(delay_s)
            return httpx.Response(200, json=BODY)
        finally:
            active -= 1

    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
         patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")):
        model = ChatGoogleGenerativeAI(model=MODEL, api_key="offline-dummy-key", max_retries=0, timeout=60)
        api = model.client._api_client
        if api._use_aiohttp():
            raise AssertionError("Adapt the mock explicitly for an aiohttp runtime")
        original = api._async_httpx_client
        original_pool = getattr(getattr(original, "_transport", None), "_pool", None)
        limits = {name: getattr(original_pool, name, None) for name in
                  ("_max_connections", "_max_keepalive_connections")}
        backend = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api._async_httpx_client = backend
        await original.aclose()
        pool = MultiProviderChatLLM(scheduler=LLMScheduler([
            EndpointState(id="google:" + MODEL + ":OFFLINE", provider="google", model=MODEL, client=model)]))

        async def raw():
            return await backend.post("https://example.test/generate", json={"contents": PROMPT})

        async def pooled():
            return await pool.ainvoke(PROMPT)

        results = []
        try:
            for name, call in (("raw_http", raw), ("pool_langchain_sdk", pooled)):
                await call()  # exclude first-call setup from measured batches
                for concurrency in concurrencies:
                    durations, dispatch, submitted_latency, lags, walls, peaks = [], [], [], [], [], []
                    for _ in range(rounds):
                        peak = 0
                        stop = asyncio.Event()
                        async def monitor():
                            while not stop.is_set():
                                before = time.perf_counter()
                                await asyncio.sleep(0.005)
                                lags.append(max(0, time.perf_counter() - before - 0.005))
                        watch = asyncio.create_task(monitor())
                        origin = time.perf_counter()
                        async def one():
                            start = time.perf_counter()
                            dispatch.append(start - origin)
                            await call()
                            end = time.perf_counter()
                            durations.append(end - start)
                            submitted_latency.append(end - origin)
                        await asyncio.gather(*(one() for _ in range(concurrency)))
                        walls.append(time.perf_counter() - origin)
                        stop.set()
                        await watch
                        peaks.append(peak)
                    results.append(dict(path=name, concurrency=concurrency, samples=len(durations),
                                        peak_transport_overlap=max(peaks), caller_ms=ms_stats(durations),
                                        from_submission_ms=ms_stats(submitted_latency),
                                        task_dispatch_ms=ms_stats(dispatch), loop_lag_ms=ms_stats(lags),
                                        batch_wall_ms=ms_stats(walls)))
        finally:
            await model.aclose()
            model.client.close()
        return dict(network_requests=0, mock_http_requests=requests,
                    mock_response_delay_ms=delay_s * 1000, rounds=rounds,
                    sdk_default_connection_limits=limits, cases=results,
                    note="Immediate unlimited pool admission, with a mocked async response delay. "
                         "This checks local overlap/loop scheduling, not upstream load or real TCP/HTTP connection queueing.")


def main():
    result = asyncio.run(audit())
    (ROOT / "evals/results/ASYNC_CONCURRENCY_AUDIT_2026-10-27.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
