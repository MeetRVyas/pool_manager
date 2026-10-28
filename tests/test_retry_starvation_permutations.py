"""Real pool/scheduler, provider doubles only; no clock mocks or sleeps.

Permute four endpoint identities across priority ranks. Responses belong to rank:
503, connection error, 504, healthy. Zero transient cooldown keeps failed endpoints
eligible, exposing starvation without relying on elapsed or virtual time.
Optional RETRY_STARVATION_RESULTS writes one diagnostic JSON row per pytest case.
"""
import asyncio
from itertools import permutations
import json
import os
from pathlib import Path

from google.genai.errors import ServerError
import httpx
from langchain_core.messages import AIMessage
import pytest

from pool_manager.benchmark import reset_tracker
from pool_manager.endpoint_state import CooldownPolicy, EndpointState
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler


ORDERS = list(permutations("ABCD"))


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("order", ORDERS, ids=lambda order: "".join(order))
def test_retry_starvation_all_priority_orders(mode, order, tmp_path):
    reset_tracker(tmp_path / "no_pricing.yaml")
    attempts = []
    endpoints = []
    errors = [
        ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "scripted 503"}}),
        httpx.ConnectError("scripted connection failure"),
        ServerError(504, {"error": {"code": 504, "status": "DEADLINE_EXCEEDED", "message": "scripted 504"}}),
        None,
    ]
    expected = [f"google:model-{name}:KEY_{name}" for name in order]
    diagnostic = {"mode": mode, "priority_order": list(order), "expected_sequence": expected,
                  "attempts": attempts, "passed": False}

    class ProviderResponse:
        def __init__(self, endpoint_id, error):
            self.endpoint_id, self.error = endpoint_id, error

        def invoke(self, messages, **kwargs):
            seen = {a["endpoint"] for a in attempts}
            eligible_untried = [ep.id for ep in endpoints
                                if ep.id not in seen and ep.available_in(scheduler.clock()) <= 0]
            attempts.append({"endpoint": self.endpoint_id,
                             "eligible_untried": eligible_untried,
                             "skipped_untried": self.endpoint_id in seen and bool(eligible_untried),
                             "result": ("503", "connection", "504", "success")[expected.index(self.endpoint_id)]})
            if self.error is not None:
                raise self.error  # Persistent failure, including if the old scheduler revisits it.
            return AIMessage(content="healthy response", usage_metadata={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    for priority, name in enumerate(order):
        endpoint_id = expected[priority]
        endpoints.append(EndpointState(
            id=endpoint_id, provider="google", model=f"model-{name}", priority=priority,
            client=ProviderResponse(endpoint_id, errors[priority]),
            rpm_limit=100, rpd_limit=100, tpm_limit=10000,
            cooldowns=CooldownPolicy(server_error_base=0, server_error_max=0),
        ))
    # Keep storage order fixed: the permutation must affect priority, not list order.
    endpoints.sort(key=lambda ep: ep.id)
    scheduler = LLMScheduler(endpoints, max_wait_seconds=0, strategy="priority")
    pool = MultiProviderChatLLM(scheduler=scheduler, max_attempts=6)
    try:
        assert all(ep.available_in(scheduler.clock()) == 0 for ep in endpoints)
        response = pool.invoke("test") if mode == "sync" else asyncio.run(pool.ainvoke("test"))
        diagnostic["response"] = response.content
        sequence = [a["endpoint"] for a in attempts]
        assert response.content == "healthy response"
        assert len(attempts) == 4, sequence
        assert sequence == expected, sequence
        assert [a["result"] for a in attempts] == ["503", "connection", "504", "success"]
        assert all(sequence.count(endpoint_id) == 1 for endpoint_id in expected)
        assert all(not a["skipped_untried"] for a in attempts), attempts
        assert all(a["endpoint"] in a["eligible_untried"] for a in attempts), attempts
        assert all(ep.calls == 1 for ep in endpoints)
        assert sum(ep.successes for ep in endpoints) == 1
        assert sum(ep.server_errors for ep in endpoints) == 3
        diagnostic["passed"] = True
    except BaseException as exc:
        diagnostic["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        diagnostic["actual_attempts"] = len(attempts)
        diagnostic["actual_sequence"] = [a["endpoint"] for a in attempts]
        print(json.dumps(diagnostic, sort_keys=True))
        destination = os.environ.get("RETRY_STARVATION_RESULTS")
        if destination:
            with Path(destination).open("a", encoding="utf-8") as report:
                report.write(json.dumps(diagnostic, sort_keys=True) + "\n")
