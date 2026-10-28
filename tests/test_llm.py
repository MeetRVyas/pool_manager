"""End-to-end through MultiProviderChatLLM with scripted fake clients and REAL SDK exceptions."""
import pytest
from pydantic import BaseModel

from pool_manager.benchmark import get_tracker, reset_tracker
from pool_manager.limits import LimitsCatalogue
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import AllAttemptsFailed, AllEndpointsExhausted
from tests.fakes import (
    FakeClient, FakeTime, ParseFail, gemini_429, gemini_client_error, gemini_server_error,
    make_ep, make_llm, make_scheduler, openai_429, openai_500, run,
)


@pytest.fixture(autouse=True)
def fresh_tracker(tmp_path):
    reset_tracker(tmp_path / "no_pricing.yaml")
    yield


def two(first_script, second_script=None, **limits):
    a = make_ep("google:flash:KEY_1", FakeClient("A", first_script), **limits)
    b = make_ep("google:flash:KEY_2", FakeClient("B", second_script), **limits)
    return a, b


# ======================================================================= 429 handling
def test_gemini_429_fails_over_within_the_same_call_and_cools_the_key_down():
    ft = FakeTime()
    a, b = two([gemini_429(retry="34s")])
    llm = make_llm([a, b], ft)

    out = run(llm.ainvoke("hi"))
    assert out.content == "from B"                              # the caller never saw the 429
    assert (a.client.calls, b.client.calls) == (1, 1)
    assert a.cooldown_remaining(ft.now()) == pytest.approx(35.0)   # Retry-After 34s + 1s pad
    assert ft.sleeps == []                                      # switched instantly, no waiting

    run(llm.ainvoke("again")); run(llm.ainvoke("again"))
    assert a.client.calls == 1                                  # A is skipped while cooling down


def test_sync_invoke_has_the_same_failover():
    a, b = two([openai_429(headers={"retry-after": "7"})])
    assert make_llm([a, b]).invoke("hi").content == "from B"
    assert a.rate_limited == 1


def test_a_cooled_down_key_comes_back_after_its_retry_after():
    ft = FakeTime()
    a, b = two([gemini_429(retry="10s")])
    llm = make_llm([a, b], ft)
    run(llm.ainvoke("1"))
    ft.t += 12
    served = {run(llm.ainvoke(str(i))).content for i in range(4)}
    assert "from A" in served


def test_per_day_quota_parks_the_key_for_a_long_time():
    ft = FakeTime()
    a, b = two([gemini_429(per_day=True)])
    run(make_llm([a, b], ft).ainvoke("hi"))
    assert a.cooldown_remaining(ft.now()) >= 3600


# ================================================================= server errors etc.
@pytest.mark.parametrize(
    "exc",
    [
        gemini_server_error(503, "UNAVAILABLE", "This model is currently experiencing high demand."),
        gemini_server_error(500, "INTERNAL", "Internal error"),
        gemini_server_error(504, "DEADLINE_EXCEEDED", "Deadline expired"),
        openai_500(),
        TimeoutError("read timed out"),
    ],
    ids=["gemini503", "gemini500", "gemini504", "openai500", "timeout"],
)
def test_server_errors_and_timeouts_fail_over_with_a_short_backoff(exc):
    ft = FakeTime()
    a, b = two([exc])
    assert run(make_llm([a, b], ft).ainvoke("hi")).content == "from B"
    assert a.server_errors == 1
    assert a.cooldown_remaining(ft.now()) == pytest.approx(5.0)


def test_a_dead_key_is_quarantined_and_the_call_still_succeeds():
    a, b = two([gemini_client_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")])
    ft = FakeTime()
    assert run(make_llm([a, b], ft).ainvoke("hi")).content == "from B"
    assert a.dead_errors == 1 and a.cooldown_remaining(ft.now()) == 3600


def test_a_bad_request_is_raised_to_the_caller_without_burning_other_endpoints():
    bad = gemini_client_error(400, "INVALID_ARGUMENT", "Unable to submit request: invalid schema")
    a, b = two([bad])
    with pytest.raises(Exception) as err:
        run(make_llm([a, b]).ainvoke("hi"))
    assert err.value is bad                                     # the original exception, untouched
    assert b.client.calls == 0                                  # retrying a bad request elsewhere is pointless
    assert a.cooldown_remaining(0.0) == 0.0                     # and the key is not penalised


def test_programming_errors_are_not_swallowed():
    a, b = two([KeyError("bug")])
    with pytest.raises(KeyError):
        run(make_llm([a, b]).ainvoke("hi"))


# ===================================================================== bounded effort
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_slow_failures_try_unvisited_model_keys_before_revisiting_primary(mode):
    ft = FakeTime(start=0)
    failures = []

    class SlowFailure(FakeClient):
        def invoke(self, *args, **kwargs):
            self.calls += 1
            failures.append(self.name)
            ft.t += 10  # earlier endpoints' cooldowns expire during this attempt
            raise gemini_server_error(503, "UNAVAILABLE", "slow failure")

        async def ainvoke(self, *args, **kwargs):
            return self.invoke(*args, **kwargs)

    eps = [
        make_ep("google:flash:KEY_1", SlowFailure("flash-1"), priority=0),
        make_ep("google:flash:KEY_2", SlowFailure("flash-2"), priority=0),
        make_ep("google:lite:KEY_1", SlowFailure("lite-1"), priority=1),
        make_ep("google:lite:KEY_2", FakeClient("healthy"), priority=1),
    ]
    llm = make_llm(eps, ft)
    response = llm.invoke("hello") if mode == "sync" else run(llm.ainvoke("hello"))
    assert response.content == "from healthy"
    assert failures == ["flash-1", "flash-2", "lite-1"]
    assert [ep.calls for ep in eps] == [1, 1, 1, 1]


def test_attempts_are_capped_and_the_provider_error_is_chained():
    eps = [make_ep(f"google:flash:KEY_{i}", FakeClient(f"c{i}", [openai_500()])) for i in range(5)]
    llm = make_llm(eps, max_attempts=3)
    with pytest.raises(AllAttemptsFailed) as err:
        run(llm.ainvoke("hi"))
    assert sum(ep.client.calls for ep in eps) == 3              # stopped at 3, did not grind through all 5
    assert err.value.__cause__ is not None                      # last provider error is attached
    assert isinstance(err.value, AllEndpointsExhausted)         # old `except AllEndpointsExhausted` still works


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_retry_can_reuse_an_endpoint_when_no_untried_endpoint_is_available(mode):
    ft = FakeTime()
    ep = make_ep("google:flash:KEY", FakeClient("recovered", [gemini_server_error(503, "UNAVAILABLE", "temporary")]))
    llm = make_llm([ep], ft)
    response = llm.invoke("hello") if mode == "sync" else run(llm.ainvoke("hello"))
    assert response.content == "from recovered"
    assert ep.calls == 2
    assert ft.total_slept == pytest.approx(5)


def test_sync_path_caps_attempts_too():
    eps = [make_ep(f"google:flash:KEY_{i}", FakeClient(f"c{i}", [openai_500()])) for i in range(5)]
    with pytest.raises(AllAttemptsFailed):
        make_llm(eps, max_attempts=2).invoke("hi")
    assert sum(ep.client.calls for ep in eps) == 2


def test_every_key_out_of_daily_quota_raises_immediately_instead_of_waiting():
    ft = FakeTime()
    eps = [make_ep(f"google:flash:KEY_{i}", FakeClient(f"c{i}", [gemini_429(per_day=True)])) for i in range(3)]
    llm = make_llm(eps, ft, max_wait_seconds=60)
    with pytest.raises(AllEndpointsExhausted) as err:
        run(llm.ainvoke("hi"))
    assert ft.total_slept == 0.0                                # no 60 s (or 120 s) sleep for a daily quota
    assert err.value.retry_after >= 3600
    assert err.value.__cause__ is not None                      # shows the 429 that exhausted the pool


def test_total_waiting_across_retries_never_exceeds_the_budget():
    ft = FakeTime()
    a = make_ep("google:flash:KEY_1", FakeClient("A", [gemini_429(retry="40s")] * 3), rpm_limit=1)
    b = make_ep("google:flash:KEY_2", FakeClient("B", [gemini_429(retry="40s")] * 3), rpm_limit=1)
    llm = make_llm([a, b], ft, max_wait_seconds=30)
    with pytest.raises(AllEndpointsExhausted):
        run(llm.ainvoke("hi"))
    assert ft.total_slept <= 30


# ================================================ pre-stop from limits / no limits
def test_with_a_yaml_limit_the_primary_model_is_pre_stopped_and_never_returns_429():
    primary = [make_ep(f"google:flash:KEY_{i}", FakeClient(f"P{i}"), rpm_limit=2) for i in (1, 2)]
    fallback = [make_ep(f"google:lite:KEY_{i}", FakeClient(f"F{i}"), priority=1, rpm_limit=15) for i in (1, 2)]
    llm = make_llm(primary + fallback)
    served = [run(llm.ainvoke(str(i))).content[-2] for i in range(8)]
    assert served.count("P") == 4 and served.count("F") == 4    # 2 keys x rpm 2 on primary, rest on fallback
    assert sum(ep.client.calls for ep in primary) == 4          # the primary was never asked a 5th time
    assert sum(ep.rate_limited for ep in primary + fallback) == 0


def test_model_with_no_known_limit_is_handled_reactively_and_the_pipeline_keeps_going():
    # No limits configured -> nothing pre-stops it; the 429 arrives, we switch, we carry on.
    a = make_ep("google:unknown:KEY_1", FakeClient("A", [gemini_429(retry="20s")]))
    b = make_ep("google:unknown:KEY_2", FakeClient("B"))
    llm = make_llm([a, b])
    outs = [run(llm.ainvoke(str(i))).content for i in range(5)]
    assert outs[0] == "from B" and set(outs) == {"from B"}
    assert a.rate_limited == 1 and a.client.calls == 1


# ================================================================ tools / structured
def test_bind_tools_preserves_tool_calls_async_and_sync():
    llm = make_llm(list(two([openai_429()])))
    bound = llm.bind_tools([{"name": "lookup"}])
    out = run(bound.ainvoke("q"))
    assert out.tool_calls and out.tool_calls[0]["name"] == "lookup"   # v1/v2 dropped these
    llm2 = make_llm(list(two([openai_500()])))
    assert llm2.bind_tools([{"name": "lookup"}]).invoke("q").tool_calls


def test_plain_chat_returns_the_endpoints_message_untouched_sync_and_async():
    from langchain_core.messages import AIMessage

    rich = AIMessage(
        content="x",
        tool_calls=[{"name": "lookup", "args": {"q": 1}, "id": "c1"}],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    )

    class RichClient(FakeClient):
        def invoke(self, messages, **kw):
            self._next()
            return rich

        async def ainvoke(self, messages, **kw):
            self._next()
            return rich

    for out in (run(make_llm([make_ep("google:m:K", RichClient("R"))]).ainvoke("q")),
                make_llm([make_ep("google:m:K", RichClient("R"))]).invoke("q")):
        assert out.tool_calls and out.tool_calls[0]["id"] == "c1"
        assert out.response_metadata == {"finish_reason": "tool_calls"}


class Answer(BaseModel):
    answer: str


def test_structured_output_default_returns_parsed_value_and_fails_over():
    llm = make_llm(list(two([gemini_429()])))
    assert run(llm.with_structured_output(Answer).ainvoke("q")) == {"answer": "B"}
    llm2 = make_llm(list(two([gemini_server_error(503, "UNAVAILABLE", "x")])))
    assert llm2.with_structured_output(Answer).invoke("q") == {"answer": "B"}


def test_structured_output_include_raw_keeps_the_caller_shape():
    out = run(make_llm(list(two([]))).with_structured_output(Answer, include_raw=True).ainvoke("q"))
    assert set(out) == {"raw", "parsed", "parsing_error"} and out["raw"].usage_metadata


def test_a_parse_failure_is_raised_and_does_not_penalise_or_switch_endpoints():
    a, b = two([ParseFail()])
    with pytest.raises(ValueError, match="bad json"):
        run(make_llm([a, b]).with_structured_output(Answer).ainvoke("q"))
    assert b.client.calls == 0 and a.server_errors == 0 and a.cooldown_remaining(0.0) == 0


# =============================================================== benchmark (from v2)
def test_benchmark_records_phase_tokens_and_the_exact_key_for_every_path():
    a, b = two([])
    llm = make_llm([a, b])
    run(llm.ainvoke("x", config={"metadata": {"phase": "extract"}}))
    run(llm.with_structured_output(Answer).ainvoke("x", config={"metadata": {"phase": "classify"}}))
    llm.bind_tools([{"name": "t"}]).invoke("x", config={"metadata": {"phase": "act"}})

    summary = get_tracker().summary()
    assert summary["total_calls"] == 3
    assert set(summary["by_phase"]) == {"extract", "classify", "act"}
    assert summary["by_phase"]["extract"]["total_input_tokens"] == 10
    assert sum(v["call_count"] for v in summary["by_key"].values()) == 3
    assert all(k.startswith("google:flash:KEY_") for k in summary["by_key"])


def test_failed_attempts_are_not_counted_as_billable_calls():
    a, b = two([gemini_429()])
    run(make_llm([a, b]).ainvoke("x"))
    assert get_tracker().summary()["total_calls"] == 1


def test_tracking_failure_never_breaks_a_call(monkeypatch):
    import pool_manager.llm as llm_mod
    monkeypatch.setattr(llm_mod, "get_tracker", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert run(make_llm(list(two([]))).ainvoke("x")).content == "from A"


# ============================================================ real LangChain plumbing
def test_works_through_langchain_batch_and_abatch():
    llm = make_llm(list(two([])))
    sync_out = llm.batch(["a", "b", "c", "d"])                  # runs on a thread pool
    assert len(sync_out) == 4 and {m.content for m in sync_out} <= {"from A", "from B"}
    async_out = run(llm.abatch(["a", "b", "c", "d"]))           # concurrent tasks
    assert len(async_out) == 4
    assert sum(ep.calls for ep in llm.scheduler.endpoints) == 8  # every request was accounted for exactly once


def test_stats_and_usage_report_are_exposed_on_the_llm():
    llm = make_llm(list(two([])))
    run(llm.ainvoke("x"))
    assert llm.stats()[0]["calls"] == 1 and "endpoints used" in llm.usage_report()
