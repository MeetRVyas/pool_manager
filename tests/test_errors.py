"""Classification of REAL provider exceptions -> what the pool should do about them."""
import httpx
import openai
import pytest

from pool_manager.errors import Kind, classify, parse_duration
from tests.fakes import (
    gemini_429, gemini_client_error, gemini_server_error,
    openai_429, openai_500, openai_status_error,
)


# ---------------------------------------------------------------- Gemini (wrapped)
def test_gemini_429_wrapper_with_no_code_is_still_a_rate_limit():
    # langchain-google-genai 4.x raises GoogleRateLimitError whose own .code is None.
    exc = gemini_429(retry="34s")
    assert getattr(exc, "code", None) is None          # the property v3 relied on
    info = classify(exc)
    assert info.kind is Kind.RATE_LIMIT
    assert info.status == 429
    assert info.retry_after == pytest.approx(34.0)


def test_per_minute_429_mentioning_billing_is_not_misread_as_daily_quota():
    # Gemini's ordinary per-minute 429 text says "check your plan and billing details".
    assert classify(gemini_429(per_day=False)).kind is Kind.RATE_LIMIT


def test_per_day_429_is_quota():
    assert classify(gemini_429(per_day=True)).kind is Kind.QUOTA


@pytest.mark.parametrize("code,status", [(500, "INTERNAL"), (503, "UNAVAILABLE"), (504, "DEADLINE_EXCEEDED")])
def test_gemini_server_errors_are_transient(code, status):
    info = classify(gemini_server_error(code, status, "Something went wrong."))
    assert info.kind is Kind.TRANSIENT and info.status == code


def test_gemini_503_high_demand_is_transient():
    exc = gemini_server_error(503, "UNAVAILABLE", "This model is currently experiencing high demand.")
    assert classify(exc).kind is Kind.TRANSIENT


def test_gemini_invalid_key_comes_back_as_http_400_and_means_dead_endpoint():
    exc = gemini_client_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")
    assert classify(exc).kind is Kind.DEAD


@pytest.mark.parametrize("code,status", [(403, "PERMISSION_DENIED"), (404, "NOT_FOUND"), (401, "UNAUTHENTICATED")])
def test_gemini_auth_and_missing_model_are_dead(code, status):
    assert classify(gemini_client_error(code, status, "nope")).kind is Kind.DEAD


def test_gemini_ordinary_400_is_fatal_and_must_reach_the_caller():
    assert classify(gemini_client_error(400, "INVALID_ARGUMENT", "Unable to submit request: bad schema")).kind is Kind.FATAL


# ------------------------------------------------------------------------- OpenAI
def test_openai_429_honours_retry_after_header():
    info = classify(openai_429(headers={"retry-after": "7"}))
    assert info.kind is Kind.RATE_LIMIT and info.retry_after == 7.0


def test_openai_retry_after_ms_header_wins():
    info = classify(openai_429(headers={"retry-after-ms": "1500", "retry-after": "9"}))
    assert info.retry_after == pytest.approx(1.5)


def test_openai_429_message_duration_is_parsed():
    exc = openai_429(body={"error": {"message": "Rate limit reached. Please try again in 6m0s.", "code": "rate_limit_exceeded"}})
    assert classify(exc).retry_after == pytest.approx(360.0)


def test_openai_insufficient_quota_is_quota_not_a_short_rate_limit():
    exc = openai_429(body={"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}})
    assert classify(exc).kind is Kind.QUOTA


def test_openai_500_is_transient():
    assert classify(openai_500()).kind is Kind.TRANSIENT


@pytest.mark.parametrize("cls,status", [(openai.AuthenticationError, 401), (openai.PermissionDeniedError, 403), (openai.NotFoundError, 404)])
def test_openai_auth_and_not_found_are_dead(cls, status):
    assert classify(openai_status_error(cls, status)).kind is Kind.DEAD


def test_openai_bad_request_is_fatal():
    assert classify(openai_status_error(openai.BadRequestError, 400)).kind is Kind.FATAL


def test_openai_connection_and_timeout_errors_are_transient():
    req = httpx.Request("POST", "https://x")
    assert classify(openai.APIConnectionError(request=req)).kind is Kind.TRANSIENT
    assert classify(openai.APITimeoutError(request=req)).kind is Kind.TRANSIENT


# ------------------------------------------------------------------------- generic
@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionResetError(), httpx.ReadTimeout("t"), httpx.ConnectError("c")])
def test_network_level_errors_are_transient(exc):
    assert classify(exc).kind is Kind.TRANSIENT


@pytest.mark.parametrize("exc", [ValueError("429 in my message"), KeyError("k"), RuntimeError("boom")])
def test_programming_errors_are_fatal_even_if_the_text_mentions_a_status(exc):
    assert classify(exc).kind is Kind.FATAL


def test_status_is_found_through_the_cause_chain():
    try:
        try:
            raise openai_429(headers={"retry-after": "3"})
        except Exception as inner:
            raise RuntimeError("wrapped by app code") from inner
    except RuntimeError as outer:
        info = classify(outer)
    assert info.kind is Kind.RATE_LIMIT and info.retry_after == 3.0


def test_langchain_typed_error_without_any_status():
    lc = pytest.importorskip("langchain_core.exceptions")
    if not hasattr(lc, "ModelRateLimitError"):
        pytest.skip("older langchain-core")
    assert classify(lc.ModelRateLimitError("slow down")).kind is Kind.RATE_LIMIT


@pytest.mark.parametrize(
    "text,expected",
    [("34.07s", 34.07), ("6m0s", 360.0), ("20ms", 0.02), ("1h2m", 3720.0), ("1m30s", 90.0), ("nonsense", None), ("", None)],
)
def test_parse_duration(text, expected):
    got = parse_duration(text)
    assert (got is None) if expected is None else got == pytest.approx(expected)
