import asyncio

from evals.audit_wrapper_overhead import audit


def test_wrapper_audit_stays_offline_and_exposes_actual_request_settings():
    result = asyncio.run(audit(iterations=3, warmup=1))
    assert result["network_requests"] == 0
    assert result["mock_requests"] == {
        "raw_http": 4, "google_sdk": 4, "langchain": 4,
        "pool_plus_langchain": 4, "langchain_503_probe": 1,
    }
    payloads = result["payloads_without_headers"]
    assert payloads["langchain"] == payloads["pool_plus_langchain"]
    assert payloads["langchain"]["generationConfig"] == {"candidateCount": 1}
    assert "thinkingConfig" not in payloads["langchain"]["generationConfig"]
    assert result["max_retries_zero_503_probe"]["mock_http_requests"] == 1
    assert result["max_retries_zero_503_probe"]["error_type"]
