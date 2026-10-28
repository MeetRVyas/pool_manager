import asyncio

from evals.audit_async_concurrency import audit


def test_async_audit_overlaps_requests_without_network():
    result = asyncio.run(audit(concurrencies=(1, 4), rounds=1, delay_s=0.02))
    assert result["network_requests"] == 0
    assert result["mock_http_requests"] == 12  # two paths, each one warmup + 1 + 4
    for row in result["cases"]:
        assert row["samples"] == row["concurrency"]
        assert row["peak_transport_overlap"] == row["concurrency"]
        assert row["caller_ms"]["p50"] >= 20
