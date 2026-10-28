import asyncio

import httpx
import pytest

from evals import direct_api_eval as direct


def test_direct_rest_cases_are_exactly_once_per_endpoint_per_phase():
    requests, active, peak = [], 0, 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        requests.append(request)
        await asyncio.sleep(0)
        active -= 1
        code = 503 if len(requests) == 1 else 200
        return httpx.Response(code, json={"usageMetadata": {"totalTokenCount": 9}},
                              headers={"server-timing": "gfet4t7; dur=1000"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), event_hooks={
            "request": [direct.request_hook], "response": [direct.response_hook]}) as client:
            return await direct.run_cases(client, {name: name + "-fake" for name in direct.KEY_NAMES})

    result = asyncio.run(run())
    assert len(requests) == 12  # the failed first request was not retried
    assert sum(row["ok"] for row in result["cases"]) == 11
    assert result["peak_active_requests"] == {"sequential": 1, "concurrent": 6}
    assert peak == 6
    for phase in ("sequential", "concurrent"):
        pairs = [(r["model"], r["key_variable"]) for r in result["cases"] if r["phase"] == phase]
        assert len(pairs) == len(set(pairs)) == 6
    assert all(not request.url.query for request in requests)
    assert all(row["headers_s"] <= row["total_s"] for row in result["cases"])
    assert all("server-timing" in row["timing_headers"] for row in result["cases"])


def test_explicit_live_opt_in_required():
    with pytest.raises(SystemExit) as exc:
        direct.main([])
    assert exc.value.code == 2


def test_env_file_does_not_inherit_missing_key(tmp_path, monkeypatch):
    path = tmp_path / "keys.env"
    path.write_text("GEMINI_API_KEY=fake-one\n")
    monkeypatch.setenv("GEMINI_API_KEY2", "stale-shell-value")
    with pytest.raises(ValueError, match="Both"):
        direct.load_keys(path)
    assert "secret" not in direct.safe_json({"message": "secret"}, {"key": "secret"})
