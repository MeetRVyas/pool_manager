"""Prepared-run safeguards, artifact integrity and tracing, with fake clients only."""
import asyncio
import json
import logging
from pathlib import Path

import pytest

from evals import run_live_profile as runner
from tests.fakes import FakeClient, make_ep, make_llm


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "keys.env"
    path.write_text("GEMINI_API_KEY=fresh-one\nGEMINI_API_KEY2=fresh-two\n")
    for key in ("GEMINI_API_KEY", "GEMINI_API_KEY2"):
        monkeypatch.setenv(key, "stale-shell-key")
    monkeypatch.setattr(runner, "source_metadata", lambda: {"git_commit": "test", "git_dirty": True})
    return path


@pytest.mark.parametrize("profile", runner.PROFILES)
def test_offline_preparation_never_constructs_a_provider(profile, env_file, monkeypatch, capsys):
    def forbid(*args, **kwargs):
        raise AssertionError("Offline preparation constructed a provider")
    monkeypatch.setattr(runner, "get_llm", forbid)
    rc = runner.main(["--profile", profile, "--env-file", str(env_file)])
    output = capsys.readouterr().out
    assert rc == 0 and "No provider clients constructed" in output
    assert "fresh-one" not in output and "fresh-two" not in output
    manifest, secrets = runner.prepare(profile, env_file)
    assert secrets == ["fresh-one", "fresh-two"]
    assert manifest["mode"] == "pool"
    assert len(manifest["endpoints"]) == (2 if profile.startswith("lite35-") else 4)


def test_single_model_profile_and_prepared_overload_have_distinct_limits(env_file):
    single, _ = runner.prepare("lite35-1000-30rpm", env_file)
    assert single["plan"]["calls"] == 1000
    assert single["plan"]["rate_per_min"] == 30
    assert single["config"]["google"]["models"] == ["gemini-3.5-flash-lite"]
    assert single["configured_rpm"] == 30
    assert single["configured_daily_attempts"] == 1000
    headroom, _ = runner.prepare("lite35-100-24rpm", env_file)
    assert headroom["plan"]["calls"] == 100
    assert headroom["plan"]["rate_per_min"] == 24
    assert headroom["plan"]["request_timeout"] == 60
    assert headroom["config"] == single["config"]
    assert headroom["model_limits"] == single["model_limits"]
    overload, _ = runner.prepare("lites-2000-60rpm", env_file)
    assert overload["plan"]["calls"] == 2000
    assert overload["plan"]["rate_per_min"] == 60
    assert overload["configured_rpm"] == 48
    assert overload["configured_daily_attempts"] == 1960
    adjusted, _ = runner.prepare("lites-1900-60rpm", env_file)
    assert adjusted["plan"]["calls"] == 1900
    assert adjusted["plan"]["rate_per_min"] == 60
    assert adjusted["configured_daily_attempts"] == 1960
    assert adjusted["configured_rpm"] == 48
    matched, _ = runner.prepare("lites-1900-48rpm", env_file)
    assert matched["plan"]["calls"] == 1900
    assert matched["plan"]["rate_per_min"] == matched["configured_rpm"] == 48
    assert matched["plan"]["pattern"] == "paced"
    assert matched["last_scheduled_arrival_s"] == 2373.75
    aligned, _ = runner.prepare("lites-1400-36rpm", env_file)
    assert aligned["plan"]["calls"] == 1400
    assert aligned["plan"]["rate_per_min"] == 36 < aligned["configured_rpm"]
    assert aligned["plan"]["pattern"] == "paced"
    assert aligned["configured_daily_attempts"] - aligned["plan"]["calls"] == 560
    original, _ = runner.prepare("soak-1000-60", env_file)
    assert original["configured_rpm"] == 48  # single-model caps do not leak into older experiments


def test_missing_env_key_does_not_fall_back_to_stale_shell_value(env_file, monkeypatch, capsys):
    env_file.write_text("GEMINI_API_KEY=fresh-one\n")
    monkeypatch.setattr(runner, "get_llm", lambda *a, **k: pytest.fail("Provider constructed"))
    assert runner.main(["--prepare", "--env-file", str(env_file)]) == 2
    assert "GEMINI_API_KEY2" in capsys.readouterr().err


def test_duplicate_keys_are_rejected_offline(env_file, capsys):
    env_file.write_text("GEMINI_API_KEY=duplicate\nGEMINI_API_KEY2=duplicate\n")
    assert runner.main(["--prepare", "--env-file", str(env_file)]) == 2
    assert "duplicate key values" in capsys.readouterr().err


def test_explicit_live_mode_writes_complete_traces_with_fake_clients(env_file, tmp_path, monkeypatch):
    monkeypatch.setitem(runner.PROFILES, "tiny", dict(calls=3, request_timeout=30, pattern="burst", config="lites.yaml"))
    def fake_pool(*args, **kwargs):
        return make_llm([make_ep("google:gemini-3.5-flash-lite:GEMINI_API_KEY", FakeClient("fake"))])
    monkeypatch.setattr(runner, "get_llm", fake_pool)
    monkeypatch.setattr(runner, "configure_logging", lambda *a: None)
    out = tmp_path / "runs"
    argv = ["--profile", "tiny", "--env-file", str(env_file), "--output-root", str(out), "--yes"]
    assert runner.main(argv) == 0
    assert runner.main(argv) == 0  # same label never overwrites a previous run
    folders = list(out.iterdir())
    assert len(folders) == 2
    for folder in folders:
        assert (folder / "manifest.json").is_file()
        assert (folder / "config.snapshot.yaml").is_file()
        assert (folder / "limits.snapshot.yaml").is_file()
        assert (folder / "summary.md").is_file()
        result = json.loads((folder / "results.json").read_text())
        assert result["state"] == "complete" and result["completed_calls"] == 3
        rows = result["results"]["pool"]["calls"]
        assert all(r["outcome"] == "ok" and len(r["attempts"]) == 1 for r in rows)
        assert all(r["scheduler_s"] >= 0 and r["provider_s"] >= 0 and r["other_s"] >= 0 for r in rows)
        assert len((folder / "calls.jsonl").read_text().splitlines()) == 3
        assert result["windows"][0]["successes"] == 3
        assert result["source"]["git_commit"] == "test"


def test_opt_in_diagnostics_and_timing_boundaries_with_fake_provider(env_file, tmp_path, monkeypatch):
    monkeypatch.setitem(runner.PROFILES, "diagnostic-tiny",
                        dict(calls=2, request_timeout=30, pattern="burst", config="lites.yaml"))
    monkeypatch.setattr(runner, "get_llm", lambda *a, **k: make_llm([
        make_ep("google:gemini-3.5-flash-lite:GEMINI_API_KEY", FakeClient("offline"))]))
    monkeypatch.setattr(runner, "configure_logging", lambda *a: None)
    out = tmp_path / "diagnostic-runs"
    assert runner.main(["--profile", "diagnostic-tiny", "--env-file", str(env_file),
                        "--output-root", str(out), "--yes", "--diagnostics"]) == 0
    folder = next(out.iterdir())
    result = json.loads((folder / "results.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["diagnostics_enabled"] is True
    assert manifest["http_backends"]
    summary = result["results"]["pool"]
    assert summary["primary_latency_boundary"] == "scheduled_arrival_to_result"
    assert summary["timing_boundaries"]["loop_stall_measurement_enabled"] is True
    for row in summary["calls"]:
        assert row["scheduled_to_result_s"] >= row["dispatch_lateness_s"]
        assert isinstance(row["overlaps_loop_stall"], bool)
        assert isinstance(row["attempts"][0]["overlaps_loop_stall"], bool)
    assert (folder / "diagnostics.summary.json").exists()


def test_failed_provider_initialization_leaves_checkpoint(env_file, tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("do not log credentials")
    monkeypatch.setattr(runner, "get_llm", fail)
    monkeypatch.setattr(runner, "configure_logging", lambda *a: None)
    out = tmp_path / "runs"
    assert runner.main(["--env-file", str(env_file), "--output-root", str(out), "--yes"]) == 2
    saved = json.loads(next(out.glob("*/checkpoint.json")).read_text())
    assert saved["state"] == "interrupted_or_failed"
    assert saved["completed_calls"] == 0


def test_json_and_log_filter_redact_values(tmp_path):
    path = tmp_path / "result.json"
    runner.write_json(path, {"error": "key=super-secret"}, ["super-secret"])
    assert "super-secret" not in path.read_text()
    record = logging.LogRecord("test", logging.WARNING, "", 0, "key=%s", ("super-secret",), None)
    assert runner.SecretFilter(["super-secret"]).filter(record)
    assert record.getMessage() == "key=[REDACTED]"


def test_cancelled_call_preserves_partial_artifacts(env_file, tmp_path, monkeypatch):
    monkeypatch.setitem(runner.PROFILES, "cancel", dict(calls=1, request_timeout=60, pattern="burst", config="lites.yaml"))
    manifest, secrets = runner.prepare("cancel", env_file)
    async def interrupted():
        entered = asyncio.Event()
        class WaitingClient(FakeClient):
            async def ainvoke(self, *args, **kwargs):
                entered.set()
                await asyncio.Future()
        pool = make_llm([make_ep("google:fake:key", WaitingClient("fake"))])
        monkeypatch.setattr(runner, "get_llm", lambda *a, **kw: pool)
        task = asyncio.create_task(runner.execute(manifest, tmp_path, secrets, 10))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(interrupted())
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["state"] == "interrupted_or_failed"
    assert saved["completed_calls"] == 1
    assert saved["calls"][0]["outcome"] == "cancelled"
    journal = json.loads((tmp_path / "calls.jsonl").read_text())
    assert journal["attempts"][0]["kind"] == "cancelled"
    assert not (tmp_path / "results.json").exists()
