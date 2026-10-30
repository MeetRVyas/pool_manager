import argparse
import asyncio
import json
import threading

import pytest

from evals import admission_probe as probe
from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler
from tests.fakes import make_ep


def args_for(tmp_path, **overrides):
    env = tmp_path / "test.env"
    env.write_text("GEMINI_API_KEY=offline-one\nGEMINI_API_KEY2=offline-two\n")
    values = dict(env_file=str(env), calls=8, pattern="burst", rate_per_min=30,
                  endpoint_rpm=1, allow_large=False, keep_going=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_prepare_is_offline_and_does_not_change_application_caps(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No provider clients during preparation")
    monkeypatch.setattr(probe, "get_llm", forbidden)
    manifest, secrets = probe.prepare_probe(args_for(tmp_path))
    assert manifest["configured_rpm"] == 4
    assert manifest["plan"]["max_attempts"] == 1
    assert manifest["plan"]["calls"] == 8
    assert secrets == ["offline-one", "offline-two"]
    assert manifest["analysis_during_workload"] is False
    assert manifest["stop_on_first_error"]


@pytest.mark.parametrize("overrides", [dict(calls=1000), dict(calls=0), dict(endpoint_rpm=13),
                                       dict(endpoint_rpm=0), dict(rate_per_min=0)])
def test_unsafe_probe_arguments_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        probe.prepare_probe(args_for(tmp_path, **overrides))


def test_background_recorder_redacts_and_detects_writer_failure(tmp_path):
    record = probe.Recorder(tmp_path / "events.jsonl", secrets=["private-value"])
    record.emit("test", data="private-value")
    status = record.close()
    assert status["valid"] and status["events_written"] == 1
    text = (tmp_path / "events.jsonl").read_text()
    assert "private-value" not in text and "[REDACTED]" in text
    broken = probe.Recorder(tmp_path / "missing" / "events.jsonl")
    broken.emit("test")
    assert broken.close()["valid"] is False


def test_execute_uses_real_pool_and_records_no_file_io_on_loop_thread(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    manifest, secrets = probe.prepare_probe(args)
    folder = tmp_path / "run"
    folder.mkdir()
    eps = [make_ep(f"g:m:K{i}") for i in range(4)]
    pool = MultiProviderChatLLM(scheduler=LLMScheduler(eps), max_attempts=1)
    monkeypatch.setattr(probe, "get_llm", lambda *args, **kwargs: pool)
    from pathlib import Path
    original_open = Path.open
    loop_thread = threading.get_ident()
    writes = []

    def observed_open(path, *args, **kwargs):
        if path.name == "events.jsonl" and args and args[0] == "x":
            writes.append(threading.get_ident())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    assert asyncio.run(probe.execute(manifest, folder, secrets)) == 0
    assert writes and all(i != loop_thread for i in writes)
    assert not (folder / "analysis.json").exists()
    result = probe.analyze(folder)
    assert result["outcomes"] == {"ok": 8}
    assert result["provider_attempts"] == result["admissions"] == 8
    assert len(set(a["waiter"] for a in result["admission_sequence"])) == 8
    assert result["served_callers_overtaken"] == 0
    assert result["recording"]["valid"]
    assert (folder / "analysis.json").is_file()
    assert "offline-one" not in (folder / "events.jsonl").read_text()
    rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "state_changed" for e in rows) == 8


def test_recording_overflow_is_visible_instead_of_blocking_producer(tmp_path, monkeypatch):
    gate = threading.Event()
    started = threading.Event()
    original = probe.Recorder._write

    def delayed(recorder):
        started.set()
        gate.wait(2)
        original(recorder)

    monkeypatch.setattr(probe.Recorder, "_write", delayed)
    record = probe.Recorder(tmp_path / "overflow.jsonl", capacity=1)
    assert started.wait(2)
    try:
        record.emit("first")
        record.emit("second")
        assert record.dropped == 1
    finally:
        gate.set()
    status = record.close()
    assert not status["valid"] and status["dropped"] == 1
