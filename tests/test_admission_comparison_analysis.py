"""The offline comparison must reject incompatible evidence, not just parse it."""
import copy
import json

from evals import analyze_admission_comparison as comparison


def summary(role, valid=True):
    return dict(valid=valid, implementation_label=role, outcomes={"ok":1000},
                plan={"calls":1000, "rate_per_min":48}, config={"models":["lite"]},
                model_limits={"rpm":12}, source={"dependencies":{"sdk":"1"},
                "source_sha256":{"evals/admission_probe.py":"same-recorder"}})


def test_comparison_refuses_mismatched_arrivals_even_with_valid_recordings(tmp_path, monkeypatch):
    baseline = summary("polling-baseline")
    fixed = summary("notified-fifo")
    fixed["plan"]["rate_per_min"] = 30
    monkeypatch.setattr(comparison, "summarize", lambda p: baseline if str(p)=="base" else fixed)
    out = tmp_path / "comparison.json"
    assert comparison.main(["--baseline","base","--fixed","fixed","--out",str(out)]) == 1
    result = json.loads(out.read_text())
    assert not result["comparability"]["plans_equal"]
    assert result["comparability"]["both_recordings_complete_and_valid"]


def test_comparison_requires_matching_recorder_and_correct_roles(tmp_path, monkeypatch):
    baseline = summary("polling-baseline")
    fixed = summary("notified-fifo")
    monkeypatch.setattr(comparison, "summarize", lambda p: baseline if str(p)=="base" else fixed)
    args = ["--baseline","base","--fixed","fixed","--out",str(tmp_path/"comparison.json")]
    assert comparison.main(args) == 0
    fixed["source"]["source_sha256"]["evals/admission_probe.py"] = "changed-recorder"
    assert comparison.main(args) == 1
    fixed["source"] = copy.deepcopy(baseline["source"])
    fixed["implementation_label"] = "polling-baseline"
    assert comparison.main(args) == 1
