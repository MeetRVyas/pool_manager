"""Offline percentiles and latency decomposition of saved live artifacts."""
import json
from pathlib import Path

from evals.metrics import percentile

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    return {"n": len(values), **{f"p{q}": percentile(values, q) for q in (50, 95, 99)}}


def analyze():
    direct_path = ROOT / "evals/results/direct_runs/20261027T162046Z_4d25067d/results.json"
    direct = json.loads(direct_path.read_text(encoding="utf-8"))["cases"]
    groups = {"all successful direct calls": [r for r in direct if r["ok"]],
              "all direct calls including HTTP errors": direct}
    for field in ("phase", "model"):
        for value in sorted({r[field] for r in direct}):
            groups[value] = [r for r in direct if r["ok"] and r[field] == value]
    raw = {name: stats([r["total_s"] for r in rows]) for name, rows in groups.items()}
    pool = {}
    for path in sorted((ROOT / "evals/results/runs").glob("*/results.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = saved["calls"]
        ok = [r for r in rows if r["outcome"] == "ok"]
        clean = [r for r in ok if len(r["attempts"]) == 1 and r["attempts"][0]["ok"]]
        pool[saved["profile"]] = {
            "source": path.relative_to(ROOT).as_posix(),
            "all_successful": stats([r["latency_s"] for r in ok]),
            "successful_without_retry": stats([r["latency_s"] for r in clean]),
            "clean_components": {field: stats([r[field] for r in clean])
                                 for field in ("scheduler_s", "provider_s", "other_s")},
            "all_components": {field: stats([r[field] for r in rows])
                               for field in ("scheduler_s", "provider_s", "other_s")},
            "clean_provider_by_model": {
                model: stats([r["provider_s"] for r in clean if r["attempts"][0]["endpoint"].split(":")[1] == model])
                for model in sorted({r["attempts"][0]["endpoint"].split(":")[1] for r in clean})},
            "failures": len(rows) - len(ok),
            "arrival_dispatch_delay_s": {
                **stats([max(0, r["actual_start_offset_s"] - r["arrival_offset_s"]) for r in rows]),
                "max": max((max(0, r["actual_start_offset_s"] - r["arrival_offset_s"]) for r in rows), default=None),
            },
            "scheduled_arrival_to_result_successful": {
                **stats([r["end_offset_s"] - r["arrival_offset_s"] for r in ok]),
                "max": max((r["end_offset_s"] - r["arrival_offset_s"] for r in ok), default=None),
            },
        }
    return {"percentile_method": "linear interpolation, index (n-1)*q/100",
            "units": "seconds", "direct": raw, "pool": pool,
            "caveat": "Component percentiles do not add to end-to-end percentiles. "
                      "provider_s includes endpoint LangChain/SDK/network/upstream time."}


def main():
    result = analyze()
    path = ROOT / "evals/results/LATENCY_ANALYSIS_2026-10-27.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
