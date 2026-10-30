"""Offline comparison only: analyze completed baseline/fixed admission recordings."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evals.admission_probe import analyze
from evals.metrics import percentile


def quantiles(values):
    return {f"p{p}_s": percentile(values, p) for p in (50, 95, 99)} | {
        "max_s": max(values, default=None), "sum_s": sum(values), "n": len(values)}


def summarize(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    completion = json.loads((folder / "completion.json").read_text())
    core = analyze(folder)  # Runs only after completion, never alongside requests.
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    arrivals, queued, exits, caller_ends = {}, {}, {}, {}
    attempts = Counter()
    providers = {}
    successful_provider, failed_provider, end_to_end_success = [], [], []
    errors = Counter()
    for e in events:
        name, caller = e["event"], e["caller"]
        if name == "arrival":
            arrivals[caller] = e
        elif name == "queue_enter":
            queued[e["waiter"]] = e
        elif name == "queue_exit":
            exits[e["waiter"]] = e
        elif name == "provider_start":
            attempts[caller] += 1
            providers[caller] = e
        elif name == "provider_end":
            start = providers.pop(caller)
            duration = e["at_s"] - start["at_s"]
            if e["ok"]:
                successful_provider.append(duration)
            else:
                failed_provider.append(duration)
                errors[f"{e.get('kind', e['error_type'])}:{e.get('status')}"] += 1
        elif name == "caller_end":
            caller_ends[caller] = e
            if e["outcome"] == "ok":
                end_to_end_success.append(e["at_s"] - arrivals[caller]["scheduled_s"])
    integrity = dict(
        state_complete=completion["state"] == "complete",
        all_planned_arrived=len(arrivals) == manifest["plan"]["calls"],
        all_planned_finished=len(caller_ends) == manifest["plan"]["calls"],
        all_queue_entries_exited=set(queued) == set(exits),
        all_provider_starts_ended=not providers,
        recorder_valid=completion["recording"]["valid"],
        recorded_event_count_matches=completion["recording"]["events_written"] == len(events),
        single_attempt_policy_observed=max(attempts.values(), default=0) <= 1)
    dispatch = [max(0.0, e["at_s"] - e["scheduled_s"]) for e in arrivals.values()]
    reservation_times = [a["queue_s"] for a in core["admission_sequence"]]
    reserved_ids = {a["waiter"] for a in core["admission_sequence"]}
    failed_queue = [exits[w]["at_s"] - q["at_s"] for w, q in queued.items() if w not in reserved_ids]
    queue_total = sum(exits[w]["at_s"] - q["at_s"] for w, q in queued.items())
    return dict(
        folder=folder.as_posix(), implementation_label=manifest.get("implementation_label"),
        plan=manifest["plan"], config=manifest["config"], model_limits=manifest["model_limits"],
        source=manifest["source"], finished_utc=completion["finished_utc"],
        integrity=integrity, valid=all(integrity.values()), recording=completion["recording"],
        event_counts=dict(Counter(e["event"] for e in events)),
        outcomes=core["outcomes"], provider_attempts=core["provider_attempts"],
        provider_errors=dict(errors), callers_retried=sum(n > 1 for n in attempts.values()),
        admissions=len(reservation_times), failed_acquisitions=len(failed_queue),
        admission_reserved=quantiles(reservation_times), admission_not_reserved=quantiles(failed_queue),
        cumulative_all_admission_s=queue_total,
        dispatch_lateness_all=quantiles(dispatch), scheduled_to_result_success=quantiles(end_to_end_success),
        provider_success=quantiles(successful_provider), provider_failed=quantiles(failed_provider),
        served_callers_overtaken=core["served_callers_overtaken"],
        observed_eligible_capacity_idle_with_queue_s=core["observed_eligible_capacity_idle_with_queue_s"],
        observed_eligible_capacity_idle_max_s=core["observed_eligible_capacity_idle_max_s"],
        per_endpoint=completion["endpoint_stats"],
        limitations=core["limitations"] + [
            "Queue-reservation percentiles include callers whose provider request subsequently failed.",
            "Idle intervals may include event-loop/host delays; they are not exclusively polling delay.",
            "Sequential provider runs are not a randomized causal experiment."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fixed", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = summarize(args.baseline)
    fixed = summarize(args.fixed) if args.fixed else None
    comparison = None
    if fixed:
        comparison = dict(
            plans_equal=baseline["plan"] == fixed["plan"],
            configs_equal=baseline["config"] == fixed["config"],
            local_limits_equal=baseline["model_limits"] == fixed["model_limits"],
            dependencies_equal=baseline["source"]["dependencies"] == fixed["source"]["dependencies"],
            recorder_hash_equal=baseline["source"]["source_sha256"]["evals/admission_probe.py"]
                == fixed["source"]["source_sha256"]["evals/admission_probe.py"],
            expected_roles=baseline["implementation_label"] == "polling-baseline"
                and fixed["implementation_label"] == "notified-fifo",
            both_recordings_complete_and_valid=baseline["valid"] and fixed["valid"])
    result = dict(baseline=baseline, fixed=fixed, comparability=comparison,
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  status="paired_recordings_analyzed" if fixed else "baseline_complete_fixed_pending_fresh_quota")
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "baseline_valid": baseline["valid"],
                      "baseline_outcomes": baseline["outcomes"], "comparability": comparison}, indent=2))
    return 0 if baseline["valid"] and (comparison is None or all(comparison.values())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
