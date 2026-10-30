"""Lightweight admission evidence. Offline preparation by default; --yes sends calls.

Example small live check: python -m evals.admission_probe --yes
Analyze later: python -m evals.admission_probe --analyze <run-folder>
No debug stacks, HTTP payload traces, checkpoints or percentiles on the call path.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
from pathlib import Path
import queue
import threading
import time
import uuid

import yaml

from evals.live_eval import Plan, arrivals
from evals.metrics import percentile
from evals.run_live_profile import ROOT, prepare, source_metadata, write_json
from evals.sim import outcome_of
from pool_manager.errors import classify
from pool_manager.llm import get_llm

CALLER = contextvars.ContextVar("admission_probe_caller", default=None)


class Recorder:
    """Bounded nonblocking producer; encoding/redaction/file IO only on writer thread."""
    def __init__(self, path, secrets=(), capacity=50000):
        self.path = Path(path)
        self.secrets = tuple(secrets)
        self.events = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.error = None
        self.written = 0
        self.origin = time.monotonic()
        self.thread = threading.Thread(target=self._write, name="admission-writer", daemon=True)
        self.thread.start()

    def emit(self, event, **fields):
        row = dict(event=event, at_s=time.monotonic() - self.origin, caller=CALLER.get(), **fields)
        try:
            self.events.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        stream = None
        try:
            stream = self.path.open("x", encoding="utf-8")
        except Exception as exc:
            self.error = type(exc).__name__
        try:
            while True:
                item = self.events.get()
                if item is None:
                    return
                if self.error is None:
                    try:
                        line = json.dumps(item, ensure_ascii=True)
                        for secret in self.secrets:
                            line = line.replace(secret, "[REDACTED]")
                        stream.write(line + "\n")
                        stream.flush()
                        self.written += 1
                    except Exception as exc:
                        # Keep draining even after a disk error, so close cannot
                        # deadlock behind a full queue. The run becomes invalid.
                        self.error = type(exc).__name__
        finally:
            if stream is not None:
                stream.close()

    def close(self):
        self.events.put(None)  # Blocking is allowed after the workload finishes.
        self.thread.join()
        return dict(events_written=self.written, dropped=self.dropped,
                    writer_error=self.error, valid=self.error is None and self.dropped == 0)


def instrument(scheduler, recorder):
    """Evaluation-local wrappers: usable on the polling baseline and new scheduler.

    Selection snapshots run within the scheduler's admission lock. They add small
    constant work for four endpoints. They observe local eligibility, not provider
    quota availability. No request bodies, API keys or exception messages recorded.
    """
    original_join = scheduler._join
    original_admit = scheduler._admit
    original_select = scheduler._try_acquire
    original_leave = scheduler._leave
    waiter_ids = {}
    identifiers = itertools.count()

    def join(*args, **kwargs):
        waiter = original_join(*args, **kwargs)
        waiter_id = next(identifiers)
        waiter_ids[id(waiter)] = waiter_id
        recorder.emit("queue_enter", waiter=waiter_id, queue_depth=len(scheduler._waiters))
        return waiter

    def select(now, attempted=None):
        available = [(ep.id, ep.available_in(now)) for ep in scheduler.endpoints]
        endpoint, delay = original_select(now, attempted)
        recorder.emit("eligibility", endpoints=available,
                      selected=None if endpoint is None else endpoint.id, retry_after_s=delay,
                      after_endpoints=[(ep.id, ep.available_in(now)) for ep in scheduler.endpoints])
        return endpoint, delay

    def admit(waiter, now, attempted):
        before = time.monotonic()
        try:
            endpoint, delay = original_admit(waiter, now, attempted)
        except Exception as exc:
            recorder.emit("admission_rejected", waiter=waiter_ids[id(waiter)],
                          error_type=type(exc).__name__)
            raise
        recorder.emit("reserved" if endpoint is not None else "wait_armed",
                      waiter=waiter_ids[id(waiter)], endpoint=None if endpoint is None else endpoint.id,
                      planned_wait_s=delay, queue_depth=len(scheduler._waiters),
                      decision_s=time.monotonic() - before)
        return endpoint, delay

    def leave(waiter):
        original_leave(waiter)
        recorder.emit("queue_exit", waiter=waiter_ids.pop(id(waiter)),
                      queue_depth=len(scheduler._waiters))

    scheduler._join, scheduler._admit = join, admit
    scheduler._try_acquire, scheduler._leave = select, leave
    for ep in scheduler.endpoints:
        for method in ("mark_success", "mark_rate_limited", "mark_server_error", "mark_dead"):
            original = getattr(ep, method)

            def changed(*args, _original=original, _endpoint=ep, **kwargs):
                result = _original(*args, **kwargs)
                recorder.emit("state_changed", endpoint=_endpoint.id,
                              available_in_s=_endpoint.available_in(scheduler.clock()),
                              queue_depth=len(scheduler._waiters))
                return result

            setattr(ep, method, changed)


class RecordedClient:
    def __init__(self, endpoint, recorder):
        self.client, self.endpoint, self.recorder = endpoint.client, endpoint.id, recorder

    async def ainvoke(self, *args, **kwargs):
        self.recorder.emit("provider_start", endpoint=self.endpoint)
        try:
            result = await self.client.ainvoke(*args, **kwargs)
        except BaseException as exc:
            fields = dict(endpoint=self.endpoint, ok=False, error_type=type(exc).__name__)
            if isinstance(exc, Exception):
                info = classify(exc)
                fields.update(kind=info.kind.value, status=info.status)
            self.recorder.emit("provider_end", **fields)
            raise
        self.recorder.emit("provider_end", endpoint=self.endpoint, ok=True,
                           usage=getattr(result, "usage_metadata", None))
        return result


def prepare_probe(args):
    manifest, secrets = prepare("soak-1000-60", args.env_file)
    if not 1 <= args.endpoint_rpm <= 12:
        raise ValueError("endpoint RPM must be between 1 and the existing local cap 12")
    if args.calls < 1 or (args.calls > 16 and not args.allow_large):
        raise ValueError("Use 1–16 calls now; larger workloads require explicit --allow-large")
    if args.rate_per_min <= 0:
        raise ValueError("rate must be positive")
    for cfg in manifest["config"].values():
        for model in cfg["models"]:
            manifest["model_limits"]["google"][model]["rpm"] = args.endpoint_rpm
    for ep in manifest["endpoints"]:
        ep["rpm"] = args.endpoint_rpm
    plan = Plan(calls=args.calls, pattern=args.pattern, rate_per_min=args.rate_per_min, seed=1,
                request_timeout=20, max_wait=70, caller_timeout=100, max_attempts=1,
                limits_path="limits.snapshot.yaml")
    manifest.update(profile="admission-probe", plan=asdict(plan),
                    implementation_label=getattr(args, "label", "current"),
                    configured_rpm=len(manifest["endpoints"]) * args.endpoint_rpm,
                    last_scheduled_arrival_s=arrivals(plan)[-1],
                    stop_on_first_error=not args.keep_going, source=source_metadata(),
                    recorder="bounded background JSON writer; no debug diagnostics or in-run percentiles",
                    analysis_during_workload=False)
    manifest["source"]["source_sha256"]["evals/admission_probe.py"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return manifest, secrets


async def execute(manifest, folder, secrets):
    write_json(folder / "manifest.json", manifest, secrets)
    for name, data in (("config", manifest["config"]), ("limits", manifest["model_limits"])):
        (folder / f"{name}.snapshot.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    plan = Plan(**manifest["plan"])
    pool = get_llm(folder / "config.snapshot.yaml", limits_path=folder / "limits.snapshot.yaml",
                   max_wait_seconds=plan.max_wait, request_timeout=plan.request_timeout,
                   caller_timeout=plan.caller_timeout, max_attempts=plan.max_attempts)
    recorder = Recorder(folder / "events.jsonl", secrets)
    instrument(pool.scheduler, recorder)
    for ep in pool.scheduler.endpoints:
        ep.client = RecordedClient(ep, recorder)
    offsets = arrivals(plan)
    origin = recorder.origin
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)  # Error metadata is recorded; no console IO in calls.
    tasks = []
    all_completed = False

    async def caller(i):
        await asyncio.sleep(max(0, origin + offsets[i] - time.monotonic()))
        token = CALLER.set(i)
        recorder.emit("arrival", scheduled_s=offsets[i])
        outcome = "ok"
        try:
            await pool.ainvoke(plan.prompt)
        except BaseException as exc:
            outcome = outcome_of(exc) if isinstance(exc, Exception) else "cancelled"
            raise
        finally:
            recorder.emit("caller_end", outcome=outcome)
            CALLER.reset(token)

    try:
        tasks = [asyncio.create_task(caller(i)) for i in range(plan.calls)]
        if manifest["stop_on_first_error"]:
            await asyncio.gather(*tasks)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
        all_completed = True
    except Exception:
        # A small smoke test stops queued work on the first error, rather than
        # exhausting fresh quota trying to force a provider to accept traffic.
        pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.disable(previous_logging)
        recording = recorder.close()
        write_json(folder / "completion.json", dict(
            state="complete" if all_completed else "stopped_on_error_or_interruption",
            finished_utc=datetime.now(timezone.utc).isoformat(), recording=recording,
            endpoint_stats=pool.stats(), analysis_pending=True), secrets)
    return 0 if all_completed and recording["valid"] else 1


def analyze(folder):
    """Offline only. Do not construct clients or access credentials."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    completion = json.loads((folder / "completion.json").read_text())
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    entered, caller_arrivals, admission, provider_start = {}, {}, [], {}
    reserved_waiters, unsuccessful_admission = set(), []
    callers, outcomes, provider_durations, failed_attempts = {}, {}, [], []
    ready_at, queue_depth, previous_time, idle_intervals = {}, 0, 0.0, []
    for event in events:
        caller = event["caller"]
        now = event["at_s"]
        if queue_depth and ready_at:
            idle_start = max(previous_time, min(ready_at.values()))
            if idle_start < now:
                idle_intervals.append((idle_start, now))
        previous_time = now
        if "queue_depth" in event:
            queue_depth = event["queue_depth"]
        if event["event"] == "eligibility":
            ready_at = {endpoint:now + delay for endpoint, delay in event["after_endpoints"]}
        elif event["event"] == "state_changed":
            ready_at[event["endpoint"]] = now + event["available_in_s"]
        if event["event"] == "arrival":
            caller_arrivals[caller] = event
        elif event["event"] == "queue_enter":
            entered[event["waiter"]] = event
        elif event["event"] == "reserved":
            start = entered[event["waiter"]]
            reserved_waiters.add(event["waiter"])
            wait = event["at_s"] - start["at_s"]
            admission.append(dict(caller=caller, waiter=event["waiter"], endpoint=event["endpoint"],
                                  queue_s=wait, queued_at_s=start["at_s"], reserved_at_s=event["at_s"]))
        elif event["event"] == "queue_exit" and event["waiter"] not in reserved_waiters:
            unsuccessful_admission.append(now - entered[event["waiter"]]["at_s"])
        elif event["event"] == "provider_start":
            provider_start[caller] = event["at_s"]
        elif event["event"] == "provider_end":
            provider_durations.append(event["at_s"] - provider_start.pop(caller))
            if not event["ok"]:
                failed_attempts.append(event)
        elif event["event"] == "caller_end":
            outcomes[event["outcome"]] = outcomes.get(event["outcome"], 0) + 1
            callers[caller] = event["at_s"] - caller_arrivals[caller]["scheduled_s"]
    served_order = [a["waiter"] for a in admission]
    served_before, overtakes = set(), 0
    for a in served_order:
        overtakes += any(older < a and older not in served_before
                       and entered[older]["at_s"] < entered[a]["at_s"]
                       for older in served_order)
        served_before.add(a)

    # Join adjacent evidence fragments; eligibility transitions split intervals.
    merged_idle = []
    for start, end in idle_intervals:
        if merged_idle and start <= merged_idle[-1][1]:
            merged_idle[-1][1] = end
        else:
            merged_idle.append([start, end])

    def percentiles(values):
        return {f"p{p}_s": percentile(values, p) for p in (50, 95, 99)}

    result = dict(planned_calls=manifest["plan"]["calls"], outcomes=outcomes,
                  admissions=len(admission), provider_attempts=sum(e["event"] == "provider_start" for e in events),
                  failed_attempts=failed_attempts, admission_success=percentiles([a["queue_s"] for a in admission]),
                  admission_without_reservation=percentiles(unsuccessful_admission),
                  admission_all=percentiles([a["queue_s"] for a in admission] + unsuccessful_admission),
                  provider_attempt=percentiles(provider_durations), scheduled_to_result=percentiles(list(callers.values())),
                  served_callers_overtaken=overtakes, admission_sequence=admission,
                  observed_eligible_capacity_idle_with_queue_s=sum(end-start for start,end in merged_idle),
                  observed_eligible_capacity_idle_max_s=max((end-start for start,end in merged_idle), default=0),
                  recording=completion["recording"],
                  limitations=["Admission includes checking and recorder overhead; not pure sleeping time.",
                               "Admission percentiles exclude acquisitions that failed or were cancelled.",
                               "Small samples do not establish reliable tail percentiles.",
                               "Local eligibility snapshots do not verify upstream quota.",
                               "Observed idle includes checking/recording time and is reconstructed from snapshots, not pure avoidable sleep.",
                               "No claim of zero instrumentation overhead."])
    write_json(folder / "analysis.json", result, ())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--analyze", type=Path)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--pattern", choices=("burst", "paced", "poisson"), default="burst")
    parser.add_argument("--rate-per-min", type=float, default=30)
    parser.add_argument("--endpoint-rpm", type=int, default=1)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--label", choices=("current", "polling-baseline", "notified-fifo"), default="current")
    parser.add_argument("--out-root", type=Path,
                        help="Store baseline and fixed evidence together, even from an exported checkout")
    args = parser.parse_args(argv)
    if args.analyze:
        result = analyze(args.analyze)
        print(json.dumps({k:v for k,v in result.items() if k != "admission_sequence"}, indent=2))
        return 0
    manifest, secrets = prepare_probe(args)
    if not args.yes:
        print(json.dumps({k:manifest[k] for k in ("profile", "plan", "configured_rpm", "stop_on_first_error")}, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.out_root.resolve() if args.out_root else ROOT / "evals/results/admission_runs"
    folder = out_root / f"{stamp}_{args.label}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    print(f"Run folder: {folder}", flush=True)
    result = asyncio.run(execute(manifest, folder, secrets))
    print("Recording finished. Analyze separately: python -m evals.admission_probe --analyze", folder)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
