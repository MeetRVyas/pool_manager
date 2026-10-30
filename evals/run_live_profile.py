"""Prepare offline or explicitly execute one pool-only evaluation profile."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

from dotenv import load_dotenv
import yaml

from evals.live_eval import Plan, arrivals, summarize_live, _fmt
from evals.metrics import percentile
from evals.loop_diagnostics import CALL as DIAGNOSTIC_CALL, LoopDiagnostics, backend_metadata, timing_summary
from evals.sim import _drive, outcome_of
from pool_manager.errors import classify
from pool_manager.limits import LimitsCatalogue
from pool_manager.llm import get_llm

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "diagnostic-paced-100": dict(calls=100, request_timeout=30, caller_timeout=90, max_attempts=2,
                                  pattern="paced", config="lites.yaml", rate_per_min=30),
    "diagnostic-burst-100": dict(calls=100, request_timeout=30, caller_timeout=90, max_attempts=2,
                                  pattern="burst", config="lites.yaml", rate_per_min=30),
    "diagnostic-12-12rpm": dict(calls=12, request_timeout=30, caller_timeout=90, max_attempts=2,
                                pattern="paced", config="lites.yaml", rate_per_min=12),
    "lites-1900-48rpm": dict(calls=1900, request_timeout=60, pattern="paced", config="lites.yaml",
                                rate_per_min=48),
    "lites-1400-36rpm": dict(calls=1400, request_timeout=60, pattern="paced", config="lites.yaml",
                                rate_per_min=36),
    "lites-1900-60rpm": dict(calls=1900, request_timeout=60, pattern="poisson", config="lites.yaml",
                                rate_per_min=60),
    "lite35-100-24rpm": dict(calls=100, request_timeout=60, pattern="poisson", config="lite35.yaml",
                                limits="lite35_full_limits.yaml", rate_per_min=24),
    "lite35-1000-30rpm": dict(calls=1000, request_timeout=60, pattern="poisson", config="lite35.yaml",
                                 limits="lite35_full_limits.yaml", rate_per_min=30),
    "lites-2000-60rpm": dict(calls=2000, request_timeout=60, pattern="poisson", config="lites.yaml",
                                rate_per_min=60),
    "soak-1000-60": dict(calls=1000, request_timeout=60, pattern="poisson", config="lites.yaml"),
    "lite-100-30": dict(calls=100, request_timeout=30, pattern="poisson", config="lites.yaml"),
    "lite-100-60": dict(calls=100, request_timeout=60, pattern="poisson", config="lites.yaml"),
    "lite-burst-100": dict(calls=100, request_timeout=60, pattern="burst", config="lites.yaml"),
    "flash-12-60": dict(calls=12, request_timeout=60, pattern="poisson", config="flash.yaml"),
}
CURRENT = contextvars.ContextVar("live_profile_call", default=None)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_metadata():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL,
                                           text=True, encoding="utf-8").strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    paths = sorted((ROOT / "pool_manager").glob("*.py")) + [
        Path(__file__), ROOT / "evals/live_eval.py", ROOT / "evals/sim.py",
        ROOT / "evals/loop_diagnostics.py", ROOT / "evals/metrics.py",
    ]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    versions = {}
    for name in ("langchain-core", "langchain-google-genai", "google-genai", "langchain-openai",
                 "python-dotenv", "pyyaml", "httpx", "httpcore", "httpx2", "truststore"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    status = git("status", "--porcelain")
    return dict(git_commit=git("rev-parse", "HEAD"), git_dirty=None if status is None else bool(status),
                source_sha256=hashes, python=platform.python_version(), dependencies=versions)


def prepare(profile, env_file):
    """No clients are constructed and no network requests are made here."""
    selected = PROFILES[profile]
    config_path = ROOT / "configs/evals" / selected["config"]
    limits_path = ROOT / "configs/evals" / selected.get("limits", "model_limits.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    limits_data = yaml.safe_load(limits_path.read_text(encoding="utf-8"))
    catalogue = LimitsCatalogue.from_dict(limits_data)
    env_file = Path(env_file).resolve()
    if not env_file.is_file():
        raise ValueError(f"Environment file not found: {env_file}")
    names = [name for cfg in config.values() for name in cfg["keys"]]
    # Isolate these configured credentials from stale inherited shell variables.
    for name in names:
        os.environ.pop(name, None)
    load_dotenv(env_file, override=True)
    if any(not os.environ.get(name) for name in names):
        missing = [name for name in names if not os.environ.get(name)]
        raise ValueError("Missing keys in the selected env file: " + ", ".join(missing))
    secrets = list(dict.fromkeys(os.environ[name] for name in names))
    if len(secrets) != len(set(names)):
        raise ValueError("Configured key variables contain duplicate key values; distinct keys are required")
    endpoints = []
    for provider, cfg in config.items():
        for model in cfg["models"]:
            lim = catalogue.lookup(provider, model)
            if lim is None or lim.rpm is None or lim.rpd is None:
                raise ValueError(f"Explicit RPM and RPD caps required: {provider}/{model}")
            for name in cfg["keys"]:
                endpoints.append(dict(id=f"{provider}:{model}:{name}", rpm=lim.rpm, tpm=lim.tpm,
                                      rpd=lim.rpd, daily_reset=lim.daily_reset))
    plan = Plan(calls=selected["calls"], request_timeout=selected["request_timeout"],
                caller_timeout=selected.get("caller_timeout"), max_attempts=selected.get("max_attempts", 6),
                rate_per_min=selected.get("rate_per_min", 30), pattern=selected["pattern"], seed=1, max_wait=70,
                limits_path="limits.snapshot.yaml")
    offsets = arrivals(plan)
    manifest = dict(profile=profile, plan=asdict(plan), config=config, model_limits=limits_data,
                    endpoints=endpoints, configured_rpm=sum(e["rpm"] for e in endpoints),
                    configured_daily_attempts=sum(e["rpd"] for e in endpoints),
                    last_scheduled_arrival_s=offsets[-1],
                    quota_note="Local experimental caps; actual project quotas and prior use are not verified. "
                               "Capacity sums assume separate projects and independent model quotas.",
                    mode="pool", source=source_metadata())
    return manifest, secrets


def write_json(path, data, secrets):
    text = json.dumps(data, indent=2, ensure_ascii=True)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class SecretFilter(logging.Filter):
    def __init__(self, secrets):
        super().__init__()
        self.secrets = secrets

    def filter(self, record):
        text = record.getMessage()
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        record.msg, record.args = text, ()
        return True


def configure_logging(folder, secrets):
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(folder / "console.log", encoding="utf-8")]
    for handler in handlers:
        handler.addFilter(SecretFilter(secrets))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    # Preserve warnings, without full request logging from transport libraries.
    for name in ("httpx", "httpx2", "google_genai"):
        logging.getLogger(name).setLevel(logging.WARNING)


class ObservedClient:
    def __init__(self, endpoint, origin=0.0):
        self.client, self.endpoint_id = endpoint.client, endpoint.id
        self.origin = origin

    async def ainvoke(self, *args, **kwargs):
        row = CURRENT.get()
        start = time.perf_counter()
        attempt = {"endpoint": self.endpoint_id, "start_offset_s": time.monotonic() - self.origin,
                   "overlaps_loop_stall": None}
        try:
            reply = await self.client.ainvoke(*args, **kwargs)
            attempt.update(ok=True, usage=getattr(reply, "usage_metadata", None))
            return reply
        except BaseException as exc:
            attempt["ok"] = False
            if isinstance(exc, Exception):
                info = classify(exc)
                attempt.update(kind=info.kind.value, status=info.status, error_type=type(exc).__name__)
            else:
                attempt["kind"] = "cancelled"
            raise
        finally:
            attempt["duration_s"] = time.perf_counter() - start
            attempt["end_offset_s"] = time.monotonic() - self.origin
            if row is not None:
                row["attempts"].append(attempt)


def windows(rows, size=100):
    groups = []
    for start in range(0, max((r["id"] for r in rows), default=-1) + 1, size):
        group = [r for r in rows if start <= r["id"] < start + size]
        if not group:
            continue
        ok = [r["latency_s"] for r in group if r["outcome"] == "ok"]
        groups.append(dict(first_call=start + 1, last_call=min(start + size, max(r["id"] for r in group) + 1),
                           completed=len(group), successes=len(ok), p50=percentile(ok, 50), p95=percentile(ok, 95),
                           scheduler_s=sum(r["scheduler_s"] for r in group),
                           provider_s=sum(a["duration_s"] for r in group for a in r["attempts"]),
                           provider_attempts=sum(len(r["attempts"]) for r in group)))
    return groups


async def execute(manifest, folder, secrets, progress_every, *, diagnostics=False, stack_dumps=False):
    write_json(folder / "manifest.json", manifest, secrets)
    for name, data in (("config.snapshot.yaml", manifest["config"]), ("limits.snapshot.yaml", manifest["model_limits"])):
        (folder / name).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    plan = Plan(**manifest["plan"])
    completed, active, peak = [], 0, 0
    checkpoint = dict(manifest, state="starting", started_at_utc=utc_now(), results=None)
    started = time.perf_counter()
    last_checkpoint = started
    pool = None
    diag = None

    def save(state):
        checkpoint.update(state=state, updated_at_utc=utc_now(), wall_seconds=time.perf_counter() - started,
                          completed_calls=len(completed), max_concurrent_calls=peak,
                          calls=sorted(completed, key=lambda r: r["id"]), windows=windows(completed),
                          per_endpoint=pool.stats() if pool is not None else [])
        write_json(folder / "checkpoint.json", checkpoint, secrets)

    save("starting")
    try:
        pool = get_llm(folder / "config.snapshot.yaml", limits_path=folder / "limits.snapshot.yaml",
                       max_wait_seconds=plan.max_wait, request_timeout=plan.request_timeout,
                       caller_timeout=plan.caller_timeout, max_attempts=plan.max_attempts)
        loop = asyncio.get_running_loop()
        origin = loop.time()
        if diagnostics:
            diag = LoopDiagnostics(folder, origin=origin, secrets=secrets, stack_dumps=stack_dumps)
            await diag.start()
        backends = {}
        for ep in pool.scheduler.endpoints:
            backend, metadata = backend_metadata(ep.client)
            metadata["trace_attached"] = bool(diag is not None and backend is not None
                                               and hasattr(backend, "event_hooks"))
            backends[ep.id] = metadata
            if metadata["trace_attached"]:
                diag.attach_httpx(backend, ep.id)
            ep.client = ObservedClient(ep, origin)
        manifest.update(http_backends=backends, loop_type=type(loop).__name__,
                        diagnostics_enabled=diagnostics, diagnostic_stack_dumps=stack_dumps)
        checkpoint.update(manifest)
        write_json(folder / "manifest.json", manifest, secrets)
        acquire = pool.scheduler.acquire

        async def observed_acquire(*args, **kwargs):
            before = time.perf_counter()
            try:
                return await acquire(*args, **kwargs)
            finally:
                row = CURRENT.get()
                if row is not None:
                    row["scheduler_s"] += time.perf_counter() - before

        pool.scheduler.acquire = observed_acquire
        offsets = arrivals(plan)

        async def call(i):
            nonlocal active, peak, last_checkpoint
            row = dict(id=i, arrival_offset_s=offsets[i], actual_start_offset_s=loop.time() - origin,
                       scheduler_s=0.0, attempts=[], outcome="ok")
            token = CURRENT.set(row)
            diagnostic_token = DIAGNOSTIC_CALL.set(i)
            before = time.perf_counter()
            active += 1
            peak = max(peak, active)
            try:
                return await pool.ainvoke(plan.prompt)
            except BaseException as exc:
                row["outcome"] = outcome_of(exc) if isinstance(exc, Exception) else "cancelled"
                raise
            finally:
                row.update(latency_s=time.perf_counter() - before, end_offset_s=loop.time() - origin)
                row["dispatch_lateness_s"] = max(0.0, row["actual_start_offset_s"] - row["arrival_offset_s"])
                row["scheduled_to_result_s"] = row["end_offset_s"] - row["arrival_offset_s"]
                row["overlaps_loop_stall"] = None if diag is None else diag.overlaps(
                    row["arrival_offset_s"], row["end_offset_s"])
                row["dispatch_overlaps_loop_stall"] = None if diag is None else diag.overlaps(
                    row["arrival_offset_s"], row["actual_start_offset_s"])
                row["provider_s"] = sum(a["duration_s"] for a in row["attempts"])
                row["other_s"] = max(0, row["latency_s"] - row["scheduler_s"] - row["provider_s"])
                active -= 1
                CURRENT.reset(token)
                DIAGNOSTIC_CALL.reset(diagnostic_token)
                completed.append(row)
                # Append each completed caller, without rewriting a growing full JSON
                # file on every completion and adding avoidable event-loop stalls.
                line = json.dumps(row, ensure_ascii=True)
                for secret in secrets:
                    line = line.replace(secret, "[REDACTED]")
                with (folder / "calls.jsonl").open("a", encoding="utf-8") as journal:
                    journal.write(line + "\n")
                if len(completed) % progress_every == 0 or time.perf_counter() - last_checkpoint >= 30:
                    save("running")
                    last_checkpoint = time.perf_counter()
                if len(completed) % progress_every == 0 or len(completed) == plan.calls:
                    logging.info("Completed %s/%s; successes=%s; active=%s; peak=%s", len(completed), plan.calls,
                                 sum(r["outcome"] == "ok" for r in completed), active, peak)

        logging.info("Starting pool-only profile %s: %s calls, %s, timeout=%ss, rate=%s/min", manifest["profile"],
                     plan.calls, plan.pattern, plan.request_timeout, plan.rate_per_min)
        results = await _drive(call, [origin + offset for offset in offsets])
        if diag is not None:
            await diag.aclose()
            # An overlapping heartbeat may be recorded after a call's journal row.
            # The final result reclassifies all calls and individual attempts.
            for row in completed:
                row["overlaps_loop_stall"] = diag.overlaps(row["arrival_offset_s"], row["end_offset_s"])
                row["dispatch_overlaps_loop_stall"] = diag.overlaps(row["arrival_offset_s"], row["actual_start_offset_s"])
                for attempt in row["attempts"]:
                    attempt["overlaps_loop_stall"] = diag.overlaps(attempt["start_offset_s"], attempt["end_offset_s"])
        summary = summarize_live(results, pool.stats())
        summary.update(target="pool", max_concurrent_calls=peak, calls=sorted(completed, key=lambda r: r["id"]))
        summary["timing_boundaries"] = timing_summary(completed)
        summary["primary_latency_boundary"] = "scheduled_arrival_to_result"
        checkpoint["results"] = {"pool": summary}
        save("complete")
        write_json(folder / "results.json", checkpoint, secrets)
        report = ["# Live evaluation result", "", f"Profile: {manifest['profile']}", "",
                  f"Success: {sum(r['outcome'] == 'ok' for r in completed)}/{plan.calls}", "",
                  "Timing boundaries (seconds):", "", "```json",
                  json.dumps(summary["timing_boundaries"], indent=2), "```", "",
                  "Successful-call percentiles; timings below are cumulative caller time, not elapsed wall time.", "",
                  "| Calls | Success | p50 | p95 | Attempts | Scheduler seconds | Provider seconds |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for w in checkpoint["windows"]:
            fmt = lambda value: "-" if value is None else f"{value:.3f}"
            report.append(f"| {w['first_call']}-{w['last_call']} | {w['successes']}/{w['completed']} | {fmt(w['p50'])} | {fmt(w['p95'])} | {w['provider_attempts']} | {w['scheduler_s']:.3f} | {w['provider_s']:.3f} |")
        (folder / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        logging.info("\n%s", _fmt(summary))
        logging.info("Primary scheduled-arrival latency (successful callers): %s",
                     summary["timing_boundaries"]["scheduled_arrival_success"])
        logging.info("Completed; artifacts: %s", folder)
        return 0 if summary["success_rate"] == 1 else 1
    except BaseException:
        save("interrupted_or_failed")
        raise
    finally:
        if diag is not None:
            await diag.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="soak-1000-60")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--label", default="fresh")
    parser.add_argument("--output-root", type=Path, default=ROOT / "evals/results/runs")
    parser.add_argument("--diagnostics", action="store_true", help="opt in to loop/debug/HTTP tracing")
    parser.add_argument("--diagnostic-stack-dumps", action="store_true",
                        help="also take ownership of process-global C traceback scheduling")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--yes", action="store_true", help="make live calls; without this flag only offline preparation runs")
    mode.add_argument("--prepare", action="store_true", help="offline validation only; no client construction or API calls")
    args = parser.parse_args(argv)
    if args.diagnostic_stack_dumps and not args.diagnostics:
        parser.error("--diagnostic-stack-dumps requires --diagnostics")
    try:
        manifest, secrets = prepare(args.profile, args.env_file)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Profile {args.profile}; calls={manifest['plan']['calls']}; timeout={manifest['plan']['request_timeout']}s; "
          f"rate={manifest['plan']['rate_per_min']}/min; pattern={manifest['plan']['pattern']}", flush=True)
    print(f"Local limits: {manifest['configured_rpm']} RPM aggregate; {manifest['configured_daily_attempts']} daily attempts. "
          f"Last scheduled arrival: {manifest['last_scheduled_arrival_s'] / 60:.2f} minutes.", flush=True)
    print("Endpoints: " + ", ".join(e["id"] for e in manifest["endpoints"]), flush=True)
    if not args.yes:
        print("OFFLINE PREPARATION COMPLETE. No provider clients constructed; no API requests sent.", flush=True)
        return 0
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", args.label)[:64].strip("-") or "fresh"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.profile}_{label}_{uuid.uuid4().hex[:8]}"
    folder = args.output_root.resolve() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    configure_logging(folder, secrets)
    manifest.update(run_id=run_id, label=label, fresh_quota_confirmed_by_provider=False,
                    automatic_resume=False)
    try:
        return asyncio.run(execute(manifest, folder, secrets, 100 if manifest["plan"]["calls"] >= 1000 else 10,
                                   diagnostics=args.diagnostics, stack_dumps=args.diagnostic_stack_dumps))
    except KeyboardInterrupt:
        logging.warning("Interrupted. Partial results preserved in %s. No automatic resume or replay.", folder)
        return 130
    except Exception as exc:
        logging.error("Run failed (%s); inspect checkpoint.json and console.log in %s", type(exc).__name__, folder)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
