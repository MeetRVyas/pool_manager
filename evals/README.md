# Evaluation

Use offline regressions to verify scheduling and failover, simulations to explore load, and explicitly enabled live calls to measure provider behavior. Generated results remain local under `evals/results/`, which is ignored by Git. Selected historical outcomes and limitations are in the [benchmark notes](../docs/benchmarks.md).

## Offline tests

From the repository root after installing `requirements.txt`:

```powershell
python -m pytest -q
python -m pytest -q tests/test_retry_starvation_permutations.py tests/test_scheduler_wakeup.py
```

Tests use provider doubles and do not require real API calls. The latest full suite passed 360 tests.

## Simulated load

```powershell
python -m evals.run_eval --quick
```

The harness uses a virtual-time event loop, quota-enforcing provider model, seeded arrival schedules, and baseline implementations. Its generated latencies are simulated. Use `python -m evals.run_eval --help` for workload and output options. The [baseline description](baselines/README.md) explains the included older implementation.

## Live workloads

Set the environment-variable key names required by your chosen configuration in `.env`. Validate actual project quotas first; two keys in one Gemini project share quota. The checked-in profiles use `GEMINI_API_KEY` and `GEMINI_API_KEY2`, whereas the application example uses `GEMINI_API_KEY1` and `GEMINI_API_KEY2`.

The PowerShell wrapper prepares a plan by default:

```powershell
.\scripts\Run-LiveEvaluation.ps1 -TestProfile soak-1000-60 -RunLabel my-run
```

Only add `-Run` when you intend to send the configured workload to the provider:

```powershell
.\scripts\Run-LiveEvaluation.ps1 -TestProfile soak-1000-60 -RunLabel my-run -Run
```

Inspect the saved plan before enabling a large run. Live profiles save manifests, configuration snapshots, per-call records, checkpoints, results, and summaries in their printed run folder. Keep the machine awake and the process alive. A partial journal is not a completed run. Failures and quota waiting are part of the result and must be reported alongside successful-call percentiles.

## Queue-focused recording

```powershell
python -m evals.admission_probe
python -m evals.admission_probe --help
```

Preparation is offline. Sending requests requires `--yes`; more than 16 callers also requires `--allow-large`. During execution, the recorder timestamps events and enqueues them for a bounded background writer. It does not calculate percentiles on the call path. Drops or writer failures invalidate recording completeness; recording overhead is small, not zero.

Analyze after the recording closes:

```powershell
python -m evals.admission_probe --analyze RUN_FOLDER
python -m evals.analyze_admission_comparison --baseline BASELINE_FOLDER --fixed FIXED_FOLDER --out comparison.json
```

The comparison checks arrival plans, limits, dependencies, recorder identity, expected implementation labels, and event completeness. Use independently fresh quota for each version and report sequential-provider confounding. The local archive contains the historical recordings; they are intentionally not required for a clean checkout or its offline tests.

## Diagnostics

`evals.direct_api_eval`, `evals.audit_wrapper_overhead`, `evals.audit_async_concurrency`, and `evals.loop_diagnostics` support direct-provider comparisons and local latency diagnosis. Inspect each module's CLI or source before use. Live direct-API calls require explicit enablement; detailed loop/debug tracing can itself add overhead. `evals.analyze_recorded_latency` is an archive-specific analysis utility and requires the historical local recordings, which are not distributed.
