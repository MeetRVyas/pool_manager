# Evaluating the pool

Everything here answers one question: **does the merged pool actually get more LLM calls through, without long waits, than what you
would otherwise be running?** The honest answer is "yes in some regimes, no in others", and this directory shows which.

> **Every number in `results/REPORT.md` and the simulation sections below is from a SIMULATION.** It tests this code against a model of a rate-limited
> provider. It is not live traffic. `live_eval.py` measures live traffic with your own keys;
> live results are saved separately from the simulation report.

The consolidated [final metrics](../docs/final_metrics.md) separate live traffic, simulations and offline audits. The [final 1,900-call fixed-spacing test at 48/minute](results/LIVE_2026-10-28_PACED_1900_48RPM.md) completed with 1,807 responses, 93 local exhaustion failures, zero provider 429s and successful p95 64.868s. The exact 2,000/60 variant was superseded; the interrupted 1,400/36 variant has no full measurement.

## What was measured

A call **succeeds** if the caller gets a response. It **fails because of rate limits** if it ends in `AllEndpointsExhausted`,
in `AllAttemptsFailed` whose cause was a 429, or in a raw 429 reaching the caller. (That last route matters: a client that
retries internally and then raises a 429 is "all endpoints exhausted" by another name.) Failures from outages, revoked keys
and 5xx are counted separately as "other".

Per run (10 virtual minutes of offered load): success %, rate-limit failures %, p50/p95/p99/max latency of successful calls,
time spent on failed calls, calls slower than 30 s, provider 429s per 100 calls, whether every key received traffic (and how
evenly), and **capacity efficiency** (success relative to the exact most any implementation could get the provider to accept,
which is what makes overload scenarios comparable).

Across 30 seeds: 95% bootstrap intervals, and paired differences (every implementation sees the *same* arrivals for a seed).

## Setup

* **Topology (illustrative, not real Gemini limits):** 2 models x 3 API keys, each key in its own project; model `flash` allows
  10 requests/min per key, `lite` 15. Nominal capacity 75 requests/min. Load is stated as a fraction of that.
* **Provider model** (`provider.py`): enforces per-key limits (sliding window; fixed window and shared-project variants),
  daily caps, outages, revoked keys, 5xx; raises the same wrapped exception types `langchain-google-genai` raises.
  The SDK's own retry loop (6 attempts, 1/2/4/8/16 s + jitter, ignoring the server's retry hint) is modelled for every client
  that does not disable it.
* **Virtual time** (`vtime.py`): a real asyncio event loop with a fake clock, so ten virtual minutes take milliseconds,
  concurrency is real, and results are deterministic.
* **Implementations:** `single_key` and `round_robin` (SDK retries on), `v3` and `v3_loose` (the previous pool, verbatim, in
  `baselines/`; it accepts ONE rpm for the whole provider, so it is run with the lowest model's number and with the highest),
  `v3_legacy_errors` (diagnostic), `merged_reactive` (this pool without a limits file), `merged` (this pool with one).
  Pools are built through their real `get_llm`; only the chat-client class is swapped for the fake.
* **Scenarios:** steady 50% / 90% load, 90% with fixed-minute provider windows, bursty fan-out, overload (150%), faults
  (outage + revoked key + 5xx), limits file 1.5x too high, limits file 0.6x too low, keys sharing one project, daily quotas;
  plus a load sweep (30-200%) and a wait-budget experiment (`max_wait_seconds` 10/30/60/120 on five scenarios).

## Results

Generated from the data (best-alternative comparison; see the report for the rule):

For every scenario, `merged` is compared with the BEST of the alternatives you could be running instead (round-robin keys, pool_manager3 with a strict limit, pool_manager3 with a generous limit), chosen by mean success rate. Verdict rule: the paired 95% interval must clear +-0.5 percentage points; otherwise it is reported as no material difference.

| Scenario | merged success % | Best alternative | Δ success, pp [95% CI] | Verdict | p95 s, merged vs alt |
|---|---:|---:|---:|---:|---:|
| `steady_50` | 100.0 | round-robin keys + SDK retries 100.0 | 0.0 [0.0, 0.0] | **no material difference** | 3.4 vs 3.4 |
| `steady_90` | 100.0 | pool_manager3, generous limit 98.2 | 1.8 [1.5, 2.2] | better | 7.4 vs 19.7 |
| `steady_90_fixedwin` | 100.0 | pool_manager3, generous limit 99.0 | 1.0 [0.8, 1.2] | better | 6.5 vs 13.3 |
| `bursty_60` | 100.0 | pool_manager3 (baseline) 93.3 | 6.7 [6.7, 6.7] | better | 61.1 vs 62.2 |
| `overload_150` | 71.8 | pool_manager3, generous limit 71.3 | 0.5 [0.3, 0.8] | **no material difference** | 54.9 vs 85.2 |
| `chaos_70` | 97.2 | pool_manager3, generous limit 88.4 | 8.8 [8.2, 9.4] | better | 35.3 vs 10.6 |
| `yaml_optimistic` | 100.0 | pool_manager3 (baseline) 99.3 | 0.6 [0.5, 0.8] | better | 4.1 vs 14.2 |
| `yaml_pessimistic` | 78.9 | round-robin keys + SDK retries 97.6 | -18.7 [-19.4, -18.0] | **WORSE** | 53.3 vs 10.5 |
| `shared_project` | 99.5 | pool_manager3 (baseline) 99.3 | 0.2 [0.0, 0.5] | **no material difference** | 4.2 vs 16.4 |
| `daily_quota` | 52.5 | round-robin keys + SDK retries 52.5 | 0.0 [0.0, 0.0] | **no material difference** | 9.0 vs 3.4 |

**Where `merged` is not better:** `steady_50` (no material difference); `overload_150` (no material difference); `yaml_pessimistic` (WORSE); `shared_project` (no material difference); `daily_quota` (no material difference). `yaml_pessimistic` is the one that matters in practice: a limits file that understates the real limits makes the pool refuse calls the provider would have served, which no amount of failover fixes. Overstated limits are harmless (see `yaml_optimistic`).

`bursty_60` is better only because the default wait budget is 60 s (the burst overflow waits for the next minute). With a 10-30 s budget the pool fails those calls exactly like the alternatives; see section 3. Its zero-width interval is a property of the scenario (every burst has the same size), not evidence of precision.

Full tables (every implementation, every scenario, with intervals): [`results/REPORT.md`](results/REPORT.md);
per-seed raw metrics: `results/results.json`.

Highlights, all simulated, 30 seeds, 95% interval in brackets:

| At 90% of capacity (`steady_90`) | Success % | p95 latency, s | Provider 429s per 100 calls |
|---|---:|---:|---:|
| **merged** | **100.0 [100.0, 100.0]** | 7.4 | 0.0 |
| merged without limits file | 98.2 [97.7, 98.6] | 5.0 | 51.6 |
| round-robin keys, SDK retries | 94.5 [94.0, 95.0] | 16.4 | 89.6 |
| pool_manager3, strict limit | 95.4 [94.6, 96.1] | 55.1 | 95.5 |
| pool_manager3, generous limit | 98.2 [97.8, 98.5] | 19.7 | 76.7 |

| Under faults (`chaos_70`) | Success % | p95 latency, s |
|---|---:|---:|
| **merged** | **97.2 [96.6, 97.8]** | 35.3 |
| pool_manager3, generous / strict limit | 88.4 / 83.1 | 10.6 / 10.9 |
| round-robin keys | 79.0 | 7.2 |

The merged pool buys that extra success by waiting (p95 35 s here). With `max_wait_seconds=10` it still succeeds 93.5% of the time and the
slowest call in a run averages 12.8 s. The waiting/success trade-off is section 3 of the report. At 90% load a 10 s budget gives 99.1%
success, p95 5.2 s and a slowest call averaging 11.1 s per run; pool_manager3 (120 s default budget) had 28.7% of calls take over 30 s and a
slowest call averaging 106.6 s per run. (Per-run maxima are averaged over seeds, so individual runs can exceed them.)

**Every key is used** in every scenario that has healthy keys (3 of 3 receive traffic; evenness 1.00 in steady state, 0.90 when a key is revoked mid-run).

## Where the pool does not help (read before quoting anything)

* **At 70% of capacity or less nothing is gained**: plain round-robin over the same keys also completes ~100% (99.7% at 70%).
  The success-rate headline only separates the implementations near capacity or under faults.
* **Overload cannot be fixed by any client.** At 150% load the pool's edge over round-robin is +3.2 pp (and +0.5 pp over
  pool_manager3 with a generous limit). At 200% pool_manager3 with a generous limit is 1.5 pp ahead of the merged pool at its default
  60 s budget (56.1% vs 54.6%); given the same 120 s budget the merged pool is ahead (59.1% [58.8, 59.5] vs 56.1% [55.7, 56.5]), so that gap is
  the wait budget, not the pool logic. The pool's overload benefit is zero wasted requests (0 provider 429s versus 71-310 per 100 calls) and
  efficiency, not throughput.
* **Keys from one Google project share one quota** (`shared_project`): extra keys add nothing; whatever gain appears comes from the second model.
* **Daily quotas** (`daily_quota`): same success as round-robin (both use every available request); the pool saves ~285 wasted
  429s per 100 calls, at the price of a higher p95 (9.0 s vs 3.4 s) because it waits for per-minute windows instead of failing.
* **A limits file that understates the real limits hurts** (`yaml_pessimistic`, file says 0.6x the truth): the pool refuses calls
  the provider would have served and completes 78.9% versus 97.6% for round-robin. **Overstating is harmless** (`yaml_optimistic`:
  100%, the reactive path catches the 429s). When unsure of a limit, round up or leave that model out of the file.
* **`bursty_60`'s gain is a wait-budget effect:** with the default 60 s budget the overflow of a synchronized burst waits for the next
  minute (p95 61 s) and succeeds; with 10-30 s it fails like the others.
* The provider is a model. Real quotas may count rejected requests, drift by time of day, or differ per tier.

## Is it worth putting on a resume?

Yes, **as a simulated evaluation with the numbers and the caveats**, and the stronger material is the method and the honest findings
rather than any single percentage:

* The percentage that separates this pool from alternatives is "completes 100% of calls at 90% of capacity vs 94.5-98%", and "97% vs 79-88%
  under faults". Quoting "99% success" with no load context would be meaningless, because at 70% load a round-robin loop gets the same.
* The harness is the part an interviewer can probe: deterministic virtual time, paired seeds, bootstrap intervals, an exact
  capacity bound, a baseline run verbatim, and a bug it found in the sliding-window limiter (a float-rounding case that let a request
  into a full window; fixed, with regression tests).

Wording that is supported by the data (keep "simulated" and the parameters):

> Built a multi-key, multi-model LLM client pool (per-model rate-limit catalogue, bounded-wait failover, cooldowns) and evaluated it in a
> seeded virtual-time simulation (30 seeds, rate-limit-enforcing provider model, 2 models x 3 keys): it completed 100% of calls at 90% of
> provider capacity versus 94.5% for round-robin key rotation and 95-98% for the previous pool version (pool_manager3), and 97% versus 79-88% under injected
> outage / revoked-key / 5xx faults, with no provider 429s when the limits file matches the provider.

> Wrote a deterministic simulation harness (4,020 stored runs, paired seeds, bootstrap CIs, capacity-bound efficiency metric) and used it to
> find and fix a floating-point bug in the sliding-window rate limiter; also quantified where the pool does *not* help (shared-project
> keys, overload, understated limits).

**Do not claim:** any production or live-traffic improvement; "reduced rate-limit errors by X%" as a general figure; higher throughput
(the pool cannot create quota); anything about real Gemini limits (the numbers are illustrative); "100% success" without "at 90% of
capacity, simulated"; or a comparison against "no pool" using only the single-key row (it is a weak straw man: one key, one model).

To upgrade the claim to a measured one, run `live_eval.py` (below) a few times at different hours and quote those numbers instead.

## Reproduce

```bash
pip install -r requirements.txt
python -m evals.run_eval --quick                # 5 seeds, core scenarios, ~1 min on 2 cores
python -m evals.run_eval                        # 30 seeds, everything: ~12 min on 2 cores -> evals/results/
python -m evals.run_eval --resume --impls merged   # only run what is missing; resumed == one-shot (tested)
pytest                                          # includes the harness's own validity tests
```

Results are deterministic per (scenario, implementation, seed). `tests/test_eval_harness.py` pins the harness itself: virtual time,
provider enforcement, the SDK retry model, v3 behaving under virtual time exactly as its source says, paired workloads,
that no run can exceed the capacity bound, and the metric arithmetic.

## Live evaluation (your keys)

For the next fresh-key experiments, use the [PowerShell runbook](../docs/live_eval_runbook.md).
`scripts/Run-LiveEvaluation.ps1` wraps `evals.run_live_profile` with isolated configs,
explicit `.env` overrides, unique result directories, per-attempt timing, source hashes,
progress and partial checkpoints. It defaults to offline preparation and requires `-Run`
for live traffic. Prepared profiles cover the 1,000-call paced 60s soak, 100-call paced
30s/60s comparisons, a 100-call burst, and an optional 12-call Flash smoke test.
The [first 1,000-call workload](results/LIVE_2026-10-27_SOAK.md) completed 1,000/1,000
at 30/minute, with p50 1.033s and p95 7.918s. Do not repeat completed profiles or run
all profiles automatically against the same daily budget. The [fresh-key small suite](results/LIVE_2026-10-27_SMALL_SUITE.md)
also completed: 100/100 with the 30s timeout, 100/100 with 3.5 Lite at 24/minute,
and 12/12 with Flash preferred. All 212 callers succeeded; the first workload
recovered 30 transient provider errors. These are completed measurements, not pending runs.

Two further profiles are available: `lite35-1000-30rpm` uses only 3.5 Lite across
two keys at the owner's full 15 RPM/500 RPD per-project caps; `lites-2000-60rpm`
is a prepared, unexecuted overload workload at 60/minute with the original four
endpoints and reduced caps. The latter exceeds both the 48-RPM combined admission
rate and 1,960 daily-attempt budget. The [single-model run](results/LIVE_2026-10-27_SINGLE_MODEL.md)
completed 976/1,000, with p95 50.075s, queueing and late daily-quota responses.
See the runbook before choosing another experiment.

The owner later selected `lites-1900-60rpm`, a slightly reduced revision of the
2,000-call workload. It [completed with 1,180/1,900 responses](results/LIVE_2026-10-27_OVERLOAD_1900.md),
zero provider 429s, 720 endpoint-exhaustion outcomes and p95 72.489s. The 60/minute
offered rate still exceeds 48 configured RPM. This observes overload and cooldown
behavior; it does not prove reduced caps are optimal. The exact 2,000-call variant
was not separately executed.

The corrected `lites-1400-36rpm` profile uses fixed spacing at 36/minute against
48 configured RPM and leaves daily retry room. Its [initial attempt was interrupted](results/LIVE_2026-10-27_ALIGNED_1400_INTERRUPTED.md)
after 3.5 Lite/key2 returned a daily-quota 429 on its second local attempt. It is
prepared but has no completed full-run measurement; available project quota must
be confirmed before restarting.

The [Flash-Lite load comparison](results/LIVE_2026-10-26_LITES_LOAD.md) records earlier
pool-only runs: 12 calls with the previous config, then 100 paced and 100 burst calls
using 3.5/3.1 Flash-Lite, including per-call timing and measured caller concurrency.

The [2026-10-26 live run](results/LIVE_2026-10-26.md) completed 12/12 pool calls
versus 8/12 single-key calls, with substantial latency and a small-sample caveat.
Its raw results are separate from the simulation data above.
That historical run was issued as a burst because of a subsequently fixed arrival-clock
bug, despite its Poisson plan label. See the [recovery verification and rerun](results/LIVE_2026-10-26_LIMITED.md)
for the timing correction, endpoint reopening evidence, and updated live measurements.

The CLI loads `.env` without overriding existing environment variables; use `--env-file`
to select another file. The supplied config uses `GEMINI_API_KEY` and `GEMINI_API_KEY2`
with Gemini 3.5 and 3.1 Flash-Lite. Both have local caps of 12 RPM, 250000 TPM, 490 RPD;
3.1 uses these caps at the owner's request, pending confirmation of actual project quotas.
See `configs/model_limits.yaml` for sources and scope. Earlier recorded runs used
different model configurations, as described in their reports.

```bash
python -m evals.live_eval --config configs/llm_config.yaml --dry-run                      # checks keys/limits, sends nothing
python -m evals.live_eval --config configs/llm_config.yaml --mode pool --calls 100 --rate-per-min 30 --yes --out live.json
```

`both` runs "first model, first key, SDK retries" and then the pool on the same arrival schedule. Calls are one-line prompts, `--max-calls`
caps spend (default 200), nothing is sent without `--yes`. One run is one sample of a quota state you do not control; repeat it at different
times, and note that keys from one project share a quota.

Live workloads use overlapping asyncio calls: `--rate-per-min` is offered arrivals per
minute, not per second; `--pattern burst` offers every call within one second. Reports
include per-call latencies and `max_concurrent_calls` (caller requests, including those
waiting for endpoints). A burst above capacity may exhaust the wait budget even without
any provider 429. Use pool-only first when preserving scarce daily quota matters.

## Files

The [latency attribution audit](results/LATENCY_ANALYSIS_2026-10-27.md) calculates
direct-call p50/p95/p99, separates no-retry caller records into acquisition/provider
components, and measures local SDK/LangChain/pool work using a guarded mock HTTP
transport. `python -m evals.analyze_recorded_latency` and
`python -m evals.audit_wrapper_overhead` are offline and spend no API quota.

The [direct REST report](results/DIRECT_API_2026-10-27.md) records 12 calls with
no pool, LangChain, SDK retries or failover: 11/12 returned successfully, and all
six concurrent calls succeeded. `python -m evals.direct_api_eval --yes` explicitly
spends another 12 requests and saves a new directory with transport/server timing.
One direct Flash request took 23.838s with 23.819s reported server duration; raw API
latency is variable too. This small sample is not a paired pool-speed benchmark.

`vtime.py` virtual clock - `provider.py` rate-limited fake provider - `scenarios.py` workloads and capacity bound - `impls.py` the
implementations - `sim.py` one run - `metrics.py` metrics and statistics - `run_eval.py` runner - `report.py` report writer -
`live_eval.py` real-API harness - `baselines/pm_v3/` the previous pool, verbatim - `results/` the committed 30-seed run.

## Queue-focused recording

`python -m evals.admission_probe` prepares an offline plan. Live calls require an explicit `--yes`; larger runs also require `--allow-large`. The recorder writes timestamped events through a bounded background-writer queue, and calculates metrics only after recording closes.

Use `python -m evals.admission_probe --analyze RUN_FOLDER` for a single recording. For a matched comparison:

```powershell
python -m evals.analyze_admission_comparison --baseline BASELINE_FOLDER --fixed FIXED_FOLDER --out comparison.json
```

The analyzer checks workload, limits, dependencies, recorder identity, and recording completeness before accepting a comparison. See the [completed paired 1,000-call investigation](../docs/post_checkpoint_wakeup_results.md) for results and interpretation.
