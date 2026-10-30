# Benchmark notes

Selected results from development and evaluation, updated 30 October 2026. These checks answer different questions; they are not one benchmark or a production SLA. The repository publishes source, regression tests, and this summary. Raw API recordings, machine logs, interview notes, and the full investigation archive are retained locally and excluded from Git.

## Retry starvation

The actual pool and scheduler were tested with four endpoints, mocking only provider responses. In priority order, the first three persistently return HTTP 503, a connection error, and HTTP 504; the fourth succeeds. All 24 endpoint-identity permutations are exercised synchronously and asynchronously.

The fixed implementation passed **48/48 cases**, with exactly four distinct provider attempts per case. The historical pre-fix implementation passed **0/48**, repeatedly selecting the first failed endpoint until its six-attempt budget ran out. This baseline was reproduced before repository curation; its original Git objects are preserved in the local history backup, not included in the public commit sequence.

Run the current regression with:

```powershell
python -m pytest -q tests/test_retry_starvation_permutations.py
```

## FIFO fairness: seeded simulation

A 1,000-caller Poisson workload used seed 1, a mean 30 arrivals/minute, two 15-RPM endpoints, a 70-second admission budget, and a virtual event loop driving the real scheduler.

| Measurement | Before FIFO | With FIFO |
| --- | ---: | ---: |
| Successful admissions | 981 | 1,000 |
| Expired callers | 19 | 0 |
| Served callers overtaken by qualifying later arrivals | 328 | 0 |
| Queue wait p50 | 2.642s | 13.879s |
| Queue wait p95 | 34.834s | 41.804s |
| Queue wait p99 | 58.725s | 46.874s |

FIFO redistributed waiting fairly: p99 fell **20.2%** while p50 and p95 increased. Percentiles cover admitted callers; the 19 earlier expiries are reported separately. Overtaking counts distinct affected served callers, using the original later-arrival threshold of more than one second. These are simulated queue times, not measured Gemini latency.

## Prompt FIFO wake-ups

A controlled burst used 100 simultaneous callers, four 12-RPM endpoints, a 70-second wait budget, and zero simulated provider service time. The polling FIFO admitted **71/100**; the notification implementation admitted **96/100**. The remaining four cannot fit: 48 slots open initially and another 48 after 60 seconds; the third window is outside the wait budget.

Current regression coverage includes notification races, queue advancement, quota reopening, cancellation, sync/async handoffs, and the immediate admission path:

```powershell
python -m pytest -q tests/test_scheduler_wakeup.py
```

### Paired live comparison

Two sequential 1,000-caller Gemini runs used the same Poisson arrival schedule (seed 1, 48/minute), configuration, dependency environment, and lightweight recorder. Four endpoints combined two Lite models with two owner-confirmed separate-project keys. Each endpoint had local caps of 12 RPM, 250,000 TPM, and 490 RPD. Both runs used a 20-second attempt timeout, 70-second admission budget, 100-second caller budget, and **one provider attempt per caller** to isolate admission behavior.

| Measurement | Polling FIFO | Notification FIFO |
| --- | ---: | ---: |
| Admissions | 1,000/1,000 | 1,000/1,000 |
| Overtaken callers | 0 | 0 |
| Provider attempts | 1,000 | 1,000 |
| Successful responses | 992 | 980 |
| Provider errors | 8 | 20 |
| Admission p50 | 8.316s | 8.130s |
| Admission p95 | 26.874s | 26.260s |
| Admission p99 | 30.327s | 29.835s |
| Observed eligible-capacity idle with queued callers | 68.902s | 6.605s |

Observed eligible-capacity idle fell **90.4%**; admission p95 fell **2.3%**. Idle is reconstructed wall time with queued work and locally eligible capacity, including recorder, event-loop, and host delays. It is not provider hardware idle time or a 90.4% queue-latency reduction.

Provider conditions worsened: the baseline had six HTTP 504s, one 503, and one 500; the fixed run had twelve 504s and eight 503s. Neither saw a provider 429. Response completion therefore declined with retries disabled. Sequential live runs cannot isolate provider conditions; the deterministic burst supplies stronger causal evidence for the wake-up fix. Both recordings passed completeness checks with no dropped events or writer errors. Calculations were performed after recording ended; timestamping and background writing still have overhead.

## Separate spare-capacity soak

An earlier 1,000-caller Gemini run at 30 arrivals/minute completed **1,000/1,000**, using exactly **1,000 provider attempts** with no errors or retries. Scheduled-arrival-to-result latency was **p50 0.931s, p95 2.980s, p99 6.787s**.

This run predates the notification wake-up fix and used different timeout/retry settings: 60-second attempts, a 70-second admission budget, up to six attempts, and no overall caller deadline. It demonstrates a successful spare-capacity run, not code-caused provider speedup. Quotas and provider latency vary across runs.

## Verification and scope

The latest complete offline suite passed **360 tests**. Live evidence covers Gemini; OpenAI-compatible support is implemented and tested offline. Quota counters are local and restart with the process. Separate keys only add quota capacity when their upstream projects and quota scopes are independent.

To reproduce current behavior, use the tests and evaluation commands in the [evaluation guide](../evals/README.md). Historical before/after numbers above summarize the preserved local experiments; they are not regenerated automatically by a clean checkout's test suite.
