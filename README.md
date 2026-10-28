# pool_manager (merged)

A LangChain chat model that spreads calls over many **(provider, model, API key)** endpoints,
stops sending to an endpoint *before* its rate limit, and survives 429s, 5xx errors, timeouts
and dead keys by switching endpoints instead of waiting or crashing.

Base: `pool_manager3`. From `pool_manager2`: the benchmark tracker (latency/tokens/cost).
`pool_manager1` was an older subset of v3 and contributed nothing new.

Start with the [final metrics and sharing guide](docs/final_metrics.md) for measured results and the minimal evidence files.

The [engineering case study](docs/project_case_study.md) documents the issues, fixes,
simulation gains, live experiments, latency diagnosis, and known limits. A
historical PDF may exist locally under `output/pdf/`; it predates the final results and is excluded from version control. Use the Markdown documents for current findings.

## Quick start

Install dependencies with `python -m pip install -r requirements.txt`.
The live-eval CLI loads `.env` automatically. For application code, load it before building the pool:

```python
from dotenv import load_dotenv
load_dotenv()
from pool_manager.llm import get_llm
llm = get_llm("configs/llm_config.yaml")          # picks up configs/model_limits.yaml if present
llm.invoke("hi"); await llm.ainvoke("hi")
llm.with_structured_output(MySchema).ainvoke(...) ; llm.bind_tools([...]).ainvoke(...)
print(llm.usage_report())                          # which model/key got how much traffic
```

## Config

`configs/llm_config.yaml` — **order of `models` is the preference order** (see `llm_config.example.yaml`).

`configs/llm_config.yaml` selects Gemini 3.5 Flash-Lite followed by 3.1 Flash-Lite,
using `GEMINI_API_KEY` and `GEMINI_API_KEY2`. Flash is excluded from the active pool.

`configs/model_limits.yaml` records limits enforced **per endpoint**. Google quotas are
per project: two keys in one project do not add capacity. The catalogue was reviewed on
2026-10-26 against [Google's rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).
Active RPM/TPM/RPD values depend on the project/tier. The project owner supplied the
3.5 quotas on 2026-10-26; the keys belong to separate projects. The owner requested
the same local caps for 3.1 Flash-Lite; its actual project quotas remain unverified:

| Model | Upstream RPM / RPD | Configured RPM / RPD | TPM |
|---|---:|---:|---:|
| 3.5 Flash-Lite | 15 / 500 | 12 / 490 | 250,000 |
| 3.1 Flash-Lite | unverified | 12 / 490 | 250,000 |

The catalogue retains inactive Flash entries for existing callers; 3.5 Flash routes
to 3.6 Flash and those aliases should not be counted as independent capacity.
The active entries are:

```yaml
google:
  gemini-3.5-flash-lite: {rpm: 12, tpm: 250000, rpd: 490, daily_reset: midnight_pt}
  gemini-3.1-flash-lite: {rpm: 12, tpm: 250000, rpd: 490, daily_reset: midnight_pt}
```

* Model with numeric limits -> pre-stopped at its limit; the pool moves to the next model/key *before* a 429.
* Model without numeric limits -> no pre-stop; the first 429 puts that key on cooldown (using the
  server's `Retry-After` when given) and the call is retried elsewhere.
* Provider-level `rpm_limit/rpd_limit/tpm_limit` in `llm_config.yaml` (the old v3 style) still work as defaults;
  a model's catalogue entry overrides them field by field.
* Typos / bad values in the limits file fail at startup. `safety_margin=0.1` pre-stops at 90%.
* Context-window limits are per request and must not be used as TPM quotas.
* The request headroom is already included in these numbers; default `safety_margin=0.0`
  avoids reducing them again. Waiting near capacity can increase p95 latency, while avoiding
  provider retries can reduce it; the result depends on load and provider behavior.

## How a call is handled

Within each call, retries prefer eligible endpoints that have not yet been attempted.
Provider/model priority and key rotation apply within that group, so an expired cooldown
cannot keep bouncing a call between failed primary keys while a healthy fallback is ready.
An attempted endpoint can still be reused when no untried endpoint is eligible, subject
to the attempt cap and wait budget. Each new call starts with normal model preference.

| Situation | What happens |
|---|---|
| endpoint at its configured limit | skipped; next key of the same model, then next model |
| 429 (per-minute) | key cools down for `Retry-After`+1 s (60 s if unspecified); same call retried elsewhere |
| 429 (daily quota / `insufficient_quota`) | key parked until the daily reset (exact for `midnight_pt/utc`, else probed hourly) |
| 5xx, timeout, connection error | key backs off 5 s, 10 s, 20 s ... (max 120 s), retried elsewhere |
| 401/403/404, invalid key | key quarantined for 1 h, retried elsewhere |
| 400 / parse error / bug | raised to the caller untouched (retrying elsewhere can't help) |
| nothing available | waits only if the earliest key frees up **within `max_wait_seconds`**, otherwise raises `AllEndpointsExhausted` immediately (`.retry_after` says how long) |
| too many failures in one call | `AllAttemptsFailed` after `max_attempts` (subclass of `AllEndpointsExhausted`) |

The pool cannot create capacity: if every key is genuinely out of quota it raises (fast, with `retry_after`)
rather than blocking. Catch `AllEndpointsExhausted` where you want to degrade or retry later.

SDK-internal retries are switched off (`max_retries=0`; Gemini's default is 6 attempts with backoff, OpenAI's is 2)
and a per-request `timeout` is set, so the pool sees failures immediately and a hung connection cannot stall a call.

## Selection

`strategy="priority"` (default): best-ranked model with headroom; among its keys, least-recently-used first
(all keys share load). `strategy="balanced"`: spread over every endpoint regardless of model order.

## Verifying that every key is used

* `llm.usage_report()` / `llm.stats()` — per-endpoint calls, 429s, errors, limit usage, cooldowns; lists never-used endpoints.
* Startup log (`logging` INFO, logger `pool_manager`) shows each endpoint and where its limits came from.
* Benchmark report has a "By API Key" section. Endpoint ids contain env-var *names*, never key values.
* Under `priority`, fallback-model keys stay idle until the primary is limited — by design.
* Google applies limits **per project, not per key**: keys from the same project share one quota.

## get_llm options

`max_wait_seconds=60` (was 120), `request_timeout=120`, `max_attempts=6`, `strategy="priority"`,
`safety_margin=0.0`, `limits_path="auto"` (`None` disables, or a path / `LimitsCatalogue`), `cooldowns=CooldownPolicy(...)`.

## Known limits

* Usage counters are in memory. After a restart the pool assumes full daily budgets; the first daily-quota 429 per key corrects it.
* One shared window per endpoint; no per-key tier overrides (put distinct limits in separate provider blocks if needed).
* Daily `midnight_pt` needs tz data (`pip install tzdata` on Windows), otherwise falls back to fixed UTC-8.

## Evaluation

The [final 1,900-call run at 48/minute](evals/results/LIVE_2026-10-28_PACED_1900_48RPM.md) completed **1,807/1,900 (95.1%)**, with p50 17.218s, p95 64.868s, p99 101.317s and zero provider 429s. All 93 failures were local exhaustion outcomes; no daily cap was reached. Equal offered and configured RPM leaves no retry/cooldown capacity. This operating-boundary result complements the spare-capacity soak below.

The maintained [engineering case study](docs/project_case_study.md) documents the findings,
fixes and evidence. Use the [fresh-key PowerShell runbook](docs/live_eval_runbook.md) for
the prepared 1,000-call soak and later 100-call timeout comparisons. The wrapper defaults
to offline preparation; only `-Run` sends live requests. The [first 1,000-call soak](evals/results/LIVE_2026-10-27_SOAK.md)
completed 1,000/1,000 calls at 30/minute, with p50 1.033s and p95 7.918s.
The [subsequent small suite](evals/results/LIVE_2026-10-27_SMALL_SUITE.md) completed
212/212 callers across the fresh 30s timeout rerun, a single-model workload with spare
minute capacity, and a Flash routing smoke test. A [1,900-call revision of the planned overload test](evals/results/LIVE_2026-10-27_OVERLOAD_1900.md)
completed at 60/minute against 48 configured RPM: 1,180 successes, 720 exhaustion outcomes,
zero provider 429s and p95 72.489s. This documents the overload tradeoff, not an optimal safety margin.

`evals/` holds a deterministic **simulated** load test (virtual-time event loop, a provider model that enforces per-key rate limits, 30 paired
seeds, bootstrap intervals) plus `live_eval.py` for measuring real traffic with your own keys. Start with
[`evals/README.md`](evals/README.md); the full tables are in `evals/results/REPORT.md`.

What it showed, and what to do about it:

* Near capacity (90% load) or under faults the pool completes more calls than round-robin keys or pool_manager3 (100% vs 94.5-98% at 90% load;
  97% vs 79-88% with an outage, a revoked key and 5xx). At 70% load or less nothing is gained: round-robin also reaches ~100%.
* **Keep `model_limits.yaml` accurate, and round up when unsure.** Limits that are too high are harmless (the 429 path catches them);
  limits that are too low make the pool refuse calls the provider would have served (78.9% vs 97.6% in the 0.6x scenario).
* **Choose `max_wait_seconds` on purpose.** It is the success/latency dial: at 90% load, 10 s gives 99.1% success with p95 5 s; 60 s gives 100%
  with p95 7 s; under faults 10 s gives 93.5% (p95 8 s) and 60 s gives 97.2% (p95 35 s). Interactive callers want 10-30 s, batch jobs 60-120 s.
* Keys from one Google project share one quota, so the pool cannot add capacity there. No client can serve load above the provider's limits.

## Tests

`pip install langchain-core langchain-openai langchain-google-genai pyyaml pytest` then `pytest` (no network, no pytest-asyncio needed;
this includes the evaluation harness's own validity tests, and a safety net that fails any test that would create a real provider client).
Regenerate the evaluation with `python -m evals.run_eval` (about 12 minutes on 2 cores; `--quick` for a minute).
