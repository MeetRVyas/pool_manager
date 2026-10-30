# Optional caller deadline

Implemented 29 October 2026. This is separate from endpoint cooldowns, per-attempt SDK timeout and cumulative scheduler-wait limits. The default remains `None` (disabled).

```python
from pool_manager.llm import get_llm, CallerDeadlineExceeded

llm = get_llm("configs/llm_config.yaml", request_timeout=20,
              max_wait_seconds=10, caller_timeout=30)
try:
    response = await llm.ainvoke("Your prompt")
except CallerDeadlineExceeded:
    # Application-specific response to an expired caller budget.
    pass
```

The budget covers pool admission, provider work and retries. Plain, tool-bound and structured-output invocation share the same deadline helpers. LangChain's outer input/callback processing and arbitrary application code are outside the helper's boundary. This is not an operating-system guarantee that blocking code can be preempted at an exact instant.

Async expiry requests cooperative cancellation and releases the acquired endpoint in `finally`. An explicitly cancelled task remains `CancelledError`; caller expiry becomes `CallerDeadlineExceeded`. Neither policy expiry nor cancellation is counted as a provider 5xx or used to penalize endpoint health. New attempts recheck the remaining budget, and acquisition cannot use more than that remainder.

Sync calls with a deadline run in a daemon worker carrying the invocation context. The caller returns on budget expiry, but Python cannot safely kill a blocked SDK call. Its endpoint reservation remains in flight until the worker really exits. No further retries occur after expiry, and a late response is discarded instead of being recorded as a successful call. Request quota already reserved is not refunded; an upstream request may still execute or be billed. Sync calls without a deadline keep the original direct execution path. Deadline-enabled sync traffic can retain blocked workers, so asynchronous invocation is preferable when cancellation and resource control matter.

No universal 8–10s default or automatic queue-depth shedding was introduced. The application chooses its budget; a short deadline can intentionally reject responses that previously succeeded slowly. Successful-call percentiles must be paired with deadline-failure counts.

The evaluation outcome `deadline_exceeded` is reported separately from quota exhaustion and provider 5xx. It is included in aggregate other-failure rates so outcome accounting remains complete. Historical result files retain their original meanings.

Tests in `tests/test_caller_deadline.py` cover both invocation modes across plain/tool/structured paths, failover before expiry, exact release behavior, queued-acquisition cancellation, explicit task cancellation and invalid budgets. The diagnostic profile `diagnostic-12-12rpm` offers 12 fixed-spaced callers at 12/minute using the existing Lite pool, with 30s attempts, a 90s caller deadline and at most two attempts per caller. It is a diagnostic sample, not a matched latency benchmark or a recommendation for application defaults.
