"""Turn results.json into REPORT.md: tables with 95% bootstrap CIs over seeds, paired deltas, and caveats."""

from __future__ import annotations

from typing import Optional

from evals.metrics import bootstrap_ci, paired_diff_ci

LABELS = {
    "single_key": "single key + SDK retries",
    "round_robin": "round-robin keys + SDK retries",
    "v3": "pool_manager3 (baseline)",
    "v3_legacy_errors": "pool_manager3, raw-error shape",
    "v3_loose": "pool_manager3, generous limit",
    "merged_reactive": "merged, no limits file",
    "merged": "**merged + limits file**",
}


def _col(runs: dict, scn: str, impl: str, metric: str) -> list[Optional[float]]:
    return [r.get(metric) for r in runs.get(scn, {}).get(impl, [])]


def _fmt(mean, lo, hi, scale=1.0, digits=1, ci=True) -> str:
    if mean is None:
        return "–"
    if not ci or lo is None:
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def _cell(runs, scn, impl, metric, scale=1.0, digits=1, ci=True) -> str:
    return _fmt(*bootstrap_ci(_col(runs, scn, impl, metric)), scale=scale, digits=digits, ci=ci)


def _other_fail(runs, scn, impl) -> list[Optional[float]]:
    s, r = _col(runs, scn, impl, "success_rate"), _col(runs, scn, impl, "ratelimit_failure_rate")
    return [None if a is None or b is None else max(0.0, 1.0 - a - b) for a, b in zip(s, r)]


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _scenario_table(runs: dict, scn: str, impls: list[str]) -> str:
    rows = []
    for impl in impls:
        if impl not in runs.get(scn, {}):
            continue
        rows.append([
            LABELS.get(impl, impl),
            _cell(runs, scn, impl, "success_rate", 100),
            _cell(runs, scn, impl, "ratelimit_failure_rate", 100, ci=False),
            _fmt(*bootstrap_ci([None if v is None else v for v in _other_fail(runs, scn, impl)]), scale=100, ci=False),
            _cell(runs, scn, impl, "provider_429_per_100_calls", 1, 1, ci=False),
            _cell(runs, scn, impl, "lat_p50", 1, 1, ci=False),
            _cell(runs, scn, impl, "lat_p95", 1, 1, ci=False),
            _cell(runs, scn, impl, "lat_p99", 1, 1, ci=False),
            _cell(runs, scn, impl, "fail_latency_mean", 1, 1, ci=False),
            _cell(runs, scn, impl, "efficiency", 100, ci=False),
        ])
    return _table(
        ["Implementation", "Success % [95% CI]", "Rate-limit failures %", "Other failures %", "Provider 429s /100 calls",
         "p50 s", "p95 s", "p99 s", "Mean s lost per failed call", "Capacity efficiency %"], rows)


def _delta_table(runs: dict, scns: list[str], a: str, b: str, metric: str, scale=100.0, digits=1, unit="pp") -> str:
    rows = []
    for scn in scns:
        if a in runs.get(scn, {}) and b in runs.get(scn, {}):
            rows.append([f"`{scn}`", _fmt(*paired_diff_ci(_col(runs, scn, a, metric), _col(runs, scn, b, metric)), scale=scale, digits=digits)])
    return _table(["Scenario", f"Δ {metric} ({unit}) [95% CI]"], rows) if rows else "_(not run)_"


ALTERNATIVES = ("round_robin", "v3", "v3_loose")      # what you would otherwise be running
MATERIAL_PP = 0.5                                      # differences smaller than this are not worth claiming


def verdict(lo: Optional[float], hi: Optional[float]) -> str:
    """Mechanical verdict on a paired difference (percentage points): the interval has to clear +-MATERIAL_PP."""
    if lo is None:
        return "n/a"
    if lo > MATERIAL_PP:
        return "better"
    if hi < -MATERIAL_PP:
        return "WORSE"
    return "no material difference"


def best_alternative(runs: dict, scn: str, impls: list[str]) -> Optional[str]:
    """The alternative with the highest mean success in this scenario. Picking the winner of the others is conservative for `merged`."""
    cands = [i for i in ALTERNATIVES if i in impls and i in runs.get(scn, {})]
    means = {i: bootstrap_ci(_col(runs, scn, i, "success_rate"))[0] for i in cands}
    means = {i: m for i, m in means.items() if m is not None}
    return max(means, key=means.get) if means else None


def summary_section(runs: dict, core: list[str], impls: list[str]) -> list[str]:
    rows, not_better = [], []
    for scn in core:
        alt = best_alternative(runs, scn, impls)
        if alt is None or "merged" not in runs.get(scn, {}):
            continue
        d_mean, d_lo, d_hi = paired_diff_ci(_col(runs, scn, "merged", "success_rate"), _col(runs, scn, alt, "success_rate"))
        v = verdict(None if d_lo is None else d_lo * 100, None if d_hi is None else d_hi * 100)
        rows.append([f"`{scn}`", _cell(runs, scn, "merged", "success_rate", 100, 1, ci=False),
                     f"{LABELS.get(alt, alt)} {_cell(runs, scn, alt, 'success_rate', 100, 1, ci=False)}",
                     _fmt(d_mean, d_lo, d_hi, scale=100), f"**{v}**" if v != "better" else v,
                     f"{_cell(runs, scn, 'merged', 'lat_p95', 1, 1, ci=False)} vs {_cell(runs, scn, alt, 'lat_p95', 1, 1, ci=False)}"])
        if v != "better":
            not_better.append((scn, v))
    out = ["## Summary (generated from the data below; nothing here is hand-edited)\n",
           f"For every scenario, `merged` is compared with the BEST of the alternatives you could be running instead (round-robin keys, "
           f"pool_manager3 with a strict limit, pool_manager3 with a generous limit), chosen by mean success rate. Verdict rule: the paired 95% "
           f"interval must clear +-{MATERIAL_PP} percentage points; otherwise it is reported as no material difference.\n",
           _table(["Scenario", "merged success %", "Best alternative", "Δ success, pp [95% CI]", "Verdict", "p95 s, merged vs alt"], rows), ""]
    if not_better:
        out.append("**Where `merged` is not better:** " + "; ".join(f"`{n}` ({v})" for n, v in not_better) + ". "
                   "`yaml_pessimistic` is the one that matters in practice: a limits file that understates the real limits makes the pool refuse calls "
                   "the provider would have served, which no amount of failover fixes. Overstated limits are harmless (see `yaml_optimistic`).\n")
    out.append("`bursty_60` is better only because the default wait budget is 60 s (the burst overflow waits for the next minute). With a 10-30 s budget "
               "the pool fails those calls exactly like the alternatives; see section 3. Its zero-width interval is a property of the scenario "
               "(every burst has the same size), not evidence of precision.\n")
    return out


def render_report(meta: dict, runs: dict) -> str:
    impls = list(meta["implementations"])
    scns = list(meta["scenarios"])
    core = [s for s in scns if not s.startswith("load_")]
    sweep = sorted([s for s in scns if s.startswith("load_")])
    n = len(meta["seeds"])
    L: list[str] = []
    w = L.append

    w("# pool_manager evaluation report (SIMULATED load test)\n")
    w(f"**Read this first.** Every number below comes from a *simulation*: {n} seeded runs per cell against a fake provider that "
      "enforces rate limits (see `evals/provider.py`). It measures how this code behaves under a stated workload and stated "
      "provider model. It is **not** a measurement of live traffic and must not be quoted as one. "
      "To measure real traffic with your own keys, use `python -m evals.live_eval`.\n")
    topo = meta["topology"]
    w(f"* Topology: models {topo['models']} x {topo['keys']} keys, per-key limits {topo['true_rpm_per_key']} requests/min "
      f"(illustrative), each key in its own project unless stated. Runs: 10 virtual minutes, open-loop Poisson arrivals "
      f"(or synchronized bursts), identical arrivals for every implementation per seed (paired).")
    w("* **Success** = the caller got a response. **Rate-limit failures** = the call failed because of rate limiting by any route: "
      "`AllEndpointsExhausted`, `AllAttemptsFailed` with 429 causes, or a raw 429 reaching the caller. "
      "**Other failures** = outages / dead keys / server errors reaching the caller.")
    w("* **Capacity efficiency** = success relative to the best ANY implementation could achieve (exact bound from the provider's "
      "quota, minus outage/revocation time). It is the fair number when load exceeds capacity.")
    w("* CIs are 95% percentile bootstraps over seeds. Latency columns are over successful calls (p50/p95/p99, virtual seconds, "
      "including time spent waiting and inside SDK back-off).")
    w(f"* Environment: Python {meta['python']}, langchain-core {meta['langchain_core']}. Wall time {meta['wall_seconds']} s.\n")

    L.extend(summary_section(runs, core, impls))

    # ---------------------------------------------------------------- headline
    w("## 1. Headline: calls that complete without a rate-limit failure\n")
    head_cols = [(i, h) for i, h in (("single_key", "single key"), ("round_robin", "round-robin"), ("v3", "pool_manager3, strict limit"),
                                     ("v3_loose", "pool_manager3, generous limit"), ("merged_reactive", "merged (no limits file)"),
                                     ("merged", "**merged**")) if i in impls]
    head_rows = []
    for scn in core:
        if scn in runs:
            head_rows.append([f"`{scn}`"] + [_cell(runs, scn, i, "success_rate", 100, 1, ci=False) if i in runs[scn] else "–" for i, _ in head_cols])
    w(_table(["Scenario"] + [h for _, h in head_cols], head_rows))
    w("\n_Success rate, %. Mean over seeds; CIs are in the per-scenario tables. Scenarios with load above capacity "
      "(`overload_150`) or shared quota (`shared_project`) have a ceiling below 100% for everyone._\n")

    w("### Paired improvement of `merged` (same seed = same arrivals)\n")
    for other, name in (("v3", "pool_manager3 (strict limit)"), ("v3_loose", "pool_manager3 (generous limit)"),
                        ("round_robin", "round-robin keys"), ("merged_reactive", "merged without the limits file")):
        if other not in impls:
            continue
        w(f"**Success rate: merged − {name}**\n")
        w(_delta_table(runs, core, "merged", other, "success_rate") + "\n")
    w("**Rate-limit failure rate: merged − pool_manager3** (negative = merged fails less)\n")
    w(_delta_table(runs, core, "merged", "v3", "ratelimit_failure_rate") + "\n")
    w("**p95 latency of successful calls: merged − pool_manager3** (seconds, negative = merged is faster)\n")
    w(_delta_table(runs, core, "merged", "v3", "lat_p95", scale=1.0, digits=1, unit="s") + "\n")
    w("**Provider 429s per 100 calls: merged − merged without limits file** (what the limits file buys)\n")
    w(_delta_table(runs, core, "merged", "merged_reactive", "provider_429_per_100_calls", scale=1.0, digits=1, unit="per 100") + "\n")

    # ---------------------------------------------------------------- sweep
    if sweep:
        w("## 2. Load sweep: success and efficiency vs offered load\n")
        header = ["Implementation"] + [f"{int(s.split('_')[1])}%" for s in sweep]
        w("**Success %**\n")
        w(_table(header, [[LABELS.get(i, i)] + [_cell(runs, s, i, "success_rate", 100, 1, ci=False) for s in sweep] for i in impls]) + "\n")
        w("**Capacity efficiency %** (success relative to the best achievable)\n")
        w(_table(header, [[LABELS.get(i, i)] + [_cell(runs, s, i, "efficiency", 100, 1, ci=False) for s in sweep] for i in impls]) + "\n")
        w("**p95 latency, successful calls (s)**\n")
        w(_table(header, [[LABELS.get(i, i)] + [_cell(runs, s, i, "lat_p95", 1, 1, ci=False) for s in sweep] for i in impls]) + "\n")

    # ---------------------------------------------------------------- wait budget
    wait_impls = list(meta.get("wait_implementations", {}))
    if wait_impls:
        w("## 3. Waiting vs. success: the `max_wait_seconds` budget\n")
        w("A call that finds every endpoint momentarily full can wait for a slot or fail at once. The pool waits only if the earliest "
          "slot is within `max_wait_seconds`; otherwise it fails immediately. A bigger budget turns failures into (slow) successes; "
          "a smaller one keeps latency bounded. pool_manager3 is shown with its own default (120 s) for reference, so compare it with the 120 s row, not the 60 s one.\n")
        order = ["merged_wait10", "merged_wait30", "merged", "merged_wait120"]
        for scn in meta.get("wait_scenarios", []):
            rows = []
            for impl in ["v3", "v3_loose"] + [o for o in order if o in wait_impls or o == "merged"]:
                if impl not in runs.get(scn, {}):
                    continue
                label = {"merged": "merged, 60 s budget (default)", "v3": "pool_manager3, strict limit (120 s default)",
                                                                                "v3_loose": "pool_manager3, generous limit (120 s default)"}.get(impl, meta["wait_implementations"].get(impl, impl))
                rows.append([label, _cell(runs, scn, impl, "success_rate", 100), _cell(runs, scn, impl, "ratelimit_failure_rate", 100, ci=False),
                             _cell(runs, scn, impl, "lat_p50", 1, 1, ci=False), _cell(runs, scn, impl, "lat_p95", 1, 1, ci=False),
                             _cell(runs, scn, impl, "lat_max", 1, 1, ci=False), _cell(runs, scn, impl, "fail_latency_mean", 1, 1, ci=False),
                             _cell(runs, scn, impl, "slow_call_rate", 100, 1, ci=False)])
            w(f"**`{scn}`**\n")
            w(_table(["Implementation", "Success % [95% CI]", "Rate-limit failures %", "p50 s", "p95 s", "max s", "Mean s spent on a failed call",
                      "Calls slower than 30 s, %"], rows) + "\n")

    # ---------------------------------------------------------------- per scenario
    w("## 4. Per-scenario detail\n")
    for scn in core:
        if scn not in runs:
            continue
        w(f"### `{scn}`\n{meta['scenarios'][scn]}\n")
        w(_scenario_table(runs, scn, impls) + "\n")

    # ---------------------------------------------------------------- keys
    w("## 5. Are all API keys used?\n")
    rows = []
    for scn in [s for s in ("steady_50", "steady_90", "bursty_60", "chaos_70") if s in runs]:
        for impl in impls:
            if impl in runs[scn] and impl != "single_key":
                rows.append([f"`{scn}`", LABELS.get(impl, impl), _cell(runs, scn, impl, "keys_used", 1, 1, ci=False),
                             _cell(runs, scn, impl, "key_jain", 1, 2, ci=False)])
    w(_table(["Scenario", "Implementation", f"Keys receiving traffic (of {topo['keys']})", "Evenness across keys (Jain, 1.0 = even)"], rows) + "\n")
    w("_`chaos_70` revokes key 3 mid-run, so fewer than 3 healthy keys is expected there._\n")

    # ---------------------------------------------------------------- limits
    w("## 6. What this evaluation does NOT show\n")
    for line in (
        "Live-provider behaviour. The provider is a model: real quotas may count rejected requests, vary by time of day, "
        "or enforce limits differently. `steady_90_fixedwin` varies the minute definition, `yaml_*` varies how wrong the limits file is, "
        "`shared_project` shows quota that is not additive; none of these replaces a live run.",
        "The SDK retry model is taken from the installed google-genai source (6 attempts, 1/2/4/8/16 s + jitter, server hint ignored). "
        "If your clients are configured differently the baselines' numbers change, not the merged pool's (it disables SDK retries).",
        "pool_manager3 accepts ONE rpm number for the whole provider, so it is shown both ways: a strict limit (the lowest model's number: "
        "nothing is over-driven, the faster model is under-used) and a generous one (the highest: the slower model is over-driven and "
        "relies on 429 handling). The merged pool takes a number per model. "
        "`pool_manager3, raw-error shape` isolates the error-shape problem by giving it the exception type its classifier was written for.",
        "Offered load above capacity cannot be served by anyone. There, compare efficiency, latency and wasted waiting, not raw success.",
        "Results depend on the topology numbers above; the load sweep shows how they move with load, not with a different number of keys or models.",
    ):
        w(f"* {line}")
    return "\n".join(L) + "\n"
