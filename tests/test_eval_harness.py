"""
Validity tests for the EVALUATION HARNESS itself. A benchmark is only worth quoting if the
benchmark is right, so these pin down: virtual time, provider enforcement, SDK-retry model,
that the v3 baseline behaves under virtual time exactly as its source says, determinism,
paired workloads, and metric maths.
"""
import asyncio
import random
import time

import pytest

from pool_manager.errors import Kind, classify
from evals import vtime
from evals.impls import IMPLS, WAIT_IMPLS, _load_v3, build, v3_limits
from evals.metrics import bootstrap_ci, jain, paired_diff_ci, percentile
from evals.provider import (
    CALL, Attempt, CallCtx, FakeGemini, Spec, World, build_exception,
)
from evals.scenarios import Scenario, all_scenarios
from evals.sim import outcome_of, simulate


def world_of(*specs, enforcement="sliding", seed=1):
    return World(list(specs), rng=random.Random(seed), enforcement=enforcement)


def spec(model="m", key="K1", project="P", rpm=3, rpd=None, **kw):
    return Spec(model=model, key_name=key, api_key=f"sk-{key}", project=project, true_rpm=rpm, true_rpd=rpd, **kw)


# ============================================================================ virtual time
def test_virtual_loop_runs_ten_virtual_minutes_instantly_and_orders_timers():
    order = []

    async def main():
        async def napper(name, secs):
            await asyncio.sleep(secs)
            order.append((name, vtime.now()))
        await asyncio.gather(napper("c", 300), napper("a", 10), napper("b", 10.5))
        return vtime.now()

    started = time.time()
    end = vtime.run_virtual(main)
    assert time.time() - started < 1.0
    assert end == pytest.approx(300.0)
    assert [n for n, _ in order] == ["a", "b", "c"]


def test_virtual_loop_detects_a_deadlock_instead_of_hanging():
    async def main():
        await asyncio.get_running_loop().create_future()     # nobody will ever resolve it

    with pytest.raises(RuntimeError, match="deadlock"):
        vtime.run_virtual(main)


# ========================================================================== provider model
async def _hit(world, sp, n, gap=0.0):
    out = []
    for _ in range(n):
        out.append(await world.attempt(sp))
        if gap:
            await asyncio.sleep(gap)
    return out


def test_sliding_provider_accepts_exactly_rpm_per_minute_and_hints_the_wait():
    sp = spec(rpm=3)

    async def main():
        w = world_of(sp)
        first = [await w.attempt(sp) for _ in range(5)]
        await asyncio.sleep(61)
        again = await w.attempt(sp)
        return first, again

    first, again = vtime.run_virtual(main)
    assert [a.kind for a in first] == ["ok", "ok", "ok", "rate", "rate"]
    assert 0 < first[3].retry_s <= 60
    assert again.kind == "ok"


def test_sliding_provider_never_exceeds_rpm_in_any_60s_window_under_random_hammering():
    sp = spec(rpm=5)

    async def main():
        w = world_of(sp)
        rng = random.Random(3)
        accepted = []
        for _ in range(600):
            admitted_at = vtime.now()                          # the provider decides at request ARRIVAL
            if (await w.attempt(sp)).kind == "ok":
                accepted.append(admitted_at)
            await asyncio.sleep(rng.uniform(0, 4))
        return accepted

    acc = vtime.run_virtual(main)
    assert acc
    for i, t in enumerate(acc):
        assert sum(1 for x in acc[i:] if x < t + 60.0) <= 5


def test_fixed_window_provider_really_allows_a_boundary_burst_that_sliding_refuses():
    sp = spec(rpm=3)
    fixed, sliding = world_of(sp, enforcement="fixed"), world_of(sp, enforcement="sliding")
    fixed._bucket(sp).phase = 0.0                                  # windows are [0,60), [60,120) ...
    f = [fixed._admit(sp, x) for x in (58.0, 58.5, 59.0, 61.0, 61.5, 62.0)]
    s = [sliding._admit(sp, x) for x in (58.0, 58.5, 59.0, 61.0, 61.5, 62.0)]
    assert f == [None] * 6                                         # 6 requests in 4 s: allowed across a clock-minute boundary
    assert s[:3] == [None] * 3 and all(v is not None for v in s[3:])
    assert fixed._admit(sp, 62.5) is not None                      # ...but the new window is now full


def test_keys_of_one_project_share_quota_but_separate_projects_do_not():
    a, b = spec(key="K1", project="P", rpm=2), spec(key="K2", project="P", rpm=2)
    c = spec(key="K3", project="Q", rpm=2)

    async def main():
        w = world_of(a, b, c)
        shared = [(await w.attempt(a)).kind, (await w.attempt(b)).kind, (await w.attempt(a)).kind]
        other = (await w.attempt(c)).kind
        return shared, other

    shared, other = vtime.run_virtual(main)
    assert shared == ["ok", "ok", "rate"]                          # b's request consumed a's quota
    assert other == "ok"


def test_daily_quota_returns_a_perday_429_with_a_misleadingly_short_hint_and_rejections_cost_nothing():
    sp = spec(rpm=100, rpd=2)

    async def main():
        w = world_of(sp)
        return [await w.attempt(sp) for _ in range(5)], w

    atts, w = vtime.run_virtual(main)
    assert [a.kind for a in atts] == ["ok", "ok", "quota_day", "quota_day", "quota_day"]
    assert atts[2].retry_s == 30.0
    assert w.counters[sp.id]["accepted"] == 2
    exc = build_exception(atts[2], "m")
    assert classify(exc).kind is Kind.QUOTA                        # the merged classifier reads it as daily


def test_outage_revocation_and_random_errors_are_injected():
    out = spec(outages=((10.0, 20.0),))
    rev = spec(key="K2", revoked_from=5.0)

    async def main():
        w = world_of(out, rev)
        r = [(await w.attempt(out)).kind]
        await asyncio.sleep(12)
        r.append((await w.attempt(out)).kind)
        r.append((await w.attempt(rev)).kind)
        return r

    assert vtime.run_virtual(main) == ["ok", "server", "auth"]


# ====================================================================== exception realism
def test_wrapped_429_has_no_code_but_a_cause_and_legacy_is_raw():
    att = Attempt(429, "rate", 12.0, "", "RESOURCE_EXHAUSTED")
    wrapped, legacy = build_exception(att, "m"), build_exception(att, "m", legacy=True)
    assert getattr(wrapped, "code", None) is None and wrapped.__cause__.code == 429
    assert legacy.code == 429
    for exc in (wrapped, legacy):                                  # the merged classifier handles both
        info = classify(exc)
        assert info.kind is Kind.RATE_LIMIT and info.retry_after == 12.0


def test_the_premise_of_the_v3_comparison_v3_misses_the_wrapped_429_but_sees_the_raw_one():
    v3 = _load_v3()
    att = Attempt(429, "rate", 12.0, "", "RESOURCE_EXHAUSTED")
    assert v3._is_rate_limit_error(build_exception(att, "m")) is False
    assert v3._is_rate_limit_error(build_exception(att, "m", legacy=True)) is True


# ==================================================================== SDK retry model
def test_sdk_default_retry_makes_six_attempts_and_waits_about_33_seconds():
    sp = spec(rpm=1)

    async def main():
        w = world_of(sp)
        await w.attempt(sp)                                        # use the only slot
        client = FakeGemini(world=w, model="m", api_key=sp.api_key, max_retries=6)
        ctx = CallCtx(); CALL.set(ctx)
        t0 = vtime.now()
        with pytest.raises(Exception):
            await client.ainvoke("x")
        return ctx.requests, vtime.now() - t0

    attempts, elapsed = vtime.run_virtual(main)
    assert attempts == 6
    assert 31.0 <= elapsed <= 36.5          # 1+2+4+8+16 s of back-off + up to 5 s jitter + 6 rejections


def test_max_retries_zero_means_a_single_attempt_and_auth_errors_are_never_retried():
    sp, rev = spec(rpm=1), spec(key="K2", revoked_from=0.0)

    async def main():
        w = world_of(sp, rev)
        await w.attempt(sp)
        no_retry = FakeGemini(world=w, model="m", api_key=sp.api_key, max_retries=0)
        started = vtime.now()
        ctx = CallCtx(); CALL.set(ctx)
        with pytest.raises(Exception):
            await no_retry.ainvoke("x")
        n_after_429 = ctx.requests
        ctx2 = CallCtx(); CALL.set(ctx2)
        with pytest.raises(Exception):
            await FakeGemini(world=w, model="m", api_key=rev.api_key, max_retries=6).ainvoke("x")
        return n_after_429, ctx2.requests, vtime.now() - started

    n429, nauth, now = vtime.run_virtual(main)
    assert n429 == 1 and nauth == 1
    assert now < 1.0                      # no back-off sleeping at all (6 retries would take >= 31 s)


# ============================================================ baseline fidelity (v3 verbatim)
def test_v3_under_virtual_time_waits_for_its_60s_window_exactly_as_its_source_says():
    _load_v3()
    from evals.baselines.pm_v3.endpoint_state import EndpointState
    from evals.baselines.pm_v3.scheduler import LLMScheduler

    async def main():
        ep = EndpointState(id="a", provider="g", model="m", rpm_limit=2, rpd_limit=100, client=None)
        s = LLMScheduler([ep], max_wait_seconds=120)
        times = []
        for _ in range(3):
            async with s.allocate_endpoint():
                times.append(vtime.now())
        return times

    t = vtime.run_virtual(main)
    assert t[0] == 0.0 and t[1] == 0.0
    assert t[2] == pytest.approx(60.0, abs=1.0)                    # window refreshes at 60 s (polled every 1 s)


def test_v3_waits_the_full_budget_then_raises_when_everything_is_cooling_down():
    _load_v3()
    from evals.baselines.pm_v3.endpoint_state import EndpointState
    from evals.baselines.pm_v3.scheduler import AllEndpointsExhausted, LLMScheduler

    async def main():
        ep = EndpointState(id="a", provider="g", model="m", rpm_limit=5, rpd_limit=100, client=None)
        ep.mark_rate_limited()                                     # 60 s cooldown (hard-coded in v3)
        s = LLMScheduler([ep], max_wait_seconds=30)
        t0 = vtime.now()
        with pytest.raises(AllEndpointsExhausted):
            async with s.allocate_endpoint():
                pass
        return vtime.now() - t0

    assert vtime.run_virtual(main) == pytest.approx(30.0, abs=1.0)  # sleeps out the whole budget before failing


def test_builders_hand_the_fake_client_the_real_sdk_retry_settings():
    scn = all_scenarios()["steady_50"]

    async def main():
        w = World(scn.specs(), rng=random.Random(1))
        for name in ("v3", "merged"):
            build(name, scn, w)
        import pool_manager.llm as merged_llm
        return True

    assert vtime.run_virtual(main)          # build() itself asserts max_retries==0 for the merged pool


def test_v3_leaves_sdk_retries_on_and_merged_turns_them_off():
    scn = all_scenarios()["steady_50"]
    seen = {}

    async def main():
        w = World(scn.specs(), rng=random.Random(1))
        for name in ("v3", "merged"):
            call = build(name, scn, w)
            llm = call.__closure__[0].cell_contents
            seen[name] = {ep.client.sdk_attempts for ep in llm.scheduler.endpoints}

    vtime.run_virtual(main)
    assert seen == {"v3": {6}, "merged": {1}}


def test_merged_endpoints_get_the_limits_from_the_yaml_file():
    scn = all_scenarios()["steady_50"]
    seen = {}

    async def main():
        w = World(scn.specs(), rng=random.Random(1))
        llm = build("merged", scn, w).__closure__[0].cell_contents
        seen.update({ep.model: ep.rpm_limit for ep in llm.scheduler.endpoints})

    vtime.run_virtual(main)
    assert seen == {"flash": 10, "lite": 15}


# ============================================================ determinism / paired workloads
def test_simulation_is_deterministic_and_seed_sensitive():
    scn = all_scenarios()["chaos_70"]
    a, b, c = simulate(scn, "merged", 5), simulate(scn, "merged", 5), simulate(scn, "merged", 6)
    assert a == b
    assert a != c


def test_every_implementation_faces_identical_offered_load_for_a_seed():
    scn = all_scenarios()["steady_50"]
    assert len({simulate(scn, impl, 3)["n_calls"] for impl in IMPLS}) == 1


def test_outcomes_partition_the_calls():
    scn = all_scenarios()["chaos_70"]
    for impl in IMPLS:
        m = simulate(scn, impl, 2)
        parts = (m["success_rate"] + m["exhausted_rate"] + m["attempts_failed_429_rate"] + m["attempts_failed_other_rate"]
                 + m["leaked_429_rate"] + m["leaked_5xx_rate"] + m["other_fail_rate"])
        assert parts == pytest.approx(1.0), impl
        assert m["ratelimit_failure_rate"] == pytest.approx(m["exhausted_rate"] + m["attempts_failed_429_rate"] + m["leaked_429_rate"])


@pytest.mark.parametrize("name", ["overload_150", "chaos_70", "daily_quota", "shared_project", "steady_90_fixedwin"])
def test_no_implementation_can_get_more_accepted_than_the_scenarios_capacity_bound(name):
    """Validates the ceiling formula itself: if any run exceeded it, the bound (and so every efficiency figure) would be wrong."""
    scn = all_scenarios()[name]
    slack = 1.0 if scn.enforcement == "sliding" else 1.0 + len(scn.specs())      # fixed windows can straddle one extra boundary
    for impl in IMPLS:
        m = simulate(scn, impl, 4)
        horizon = m["virtual_seconds"]
        assert m["accepted_requests"] <= scn.max_accepted(horizon) + slack, (impl, m["accepted_requests"], scn.max_accepted(horizon))


@pytest.mark.parametrize("impl", sorted(WAIT_IMPLS))
def test_ceiling_stays_a_valid_bound_when_calls_wait_long_after_the_last_arrival(impl):
    """With a 120 s wait budget calls finish ~2 minutes after the last arrival, and the provider keeps accepting requests
    until then. The ceiling must be computed over the run's real end time: a horizon that stops at the last arrival
    would put it BELOW what the provider really accepted (and efficiency would exceed 100%, hidden by the cap)."""
    scn = all_scenarios()["overload_150"]
    m = simulate(scn, impl, 1)
    assert m["success_rate"] <= m["ceiling_success"] + 1e-9, (impl, m["success_rate"], m["ceiling_success"])
    assert m["accepted_requests"] <= scn.max_accepted(m["virtual_seconds"]) + 1.0
    if impl == "merged_wait120":
        assert m["virtual_seconds"] > scn.duration_s + 60.0                           # the tail really is long
        assert m["success_rate"] > scn.max_accepted(scn.duration_s) / m["n_calls"]    # ... and a tail-blind bound would be wrong


def test_efficiency_is_capped_by_the_ceiling_and_equals_success_when_load_is_servable():
    light = simulate(all_scenarios()["steady_50"], "merged", 1)
    assert light["ceiling_success"] == 1.0 and light["efficiency"] == pytest.approx(light["success_rate"])
    over = simulate(all_scenarios()["overload_150"], "merged", 1)
    assert over["ceiling_success"] < 1.0 and over["efficiency"] <= 1.0


def test_outcome_classification_of_what_the_caller_sees():
    from pool_manager.scheduler import AllAttemptsFailed, AllEndpointsExhausted
    assert outcome_of(AllEndpointsExhausted("x")) == "exhausted"
    rate_cause, outage_cause = AllAttemptsFailed("x"), AllAttemptsFailed("y")
    rate_cause.__cause__ = build_exception(Attempt(429, "rate", 3.0, "", "RESOURCE_EXHAUSTED"), "m")
    outage_cause.__cause__ = build_exception(Attempt(503, "server", None, "x", "UNAVAILABLE"), "m")
    assert outcome_of(rate_cause) == "attempts_failed"             # counts as a rate-limit failure
    assert outcome_of(outage_cause) == "attempts_failed_other"     # does not
    assert outcome_of(build_exception(Attempt(429, "rate", 3.0, "", "RESOURCE_EXHAUSTED"), "m")) == "leaked_429"
    assert outcome_of(build_exception(Attempt(503, "server", None, "x", "UNAVAILABLE"), "m")) == "leaked_5xx"
    assert outcome_of(ValueError("bug")) == "other"


# ================================================================================ metric maths
def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20], 95) == pytest.approx(19.5)
    assert percentile([], 50) is None


def test_jain_index_bounds():
    assert jain([5, 5, 5]) == pytest.approx(1.0)
    assert jain([9, 0, 0]) == pytest.approx(1 / 3)
    assert jain([0, 0]) is None


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_data():
    rng = random.Random(0)
    small = [rng.gauss(0.9, 0.05) for _ in range(10)]
    big = [rng.gauss(0.9, 0.05) for _ in range(200)]
    m1, lo1, hi1 = bootstrap_ci(small)
    m2, lo2, hi2 = bootstrap_ci(big)
    assert lo1 <= m1 <= hi1 and lo2 <= m2 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)
    assert bootstrap_ci([None, None]) == (None, None, None)
    assert bootstrap_ci([3.0]) == (3.0, 3.0, 3.0)


def test_paired_difference_uses_matching_seeds_only():
    mean, lo, hi = paired_diff_ci([0.9, 0.8, None, 0.7], [0.5, 0.5, 0.5, None])
    assert mean == pytest.approx(0.35) and lo <= mean <= hi


# ============================================================================ v3 configuration + resumable runs
def test_v3_can_be_configured_strict_or_generous_from_the_same_limits_file():
    scn = all_scenarios()["steady_90"]                                   # flash 10 rpm, lite 15 rpm
    assert v3_limits(scn, loose=False)["rpm_limit"] == 10
    assert v3_limits(scn, loose=True)["rpm_limit"] == 15
    daily = all_scenarios()["daily_quota"]                               # flash 30/day, lite 50/day
    assert (v3_limits(daily, loose=False)["rpd_limit"], v3_limits(daily, loose=True)["rpd_limit"]) == (30, 50)


def test_v3_generous_limit_overdrives_the_slower_model_when_a_burst_arrives():
    """Not a uniform ranking (the generous setting does better at steady 90% load): in a synchronized burst the single
    generous number lets 15 requests/min through to flash keys that only allow 10, so the provider rejects far more."""
    scn = all_scenarios()["bursty_60"]
    strict, loose = simulate(scn, "v3", 1), simulate(scn, "v3_loose", 1)
    assert strict["n_calls"] == loose["n_calls"]
    assert loose["provider_429_per_100_calls"] > 1.5 * strict["provider_429_per_100_calls"]
    assert loose["success_rate"] < strict["success_rate"]


def test_pending_cells_skips_only_complete_cells():
    from evals.run_eval import pending_cells
    done = {"a": {"x": [{}, {}], "y": [{}]}}                             # a/x complete for 2 seeds, a/y only 1 of 2
    wanted = [("a", "x", [1, 2]), ("a", "y", [1, 2]), ("b", "x", [1, 2])]
    assert pending_cells(done, wanted) == [("a", "y", [1, 2]), ("b", "x", [1, 2])]


def test_resumed_run_equals_a_one_shot_run(tmp_path):
    """Results are deterministic per (scenario, implementation, seed), so building results in two steps changes nothing."""
    import json
    from evals.run_eval import main
    one, two = tmp_path / "one", tmp_path / "two"
    common = ["--seeds", "2", "--workers", "1", "--no-wait-sweep"]
    main(common + ["--scenarios", "steady_50,bursty_60", "--impls", "merged,v3_loose", "--out", str(one)])
    main(common + ["--scenarios", "steady_50", "--impls", "merged", "--out", str(two)])
    main(common + ["--scenarios", "steady_50,bursty_60", "--impls", "merged,v3_loose", "--out", str(two), "--resume"])
    a, b = json.loads((one / "results.json").read_text()), json.loads((two / "results.json").read_text())
    assert a["runs"] == b["runs"]
    assert list(b["meta"]["scenarios"]) == ["steady_50", "bursty_60"] and list(b["meta"]["implementations"]) == ["v3_loose", "merged"]
    assert "pool_manager3, generous limit" in (two / "REPORT.md").read_text()


def test_resume_refuses_results_made_with_different_seeds(tmp_path):
    from evals.run_eval import main
    args = ["--workers", "1", "--no-wait-sweep", "--scenarios", "steady_50", "--impls", "merged", "--out", str(tmp_path)]
    main(["--seeds", "2"] + args)
    with pytest.raises(SystemExit):
        main(["--seeds", "3", "--resume"] + args)
