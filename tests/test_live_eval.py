"""
live_eval.py talks to real providers, which this test suite never does. What CAN be verified is
everything around the network call: the schedule, the guards against accidental spend, the
plumbing that builds the pool / the single-key baseline, and the arithmetic of the report. The
"provider" here is the evaluation harness's rate-limit-enforcing fake, driven on virtual time.
"""
import functools
import os
import random

import pytest
import yaml

from pool_manager import llm as pm
from evals import live_eval, vtime
from evals.metrics import CallResult
from evals.provider import CallCtx, FakeGemini, World
from evals.scenarios import MODELS, N_KEYS, Scenario

SCN = Scenario("live-test", "", rho=1.0)


@pytest.fixture(autouse=True)
def no_real_clients(monkeypatch):
    """Safety net: unless a test installs the fake provider, constructing a real client is a test failure. If a spend guard
    in live_eval ever regressed, the test must fail here instead of sending requests to a real API."""
    def explode(*a, **k):
        raise AssertionError("a real provider client was constructed in a test")
    monkeypatch.setattr(pm, "ChatGoogleGenerativeAI", explode)
    monkeypatch.setattr(pm, "ChatOpenAI", explode)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    names = [f"LIVE_KEY_{k}" for k in range(1, N_KEYS + 1)]
    for k, n in enumerate(names, start=1):
        monkeypatch.setenv(n, f"sk-eval-{k}")
    path = tmp_path / "llm_config.yaml"
    path.write_text(yaml.safe_dump({"google": {"models": list(MODELS), "keys": names}}))
    limits = tmp_path / "model_limits.yaml"
    limits.write_text(yaml.safe_dump({"google": {m: {"rpm": r} for m, r in {"flash": 10, "lite": 15}.items()}}))
    return path, limits


def patch_provider(monkeypatch, world):
    fake = functools.partial(FakeGemini, world=world)
    monkeypatch.setattr(pm, "ChatGoogleGenerativeAI", fake)


def new_world(seed=1):
    return World(SCN.specs(), rng=random.Random(seed))


# ------------------------------------------------------------------------------ schedule + stats
def test_arrivals_are_deterministic_sorted_and_the_right_size():
    plan = live_eval.Plan(calls=40, rate_per_min=60, seed=7)
    a, b = live_eval.arrivals(plan), live_eval.arrivals(plan)
    assert a == b and len(a) == 40 and a == sorted(a)
    assert live_eval.arrivals(live_eval.Plan(calls=40, rate_per_min=60, seed=8)) != a


def test_poisson_arrivals_have_the_requested_mean_rate():
    plan = live_eval.Plan(calls=4000, rate_per_min=120, seed=3)
    span = live_eval.arrivals(plan)[-1]
    assert 4000 / span * 60 == pytest.approx(120, rel=0.06)


def test_burst_arrivals_land_within_one_second():
    t = live_eval.arrivals(live_eval.Plan(calls=50, pattern="burst"))
    assert len(t) == 50 and max(t) <= 1.0


def test_unknown_pattern_is_rejected():
    with pytest.raises(ValueError):
        live_eval.arrivals(live_eval.Plan(pattern="zigzag"))


@pytest.mark.parametrize("k,n,lo,hi", [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634), (50, 100, 0.4038, 0.5962)])
def test_wilson_matches_known_values(k, n, lo, hi):
    got = live_eval.wilson(k, n)
    assert got[0] == pytest.approx(lo, abs=2e-3) and got[1] == pytest.approx(hi, abs=2e-3)


def test_wilson_of_nothing_is_the_whole_range():
    assert live_eval.wilson(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------------- end to end
def run(mode, cfg_paths, plan, world, monkeypatch):
    path, limits = cfg_paths
    patch_provider(monkeypatch, world)
    plan = live_eval.Plan(**{**plan.__dict__, "limits_path": str(limits)})
    return vtime.run_virtual(lambda: live_eval.run_mode(mode, path, plan, clock=vtime.now))


def test_pool_uses_every_key_and_succeeds_below_capacity(cfg, monkeypatch):
    plan = live_eval.Plan(calls=120, rate_per_min=40, seed=2)             # ~half of the 75/min capacity
    m = run("pool", cfg, plan, new_world(), monkeypatch)
    assert m["n_calls"] == 120 and m["success_rate"] == 1.0
    assert m["keys_used"] == N_KEYS == m["keys_total"]
    assert m["pool_rate_limit_responses"] == 0                           # the limits file pre-stopped everything
    assert m["outcomes"] == {"ok": 120}
    assert 0.0 <= m["success_ci95_wilson"][0] <= 1.0 and m["success_ci95_wilson"][1] == 1.0


def test_report_only_contains_metrics_that_are_valid_without_provider_counters(cfg, monkeypatch):
    m = run("pool", cfg, live_eval.Plan(calls=20, rate_per_min=30), new_world(), monkeypatch)
    for invalid in ("efficiency", "ceiling_success", "provider_429_per_100_calls", "wasted_request_pct", "accepted_requests"):
        assert invalid not in m
    assert set(live_eval.LIVE_KEYS) <= set(m)


def touched(world):
    """Endpoints that received at least one HTTP request, according to the PROVIDER's own counters."""
    return {ep for ep, c in world.counters.items() if sum(c.values()) > 0}


def test_single_key_baseline_uses_one_key_with_sdk_retries_and_does_worse_under_load(cfg, monkeypatch):
    plan = live_eval.Plan(calls=120, rate_per_min=90, seed=4)            # far above one key's 10/min
    w_single, w_pool = new_world(), new_world()
    single = run("single-key", cfg, plan, w_single, monkeypatch)
    pooled = run("pool", cfg, plan, w_pool, monkeypatch)
    assert single["success_rate"] < 0.5 < pooled["success_rate"]
    assert "keys_used" not in single and "SDK" in single["note"]
    assert single["target"] == "single-key (flash, LIVE_KEY_1)"
    assert touched(w_single) == {"google:flash:KEY_1"}                   # one model, one key, nothing else
    assert len(touched(w_pool)) == len(MODELS) * N_KEYS                  # the pool reached every (model, key)
    # SDK retries are on: far more HTTP requests than calls, because each failed call was retried inside the client
    assert sum(sum(c.values()) for c in w_single.counters.values()) > 2 * plan.calls


def test_pool_reactive_mode_ignores_the_limits_file(cfg, monkeypatch):
    path, limits = cfg
    patch_provider(monkeypatch, new_world())
    target = live_eval.build_target("pool-reactive", path, live_eval.Plan(limits_path=str(limits)), clock=vtime.now)
    assert target.name == "pool-reactive"
    plan = live_eval.Plan(calls=60, rate_per_min=60, seed=5, limits_path=str(limits))
    reactive = run("pool-reactive", cfg, plan, new_world(), monkeypatch)
    limited = run("pool", cfg, plan, new_world(), monkeypatch)
    assert reactive["pool_rate_limit_responses"] > 0 == limited["pool_rate_limit_responses"]


def test_modes_run_back_to_back_with_a_pause_between_them(cfg, monkeypatch):
    path, limits = cfg
    world = new_world()
    patch_provider(monkeypatch, world)
    plan = live_eval.Plan(calls=30, rate_per_min=30, limits_path=str(limits))

    async def go():
        t0 = vtime.now()
        out = await live_eval.run_modes(["single-key", "pool"], path, plan, pause_s=1000.0, clock=vtime.now)
        return out, vtime.now() - t0

    out, elapsed = vtime.run_virtual(go)
    assert list(out) == ["single-key", "pool"]
    assert elapsed >= 1000.0          # the runs themselves take ~1 virtual minute each, so only a real pause gets here


# ------------------------------------------------------------------------------------ CLI guards
def requests_sent(world):
    return sum(sum(c.values()) for c in world.counters.values())


def test_dry_run_checks_the_config_and_sends_nothing(cfg, capsys, monkeypatch):
    path, limits = cfg
    world = new_world()
    patch_provider(monkeypatch, world)
    rc = live_eval.main(["--config", str(path), "--limits", str(limits), "--dry-run", "--calls", "10", "--yes"])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run" in out and "flash" in out and "LIVE_KEY_3" in out
    assert requests_sent(world) == 0                                       # even with --yes, a dry run sends nothing


def test_without_yes_nothing_is_sent_and_the_exit_code_says_so(cfg, capsys, monkeypatch):
    path, limits = cfg
    world = new_world()
    patch_provider(monkeypatch, world)
    rc = live_eval.main(["--config", str(path), "--limits", str(limits), "--calls", "10"])
    assert rc == 2 and "--yes" in capsys.readouterr().err
    assert requests_sent(world) == 0


def test_with_yes_the_cli_really_runs_and_writes_results(cfg, tmp_path, capsys, monkeypatch):
    """Whole CLI path on the real event loop: the fake provider sleeps for real, so keep it tiny and fast."""
    path, limits = cfg
    world = new_world()
    world.latency_median, world.latency_sigma, world.reject_latency = 0.001, 0.0, 0.001
    patch_provider(monkeypatch, world)
    out_file = tmp_path / "r.json"
    rc = live_eval.main(["--config", str(path), "--limits", str(limits), "--calls", "6", "--rate-per-min", "6000",
                         "--yes", "--out", str(out_file)])
    text = capsys.readouterr().out
    assert rc == 0 and "[pool]" in text and "success" in text
    import json
    data = json.loads(out_file.read_text())
    assert data["results"]["pool"]["n_calls"] == 6 and data["plan"]["calls"] == 6
    assert requests_sent(world) >= 6


def test_calls_above_the_hard_cap_are_refused(cfg, capsys):
    path, _ = cfg
    with pytest.raises(SystemExit) as e:
        live_eval.main(["--config", str(path), "--calls", "500", "--yes"])
    assert e.value.code == 2


def test_missing_key_env_var_is_reported_not_crashed(cfg, capsys, monkeypatch):
    path, _ = cfg
    monkeypatch.delenv("LIVE_KEY_2")
    rc = live_eval.main(["--config", str(path), "--dry-run"])
    assert rc == 2 and "LIVE_KEY_2" in capsys.readouterr().err


def test_env_file_loads_missing_keys_without_overriding_existing_values(cfg, tmp_path, monkeypatch, capsys):
    path, limits = cfg
    world = new_world()
    patch_provider(monkeypatch, world)
    existing = os.environ["LIVE_KEY_1"]
    key2 = os.environ["LIVE_KEY_2"]
    monkeypatch.delenv("LIVE_KEY_2")
    env_file = tmp_path / "keys.env"
    env_file.write_text(f"LIVE_KEY_1=must-not-override\nLIVE_KEY_2={key2}\n")
    rc = live_eval.main(["--config", str(path), "--limits", str(limits),
                         "--env-file", str(env_file), "--dry-run"])
    assert rc == 0
    assert os.environ["LIVE_KEY_1"] == existing
    assert os.environ["LIVE_KEY_2"] == key2
    assert requests_sent(world) == 0
    capsys.readouterr()


def test_formatting_handles_a_run_where_every_call_failed():
    results = [CallResult(0, 0.0, 1.0, "exhausted", CallCtx())]
    m = live_eval.summarize_live(results, [])
    m["target"] = "t"
    text = live_eval._fmt(m)
    assert m["success_rate"] == 0.0 and m["lat_p50"] is None and m["ratelimit_failure_rate"] == 1.0
    assert "success              0.0%" in text and "p50 -" in text          # undefined percentiles print as "-", no crash
