"""Per-endpoint limit accounting (pre-stop) and cooldown behaviour. Deterministic: explicit `now`."""
from datetime import datetime, timezone

import pytest

from pool_manager.endpoint_state import CooldownPolicy
from tests.fakes import make_ep


def test_rpm_pre_stop_and_exact_sliding_release():
    ep = make_ep("g:m:K", rpm_limit=3)
    assert [ep.try_reserve(now) for now in (0.0, 10.0, 20.0)] == [True, True, True]
    assert ep.try_reserve(30.0) is False                       # pre-stopped: no request is sent
    assert ep.available_in(30.0) == pytest.approx(30.0)        # first slot frees at t=60
    assert ep.try_reserve(59.9) is False
    assert ep.try_reserve(60.0) is True                        # exactly when the oldest request ages out
    assert ep.available_in(60.0) == pytest.approx(10.0)        # next slot (t=10) frees at 70


def test_sliding_window_has_no_boundary_burst():
    # A fixed per-minute counter would allow 2*rpm around the minute boundary; sliding must not.
    ep = make_ep("g:m:K", rpm_limit=2)
    assert ep.try_reserve(58.0) and ep.try_reserve(59.0)
    assert ep.try_reserve(61.0) is False                       # 3 requests inside any 60 s span: refused
    assert ep.try_reserve(118.0) is True


def test_no_limit_means_never_pre_stopped():
    ep = make_ep("g:m:K")
    assert all(ep.try_reserve(0.0) for _ in range(500))
    assert ep.available_in(0.0) == 0.0


def test_rolling_daily_limit():
    ep = make_ep("g:m:K", rpd_limit=2)
    assert ep.try_reserve(0.0) and ep.try_reserve(3600.0)
    assert ep.try_reserve(7200.0) is False
    assert ep.available_in(7200.0) == pytest.approx(86400.0 - 7200.0)
    assert ep.try_reserve(86400.0) is True                     # first request aged out


def test_calendar_daily_reset_frees_at_the_reset_instant():
    wall = datetime(2026, 10, 24, 23, 0, tzinfo=timezone.utc)   # one hour before midnight UTC
    ep = make_ep("g:m:K", rpd_limit=2, daily_reset="midnight_utc", wall_clock=lambda: wall)
    assert ep.try_reserve(0.0) and ep.try_reserve(1.0)
    assert ep.try_reserve(2.0) is False
    assert ep.available_in(2.0) == pytest.approx(3600.0 - 2.0)
    assert ep.try_reserve(3599.0) is False
    assert ep.try_reserve(3600.0) is True                      # new day: counter reset
    assert ep.try_reserve(3601.0) is True
    assert ep.try_reserve(3602.0) is False                     # budget of the new day is spent too


def test_token_budget_pre_stop_uses_real_usage():
    ep = make_ep("g:m:K", tpm_limit=1000)
    assert ep.try_reserve(0.0)
    ep.mark_success(1.0, tokens=1200)                          # that one call blew the minute's budget
    assert ep.try_reserve(2.0) is False
    assert ep.available_in(2.0) == pytest.approx(59.0)         # tokens were recorded at t=1 -> free at 61
    assert ep.try_reserve(61.0) is True


def test_token_tracking_is_skipped_when_no_tpm_limit_is_configured():
    ep = make_ep("g:m:K")
    ep.mark_success(0.0, tokens=10**9)
    assert ep.try_reserve(0.1)


def test_all_limits_must_hold_at_once():
    ep = make_ep("g:m:K", rpm_limit=1, rpd_limit=1000)
    assert ep.try_reserve(0.0)
    assert ep.available_in(1.0) == pytest.approx(59.0)         # rpm is the binding constraint


# ----------------------------------------------------------------------- cooldowns
def test_429_with_retry_after_cools_down_for_that_long_plus_pad():
    ep = make_ep("g:m:K")
    assert ep.mark_rate_limited(100.0, retry_after=34.0) == pytest.approx(35.0)
    assert ep.available_in(100.0) == pytest.approx(35.0)
    assert ep.available_in(135.0) == 0.0


def test_429_without_hint_uses_the_default_window():
    assert make_ep("g:m:K").mark_rate_limited(0.0) == 60.0


def test_retry_after_is_clamped():
    cp = CooldownPolicy(rate_limit_min=2.0, rate_limit_max=100.0, retry_after_pad=0.0)
    ep = make_ep("g:m:K", cooldowns=cp)
    assert ep.mark_rate_limited(0.0, retry_after=0.0) == 2.0
    assert ep.mark_rate_limited(0.0, retry_after=10_000.0) == 100.0


def test_a_shorter_cooldown_never_shortens_a_longer_one():
    ep = make_ep("g:m:K")
    ep.mark_rate_limited(0.0, retry_after=100.0)
    ep.mark_rate_limited(1.0, retry_after=1.0)
    assert ep.cooldown_remaining(1.0) == pytest.approx(100.0)


def test_quota_exhaustion_sleeps_until_the_calendar_reset():
    wall = datetime(2026, 10, 24, 22, 0, tzinfo=timezone.utc)
    ep = make_ep("g:m:K", rpd_limit=100, daily_reset="midnight_utc", wall_clock=lambda: wall)
    secs = ep.mark_rate_limited(0.0, retry_after=30.0, quota_exhausted=True)
    assert secs == pytest.approx(2 * 3600 + 1.0)               # 2h to midnight UTC + pad
    assert ep.quota_hits == 1 and ep.rate_limited == 0


def test_quota_exhaustion_without_a_known_reset_probes_hourly():
    ep = make_ep("g:m:K")
    assert ep.mark_rate_limited(0.0, quota_exhausted=True) == 3600.0


def test_server_errors_back_off_exponentially_and_reset_on_success():
    ep = make_ep("g:m:K")
    assert [ep.mark_server_error(0.0) for _ in range(3)] == [5.0, 10.0, 20.0]
    ep.mark_success(100.0)
    assert ep.mark_server_error(100.0) == 5.0                  # streak reset


def test_server_error_backoff_is_capped():
    ep = make_ep("g:m:K")
    last = [ep.mark_server_error(0.0) for _ in range(12)][-1]
    assert last == 120.0


def test_dead_endpoint_is_quarantined_then_retried():
    ep = make_ep("g:m:K")
    assert ep.mark_dead(0.0, detail="API key not valid") == 3600.0
    assert ep.available_in(10.0) == pytest.approx(3590.0)
    assert ep.available_in(3600.0) == 0.0
    assert "API key not valid" in ep.last_error


def test_release_never_goes_negative():
    ep = make_ep("g:m:K")
    ep.release()
    assert ep.inflight == 0
    ep.try_reserve(0.0)
    ep.release(); ep.release()
    assert ep.inflight == 0


def test_snapshot_reports_usage_and_limit_source():
    ep = make_ep("g:m:K", rpm_limit=10, limits_source="model_limits.yaml [google/m]")
    ep.try_reserve(0.0)
    snap = ep.snapshot(1.0)
    assert (snap["calls"], snap["rpm_used"], snap["rpm_limit"]) == (1, 1, 10)
    assert snap["limits_source"].startswith("model_limits.yaml")


# ------------------------------------------------- float-boundary regression (found by the eval)
def test_waking_exactly_at_expiry_never_overfills_the_window_for_awkward_timestamps():
    """
    `ts + span - now` and `ts <= now - span` are not equivalent in floating point. A waiter that
    wakes at the computed free-up time must not be admitted into a still-full window.
    (Found by the load simulation: ~2% of requests at the boundary were sent one too many.)
    """
    import random

    rng = random.Random(7)
    for _ in range(20_000):
        limit = rng.choice([1, 2, 5, 10])
        ep = make_ep("g:m:K", rpm_limit=limit)
        t = rng.uniform(0, 5000)
        for _ in range(limit):
            assert ep.try_reserve(t)
            t += rng.uniform(0, 0.5)
        asked = t
        wait = ep.available_in(asked)                  # blocked: window is full
        assert wait > 0
        wake = asked + wait                            # exactly what the scheduler does
        ep.try_reserve(wake)
        assert ep._minute._total <= limit, (limit, asked, wait, wake)


def test_blocked_endpoint_never_reports_zero_wait():
    ep = make_ep("g:m:K", rpm_limit=1)
    ep.try_reserve(33.56646023526674)
    now = 33.56646023526674 + 60.0 - 7.1e-15           # a hair before the entry really expires
    assert ep.available_in(now) > 0.0
