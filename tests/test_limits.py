"""The YAML model-limits catalogue, how limits are resolved, and daily-reset arithmetic."""
from datetime import datetime, timezone

import pytest
import yaml

from pool_manager.limits import (
    Limits, LimitsCatalogue, parse_limits_entry, resolve_limits, seconds_until_daily_reset,
)


def cat(text: str) -> LimitsCatalogue:
    return LimitsCatalogue.from_dict(yaml.safe_load(text), origin="test.yaml")


SAMPLE = """
google:
  gemini-2.5-flash: {rpm: 10, tpm: 250000, rpd: 250, daily_reset: midnight_pt}
  gemini-2.5-flash-lite: {rpm: 15}
  "gemini-2.5-*": {rpm: 7}
  "gemini-*": {rpm: 3}
"*":
  "llama-*": {rpm: 30}
openai:
  gpt-4o-mini: {rpm: 500}
"""


# ----------------------------------------------------------------------- lookup
def test_exact_match_beats_globs():
    got = cat(SAMPLE).lookup("google", "gemini-2.5-flash")
    assert (got.rpm, got.tpm, got.rpd, got.daily_reset) == (10, 250000, 250, "midnight_pt")


def test_longest_matching_glob_wins():
    c = cat(SAMPLE)
    assert c.lookup("google", "gemini-2.5-flash-preview-09-2025").rpm == 7   # "gemini-2.5-*" beats "gemini-*"
    assert c.lookup("google", "gemini-1.5-pro").rpm == 3                       # only "gemini-*"


def test_exact_flash_lite_is_not_swallowed_by_flash_entry():
    assert cat(SAMPLE).lookup("google", "gemini-2.5-flash-lite").rpm == 15


def test_gemini_and_google_are_the_same_provider():
    assert cat(SAMPLE).lookup("gemini", "gemini-2.5-flash").rpm == 10


def test_matching_is_case_insensitive():
    assert cat(SAMPLE).lookup("Google", "Gemini-2.5-FLASH").rpm == 10


def test_provider_agnostic_section_is_the_last_resort():
    c = cat(SAMPLE)
    assert c.lookup("groq", "llama-3.3-70b").rpm == 30
    assert c.lookup("google", "llama-3.3-70b").rpm == 30


def test_unknown_model_has_no_entry():
    assert cat(SAMPLE).lookup("openai", "gpt-9") is None
    assert cat(SAMPLE).lookup("mistral", "mixtral") is None


def test_lookup_reports_where_the_limit_came_from():
    assert "test.yaml" in cat(SAMPLE).lookup("google", "gemini-2.5-flash").source


def test_aliases_rpm_limit_style_names_are_accepted():
    assert cat("google: {m: {rpm_limit: 4, rpd_limit: 9, tpm_limit: 100}}").lookup("google", "m").rpd == 9


def test_empty_file_is_an_empty_catalogue():
    assert not LimitsCatalogue.from_dict(None)
    assert LimitsCatalogue.from_dict(None).lookup("google", "x") is None


# -------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "bad",
    [
        {"rmp": 10},                          # typo must not silently disable the limit
        {"rpm": 0}, {"rpm": -5}, {"rpm": "10"}, {"rpm": 1.5}, {"rpm": True},
        {"rpm": 10, "daily_reset": "noon"},
        "not a mapping", 10, [1, 2],
    ],
)
def test_bad_entries_are_rejected_loudly(bad):
    with pytest.raises(ValueError):
        parse_limits_entry(bad, "x")


def test_bad_top_level_shapes_are_rejected():
    with pytest.raises(ValueError):
        LimitsCatalogue.from_dict(["google"])
    with pytest.raises(ValueError):
        LimitsCatalogue.from_dict({"google": ["gemini-2.5-flash"]})


def test_null_field_means_unlimited():
    assert parse_limits_entry({"rpm": 10, "rpd": None}, "x").rpd is None


# ----------------------------------------------------------------- resolution order
def test_catalogue_overrides_provider_defaults_field_by_field():
    c = cat("google: {m: {rpm: 10}}")
    got = resolve_limits("google", "m", c, {"rpm_limit": 99, "rpd_limit": 50})
    assert (got.rpm, got.rpd, got.tpm) == (10, 50, None)    # rpm from YAML, rpd filled from provider default


def test_provider_defaults_used_when_model_has_no_catalogue_entry():
    got = resolve_limits("google", "other", cat("google: {m: {rpm: 10}}"), {"rpm_limit": 5, "rpd_limit": 20})
    assert (got.rpm, got.rpd) == (5, 20) and "provider defaults" in got.source


def test_no_limits_anywhere_means_reactive_only():
    got = resolve_limits("google", "other", cat("google: {m: {rpm: 10}}"), {})
    assert got.is_empty and "reactive" in got.source


def test_no_catalogue_at_all_is_fine():
    assert resolve_limits("google", "m", None, {"rpm_limit": 5}).rpm == 5
    assert resolve_limits("google", "m", None, {}).is_empty


def test_invalid_provider_level_limit_is_rejected():
    with pytest.raises(ValueError):
        resolve_limits("google", "m", None, {"rpm_limit": "ten"})


# ------------------------------------------------------------------------ margin
def test_safety_margin_pre_stops_early_but_never_below_one():
    got = Limits(rpm=10, rpd=250, tpm=1000).with_margin(0.1)
    assert (got.rpm, got.rpd, got.tpm) == (9, 225, 900)
    assert Limits(rpm=1).with_margin(0.5).rpm == 1
    assert Limits(rpm=10).with_margin(0).rpm == 10


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2])
def test_invalid_margin_rejected(bad):
    with pytest.raises(ValueError):
        Limits(rpm=10).with_margin(bad)


# --------------------------------------------------------------------- daily reset
def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_rolling_has_no_reset_instant():
    assert seconds_until_daily_reset("rolling", utc(2026, 10, 24, 12)) is None


def test_midnight_utc():
    assert seconds_until_daily_reset("midnight_utc", utc(2026, 10, 24, 21, 30)) == 2.5 * 3600


def test_midnight_pacific_on_a_normal_day():
    # 21:00 UTC on Oct 24 = 14:00 PDT; next Pacific midnight = 07:00 UTC Oct 25 -> 10 h away
    assert seconds_until_daily_reset("midnight_pt", utc(2026, 10, 24, 21, 0)) == 10 * 3600


def test_midnight_pacific_on_the_dst_change_day_is_24_5_hours_not_23_5():
    # DST ends Nov 1 2026. 07:30 UTC = 00:30 PDT; next midnight is Nov 2 00:00 PST = 08:00 UTC.
    assert seconds_until_daily_reset("midnight_pt", utc(2026, 11, 1, 7, 30)) == 24.5 * 3600


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        seconds_until_daily_reset("midnight_utc", datetime(2026, 10, 24))
