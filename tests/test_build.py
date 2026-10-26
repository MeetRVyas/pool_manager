"""build_endpoints/get_llm with the REAL client classes (dummy keys, no network)."""
import logging

import pytest
import yaml

from pool_manager.llm import build_endpoints, describe_endpoints, get_llm
from pool_manager.limits import LimitsCatalogue
from tests.fakes import FakeClient, run

LIMITS = """
google:
  gemini-2.5-flash:      {rpm: 10, tpm: 250000, rpd: 250, daily_reset: midnight_pt}
  gemini-2.5-flash-lite: {rpm: 15}
"""


@pytest.fixture
def env(monkeypatch):
    for i in (1, 2, 3):
        monkeypatch.setenv(f"GEMINI_KEY_{i}", f"gem-secret-{i}")
    monkeypatch.setenv("GROQ_KEY_1", "groq-secret-1")
    return monkeypatch


def write(tmp_path, cfg: dict, name="llm_config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    return path


def limits_file(tmp_path, text=LIMITS):
    path = tmp_path / "model_limits.yaml"
    path.write_text(text)
    return path


CFG = {
    "google": {"models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"], "keys": ["GEMINI_KEY_1", "GEMINI_KEY_2", "GEMINI_KEY_3"]},
    "groq": {"base_url": "https://api.groq.test/v1", "models": ["llama-3.3-70b"], "keys": ["GROQ_KEY_1"]},
}


def secret(client):
    key = getattr(client, "google_api_key", None) or client.openai_api_key
    return key.get_secret_value()


# ------------------------------------------------------------- every key is wired in
def test_every_key_is_attached_to_every_model_of_its_provider(env, tmp_path):
    eps = build_endpoints(write(tmp_path, CFG), limits_path=limits_file(tmp_path))
    assert len(eps) == 2 * 3 + 1
    google = [e for e in eps if e.provider == "google"]
    for model in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
        keys_for_model = sorted(secret(e.client) for e in google if e.model == model)
        assert keys_for_model == ["gem-secret-1", "gem-secret-2", "gem-secret-3"]   # all 3 keys, each model
    for e in eps:                                                                   # id names the env var, never the secret
        assert "secret" not in e.id
        assert e.id == f"{e.provider}:{e.model}:{e.id.split(':')[-1]}"


def test_endpoint_order_is_preference_order_model_major(env, tmp_path):
    eps = build_endpoints(write(tmp_path, CFG), limits_path=limits_file(tmp_path))
    assert [e.id for e in eps[:4]] == [
        "google:gemini-2.5-flash:GEMINI_KEY_1", "google:gemini-2.5-flash:GEMINI_KEY_2",
        "google:gemini-2.5-flash:GEMINI_KEY_3", "google:gemini-2.5-flash-lite:GEMINI_KEY_1",
    ]
    assert [e.priority for e in eps] == [0, 0, 0, 1, 1, 1, 2]      # all keys of a model share a priority tier


def test_sdk_internal_retries_are_disabled_and_a_timeout_is_set(env, tmp_path):
    eps = build_endpoints(write(tmp_path, CFG), limits_path=None, request_timeout=45)
    google = next(e for e in eps if e.provider == "google").client
    groq = next(e for e in eps if e.provider == "groq").client
    assert google.max_retries == 0 and google.timeout == 45         # default would be 6 retries, no timeout
    assert groq.max_retries == 0 and groq.request_timeout == 45     # default would be 2 retries, 600 s


def test_provider_level_timeout_overrides_the_global_one(env, tmp_path):
    cfg = {"google": {**CFG["google"], "timeout": 300}}
    eps = build_endpoints(write(tmp_path, cfg), limits_path=None, request_timeout=45)
    assert eps[0].client.timeout == 300


def test_temperature_goes_to_openai_compatible_clients_only(env, tmp_path):
    eps = build_endpoints(write(tmp_path, CFG), temperature=0.3, limits_path=None)
    assert next(e for e in eps if e.provider == "groq").client.temperature == 0.3
    assert next(e for e in eps if e.provider == "google").client.temperature != 0.3   # v3 behaviour kept


# -------------------------------------------------------------------- limits wiring
def test_limits_come_from_the_yaml_and_are_reported(env, tmp_path):
    eps = build_endpoints(write(tmp_path, CFG), limits_path=limits_file(tmp_path))
    flash = eps[0]
    assert (flash.rpm_limit, flash.tpm_limit, flash.rpd_limit, flash.daily_reset) == (10, 250000, 250, "midnight_pt")
    assert "model_limits.yaml" in flash.limits_source
    assert eps[3].rpm_limit == 15                                   # flash-lite
    unknown = eps[-1]                                               # llama: not in the file
    assert unknown.rpm_limit is None and "reactive" in unknown.limits_source
    assert "reactive only" in describe_endpoints(eps)


def test_old_v3_style_provider_level_limits_still_work(env, tmp_path):
    cfg = {"google": {**CFG["google"], "rpm_limit": 5, "rpd_limit": 20}}
    eps = build_endpoints(write(tmp_path, cfg), limits_path=None)
    assert all((e.rpm_limit, e.rpd_limit) == (5, 20) for e in eps)


def test_yaml_beats_provider_defaults_field_by_field(env, tmp_path):
    cfg = {"google": {**CFG["google"], "rpm_limit": 99, "rpd_limit": 20}}
    eps = build_endpoints(write(tmp_path, cfg), limits_path=limits_file(tmp_path))
    assert (eps[0].rpm_limit, eps[0].rpd_limit) == (10, 250)
    assert (eps[3].rpm_limit, eps[3].rpd_limit) == (15, 20)         # lite has no rpd in YAML -> default fills in


def test_safety_margin_pre_stops_early(env, tmp_path):
    eps = build_endpoints(write(tmp_path, CFG), limits_path=limits_file(tmp_path), safety_margin=0.2)
    assert eps[0].rpm_limit == 8 and eps[0].rpd_limit == 200


def test_limits_can_be_passed_as_a_prebuilt_catalogue(env, tmp_path):
    cat = LimitsCatalogue.from_dict({"google": {"gemini-2.5-flash": {"rpm": 1}}})
    assert build_endpoints(write(tmp_path, CFG), limits_path=cat)[0].rpm_limit == 1


def test_explicit_missing_limits_file_is_an_error_but_auto_is_a_warning(env, tmp_path, monkeypatch, caplog):
    cfg = write(tmp_path, CFG)
    with pytest.raises(FileNotFoundError):
        build_endpoints(cfg, limits_path=tmp_path / "nope.yaml")
    monkeypatch.chdir(tmp_path)                                     # no configs/model_limits.yaml here
    with caplog.at_level(logging.WARNING, logger="pool_manager"):
        eps = build_endpoints(cfg)
    assert all(e.rpm_limit is None for e in eps) and "WITHOUT proactive rate limits" in caplog.text


def test_auto_picks_up_configs_model_limits_yaml(env, tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "model_limits.yaml").write_text(LIMITS)
    monkeypatch.chdir(tmp_path)
    assert build_endpoints(write(tmp_path, CFG))[0].rpm_limit == 10


def test_a_typo_in_the_limits_file_fails_at_startup_not_silently(env, tmp_path):
    bad = limits_file(tmp_path, "google: {gemini-2.5-flash: {rmp: 10}}")
    with pytest.raises(ValueError, match="unknown field"):
        build_endpoints(write(tmp_path, CFG), limits_path=bad)


# ------------------------------------------------------------------ config hygiene
def test_missing_env_var_is_reported_by_name(env, tmp_path):
    cfg = {"google": {"models": ["m"], "keys": ["GEMINI_KEY_1", "NOT_SET_ANYWHERE"]}}
    with pytest.raises(ValueError, match="NOT_SET_ANYWHERE"):
        build_endpoints(write(tmp_path, cfg), limits_path=None)


def test_two_env_vars_holding_the_same_key_are_used_once(env, tmp_path, caplog):
    env.setenv("GEMINI_KEY_2", "gem-secret-1")                      # same value as KEY_1
    cfg = {"google": {"models": ["m"], "keys": ["GEMINI_KEY_1", "GEMINI_KEY_2", "GEMINI_KEY_3"]}}
    with caplog.at_level(logging.WARNING, logger="pool_manager"):
        eps = build_endpoints(write(tmp_path, cfg), limits_path=None)
    assert [e.id.split(":")[-1] for e in eps] == ["GEMINI_KEY_1", "GEMINI_KEY_3"]
    assert "same API key" in caplog.text


@pytest.mark.parametrize("bad", [{"google": {"models": [], "keys": ["GEMINI_KEY_1"]}}, {"google": {"models": ["m"]}}, {"google": "x"}, {}])
def test_malformed_config_is_rejected(env, tmp_path, bad):
    with pytest.raises(ValueError):
        build_endpoints(write(tmp_path, bad), limits_path=None)


# ------------------------------------------------------------ whole thing, all keys
def test_get_llm_end_to_end_every_key_receives_traffic(env, tmp_path):
    llm = get_llm(write(tmp_path, CFG), limits_path=limits_file(tmp_path), strategy="balanced")
    for ep in llm.scheduler.endpoints:                              # swap the network clients for fakes
        ep.client = FakeClient(ep.id)
    for i in range(14):
        run(llm.ainvoke(str(i)))
    assert llm.scheduler.unused_endpoints() == []
    used_keys = {e["id"].split(":")[-1] for e in llm.stats() if e["calls"]}
    assert used_keys == {"GEMINI_KEY_1", "GEMINI_KEY_2", "GEMINI_KEY_3", "GROQ_KEY_1"}


def test_get_llm_priority_mode_rotates_all_keys_of_the_active_model(env, tmp_path):
    llm = get_llm(write(tmp_path, CFG), limits_path=limits_file(tmp_path))
    for ep in llm.scheduler.endpoints:
        ep.client = FakeClient(ep.id)
    for i in range(9):
        run(llm.ainvoke(str(i)))
    calls = {e["id"]: e["calls"] for e in llm.stats()}
    assert [calls[f"google:gemini-2.5-flash:GEMINI_KEY_{i}"] for i in (1, 2, 3)] == [3, 3, 3]
    assert sum(v for k, v in calls.items() if "lite" in k or "groq" in k) == 0   # fallbacks untouched until needed
