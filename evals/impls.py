"""
The implementations being compared. Every one is driven through the same fake provider.

  single_key       one key, primary model, SDK default retries   ("just call the model")
  round_robin      rotate over every key+model, SDK default retries, no failover, no cooldown
                   (the typical hand-rolled "key rotation")
  v3               the user's pool_manager3, verbatim, configured the way v3 can be:
                   ONE provider-wide rpm/rpd limit (the most conservative model's number)
  v3_loose         same as v3 but its single limit is the MOST generous model's number (the other way
                   to configure v3: no model is pre-stopped too early, but the slower one is over-driven)
  v3_legacy_errors same, but the provider raises the raw google.genai error shape v3's
                   classifier was written for (shows v3 at its best, isolating the error-shape issue)
  merged_reactive  the merged pool WITHOUT a limits file (reactive 429 handling only)
  merged           the merged pool WITH the limits file

Pools are built through their real builders (`get_llm`); only the chat-client class is swapped
for FakeGemini, which receives the exact kwargs the real builder passes (so `max_retries=0`
in the merged builder genuinely switches the SDK retry model off, and v3's missing
`max_retries` genuinely leaves the SDK default of 6 on).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import itertools
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

import yaml
from langchain_core.messages import HumanMessage

from evals import vtime
from evals.provider import FakeGemini, World
from evals.scenarios import MODELS, N_KEYS, Scenario

IMPLS = {
    "single_key": "One key, primary model, SDK retries",
    "round_robin": "Round-robin over keys+models, SDK retries",
    "v3": "pool_manager3 (baseline)",
    "v3_legacy_errors": "pool_manager3, raw-error shape",
    "v3_loose": "pool_manager3, limit = most generous model's number",
    "merged_reactive": "merged, no limits file",
    "merged": "merged + limits file",
}

# Extra variants for the wait-budget experiment: the merged pool with a different `max_wait_seconds`.
WAIT_BUDGETS = (10.0, 30.0, 120.0)           # (60 s is the default, i.e. plain "merged")
WAIT_IMPLS = {f"merged_wait{int(w)}": f"merged + limits file, {int(w)} s wait budget" for w in WAIT_BUDGETS}

_MESSAGES = [HumanMessage("ping")]
Call = Callable[[int], Awaitable[Any]]


@contextlib.contextmanager
def _patched_client(module: Any, world: World, *, legacy_errors: bool = False) -> Iterator[None]:
    original = module.ChatGoogleGenerativeAI
    module.ChatGoogleGenerativeAI = functools.partial(FakeGemini, world=world, legacy_errors=legacy_errors)
    try:
        yield
    finally:
        module.ChatGoogleGenerativeAI = original


def _write_config(tmp: Path, scn: Scenario, extra: dict | None = None) -> Path:
    keys = [f"EVAL_KEY_{k}" for k in range(1, N_KEYS + 1)]
    for k, name in enumerate(keys, start=1):
        os.environ[name] = f"sk-eval-{k}"
    cfg = {"google": {"models": list(MODELS), "keys": keys, **(extra or {})}}
    path = tmp / "llm_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _write_limits(tmp: Path, scn: Scenario) -> Path:
    path = tmp / "model_limits.yaml"
    path.write_text(yaml.safe_dump({"google": scn.yaml_limits()}))
    return path


def v3_limits(scn: Scenario, *, loose: bool) -> dict:
    """pool_manager3 takes ONE rpm and ONE rpd for the whole provider (and requires both).
    Strict = the most conservative model's number (never over-drive anything); loose = the most generous."""
    lim = scn.yaml_limits()
    pick = max if loose else min
    return {
        "rpm_limit": pick(v["rpm"] for v in lim.values()),
        "rpd_limit": pick(v.get("rpd", 10**9) for v in lim.values()),
    }


_V3: Any = None


def _load_v3() -> Any:
    """Import the verbatim v3 baseline with datetime.now() bound to virtual time."""
    global _V3
    if _V3 is None:
        for name in [m for m in sys.modules if m.startswith("evals.baselines.pm_v3.")]:
            del sys.modules[name]
        with vtime.virtual_datetime():
            importlib.import_module("evals.baselines.pm_v3.endpoint_state")
            importlib.import_module("evals.baselines.pm_v3.scheduler")
        _V3 = importlib.import_module("evals.baselines.pm_v3.llm")
    return _V3


def _pool_call(llm: Any) -> Call:
    async def call(i: int) -> Any:
        return await llm._agenerate(_MESSAGES)

    return call


def build(name: str, scn: Scenario, world: World) -> Call:
    """Construct implementation `name` for this scenario/world and return `async call(i)`."""
    tmp = Path(tempfile.mkdtemp(prefix="pool_eval_"))

    if name == "single_key":
        client = FakeGemini(world=world, model=MODELS[0], api_key="sk-eval-1", max_retries=6)

        async def call(i: int) -> Any:
            return await client.ainvoke(_MESSAGES)

        return call

    if name == "round_robin":
        clients = [FakeGemini(world=world, model=s.model, api_key=s.api_key, max_retries=6) for s in world.specs]
        ring = itertools.cycle(clients)

        async def call(i: int) -> Any:
            return await next(ring).ainvoke(_MESSAGES)

        return call

    if name in ("v3", "v3_legacy_errors", "v3_loose"):
        v3 = _load_v3()
        cfg = _write_config(tmp, scn, v3_limits(scn, loose=(name == "v3_loose")))
        with _patched_client(v3, world, legacy_errors=(name == "v3_legacy_errors")):
            llm = v3.get_llm(cfg)                                                # defaults: max_wait_seconds=120
        return _pool_call(llm)

    if name in ("merged", "merged_reactive") or name in WAIT_IMPLS:
        from pool_manager import llm as merged

        cfg = _write_config(tmp, scn)
        limits = _write_limits(tmp, scn) if name != "merged_reactive" else None
        kwargs = {"max_wait_seconds": float(name.removeprefix("merged_wait"))} if name in WAIT_IMPLS else {}
        with _patched_client(merged, world):
            llm = merged.get_llm(cfg, limits_path=limits, **kwargs)              # production builder, defaults
        llm.scheduler.clock = vtime.now
        assert all(ep.client.max_retries == 0 for ep in llm.scheduler.endpoints), "builder must disable SDK retries"
        return _pool_call(llm)

    raise KeyError(name)
