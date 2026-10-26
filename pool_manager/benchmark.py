"""
Latency / token / cost tracking for every LLM call in the pipeline.

Every phase shares one MultiProviderChatLLM instance (see llm_registry.py),
and every phase's chain ends in either `.with_structured_output(...)` or a
plain chat call -- so this is a small, self-contained side-channel tracker,
recorded from the single failover helper in pool_manager/llm.py that already
does rate-limit bookkeeping (EndpointState), so sync, async, tool-bound and
structured-output calls are all covered. Kept out of the pipeline's
`State` for the same reason EndpointState is: it's infra/telemetry, not
pipeline domain data.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PRICING_PATH = "configs/pricing.yaml"


@dataclass
class CallRecord:
    phase: str
    provider: str
    model: str
    latency_s: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    endpoint_id: Optional[str] = None  # "provider:model:KEY_ENV_NAME" (never the key itself)


@dataclass
class _Bucket:
    call_count: int = 0
    total_latency_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tokens_known: bool = False

    def add(self, record: CallRecord) -> None:
        self.call_count += 1
        self.total_latency_s += record.latency_s
        if record.input_tokens is not None and record.output_tokens is not None:
            self.total_input_tokens += record.input_tokens
            self.total_output_tokens += record.output_tokens
            self.tokens_known = True

    def as_dict(self) -> dict:
        avg_latency = self.total_latency_s / self.call_count if self.call_count else 0.0
        return {
            "call_count": self.call_count,
            "total_latency_s": round(self.total_latency_s, 3),
            "avg_latency_s": round(avg_latency, 3),
            "total_input_tokens": self.total_input_tokens if self.tokens_known else None,
            "total_output_tokens": self.total_output_tokens if self.tokens_known else None,
        }


class BenchmarkTracker:
    """Accumulates CallRecords in memory for the life of one pipeline run."""

    def __init__(self, pricing_path: str | Path = PRICING_PATH) -> None:
        self._by_phase: dict[str, _Bucket] = defaultdict(_Bucket)
        self._by_endpoint: dict[tuple[str, str], _Bucket] = defaultdict(_Bucket)
        self._by_key: dict[str, _Bucket] = defaultdict(_Bucket)
        self._call_count = 0
        self._lock = threading.Lock()  # sync calls may come from worker threads
        self._pricing = self._load_pricing(pricing_path)

    @staticmethod
    def _load_pricing(path: str | Path) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            with open(p) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            # A malformed pricing file should never break a pipeline run --
            # cost just reports as unknown.
            return {}

    def record(self, record: CallRecord) -> None:
        with self._lock:
            self._call_count += 1
            self._by_phase[record.phase].add(record)
            self._by_endpoint[(record.provider, record.model)].add(record)
            if record.endpoint_id:
                self._by_key[record.endpoint_id].add(record)

    def _cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        rates = self._pricing.get(model)
        if not rates:
            return None
        in_rate = rates.get("input_per_1m")
        out_rate = rates.get("output_per_1m")
        if in_rate is None or out_rate is None:
            return None
        return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate

    def summary(self) -> dict:
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        by_phase = {phase: b.as_dict() for phase, b in self._by_phase.items()}
        by_key = {endpoint_id: b.as_dict() for endpoint_id, b in self._by_key.items()}

        by_endpoint: dict[str, dict] = {}
        total_cost = 0.0
        cost_known = False
        for (provider, model), b in self._by_endpoint.items():
            entry = b.as_dict()
            cost = (
                self._cost_usd(model, b.total_input_tokens, b.total_output_tokens)
                if b.tokens_known
                else None
            )
            if cost is not None:
                total_cost += cost
                cost_known = True
            entry["estimated_cost_usd"] = round(cost, 4) if cost is not None else None
            by_endpoint[f"{provider}:{model}"] = entry

        return {
            "total_calls": self._call_count,
            "total_estimated_cost_usd": round(total_cost, 4) if cost_known else None,
            "by_phase": by_phase,
            "by_endpoint": by_endpoint,
            "by_key": by_key,
        }


def render_markdown(summary: dict) -> str:
    """Human-readable counterpart to summary() -- outputs/benchmark.md."""
    lines = ["# Benchmark Report", "", f"Total LLM calls: {summary['total_calls']}"]

    total_cost = summary["total_estimated_cost_usd"]
    lines.append(
        f"Total estimated cost: ${total_cost}"
        if total_cost is not None
        else "Total estimated cost: n/a (no pricing configured -- see configs/pricing.yaml)"
    )

    lines += ["", "## By Phase", ""]
    for phase, b in summary["by_phase"].items():
        tokens = (
            f", {b['total_input_tokens']} in / {b['total_output_tokens']} out tokens"
            if b["total_input_tokens"] is not None
            else ""
        )
        lines.append(
            f"- **{phase}** — {b['call_count']} calls, "
            f"avg latency {b['avg_latency_s']}s, total {b['total_latency_s']}s{tokens}"
        )

    lines += ["", "## By Endpoint", ""]
    for endpoint, b in summary["by_endpoint"].items():
        cost = f"${b['estimated_cost_usd']}" if b["estimated_cost_usd"] is not None else "n/a"
        lines.append(
            f"- **{endpoint}** — {b['call_count']} calls, "
            f"avg latency {b['avg_latency_s']}s, cost {cost}"
        )

    if summary.get("by_key"):
        lines += ["", "## By API Key (endpoint)", ""]
        for endpoint_id, b in summary["by_key"].items():
            lines.append(
                f"- **{endpoint_id}** — {b['call_count']} calls, avg latency {b['avg_latency_s']}s"
            )

    return "\n".join(lines) + "\n"


_tracker: Optional[BenchmarkTracker] = None


def get_tracker() -> BenchmarkTracker:
    global _tracker
    if _tracker is None:
        _tracker = BenchmarkTracker()
    return _tracker


def reset_tracker(pricing_path: str | Path = PRICING_PATH) -> BenchmarkTracker:
    """Starts a fresh tracker -- used at the top of a pipeline run, and by tests."""
    global _tracker
    _tracker = BenchmarkTracker(pricing_path)
    return _tracker
