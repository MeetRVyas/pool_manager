"""
Model rate-limit catalogue.

The catalogue is a YAML file that records the *configured* rate limits of the
models you use, so the pool can stop sending to an endpoint BEFORE the provider
answers 429. A model that has no entry is simply not pre-limited: the pool then
relies on reactive handling (cooldown + failover when a 429 actually arrives).

Illustrative file format (counters are per endpoint; Gemini quotas are per project)::

    google:                                  # provider name, as used in llm_config
      gemini-2.5-flash:                      # exact model name ...
        rpm: 10                              #   requests per minute
        tpm: 250000                          #   tokens per minute
        rpd: 250                             #   requests per day
        daily_reset: midnight_pt             #   rolling (default) | midnight_pt | midnight_utc
      "gemini-2.5-flash-lite*":              # ... or a glob; the most specific match wins
        rpm: 15
      "*":                                   # fallback for any other model of this provider
        rpm: 5
    "*":                                     # provider-agnostic section (checked last)
      "llama-*": {rpm: 30}

Every field is optional; omit a field to leave that dimension unlimited.
Unknown fields and non-positive numbers are rejected so a typo (``rmp: 10``)
cannot silently disable a limit.
"""

from __future__ import annotations

import fnmatch
import functools
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

logger = logging.getLogger("pool_manager.limits")

DAILY_RESET_MODES = ("rolling", "midnight_pt", "midnight_utc")

# yaml field name (and accepted aliases) -> canonical field
_FIELD_ALIASES = {
    "rpm": "rpm", "rpm_limit": "rpm",
    "rpd": "rpd", "rpd_limit": "rpd",
    "tpm": "tpm", "tpm_limit": "tpm",
    "daily_reset": "daily_reset",
}

# `gemini` and `google` both mean the Google GenAI client (matches build_endpoints).
_PROVIDER_ALIASES = {"gemini": "google"}


def _norm_provider(name: str) -> str:
    name = str(name).strip().lower()
    return _PROVIDER_ALIASES.get(name, name)


@dataclass(frozen=True)
class Limits:
    """Per-endpoint limits. ``None`` means 'not limited proactively'."""

    rpm: Optional[int] = None
    rpd: Optional[int] = None
    tpm: Optional[int] = None
    daily_reset: str = "rolling"
    source: str = "none (reactive handling only)"

    @property
    def is_empty(self) -> bool:
        return self.rpm is None and self.rpd is None and self.tpm is None

    def with_margin(self, margin: float) -> "Limits":
        """Pre-stop early: margin=0.1 enforces 90% of each limit (never below 1)."""
        if not 0.0 <= margin < 1.0:
            raise ValueError(f"safety_margin must be in [0, 1), got {margin!r}")
        if margin == 0.0:
            return self

        def shrink(v: Optional[int]) -> Optional[int]:
            return None if v is None else max(1, int(v * (1.0 - margin)))

        return replace(self, rpm=shrink(self.rpm), rpd=shrink(self.rpd), tpm=shrink(self.tpm))

    def describe(self) -> str:
        parts = []
        if self.rpm is not None:
            parts.append(f"rpm={self.rpm}")
        if self.tpm is not None:
            parts.append(f"tpm={self.tpm}")
        if self.rpd is not None:
            parts.append(f"rpd={self.rpd}({self.daily_reset})")
        return " ".join(parts) if parts else "unlimited"


def parse_limits_entry(raw: Any, where: str) -> Limits:
    """Validate one ``{rpm: .., rpd: .., tpm: .., daily_reset: ..}`` mapping."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping such as {{rpm: 10, rpd: 250}}, got {raw!r}")
    fields: dict[str, Any] = {}
    for key, value in raw.items():
        name = _FIELD_ALIASES.get(str(key).strip().lower())
        if name is None:
            raise ValueError(
                f"{where}: unknown field {key!r} (allowed: rpm, rpd, tpm, daily_reset)"
            )
        if name == "daily_reset":
            if value not in DAILY_RESET_MODES:
                raise ValueError(
                    f"{where}: daily_reset must be one of {DAILY_RESET_MODES}, got {value!r}"
                )
        else:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{where}: {key!r} must be a positive integer, got {value!r}")
        fields[name] = value
    return Limits(**fields)


def _specificity(pattern: str) -> int:
    """How many literal characters a glob pins down (longer = more specific)."""
    return sum(1 for ch in pattern if ch not in "*?[]")


class LimitsCatalogue:
    """Looks up the limits for a (provider, model) pair."""

    def __init__(
        self,
        sections: Optional[Mapping[str, Mapping[str, Limits]]] = None,
        origin: str = "<memory>",
    ) -> None:
        self._sections: dict[str, dict[str, Limits]] = {
            _norm_provider(p): {str(pat).lower(): lim for pat, lim in entries.items()}
            for p, entries in (sections or {}).items()
        }
        self.origin = origin

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: Any, origin: str = "<dict>") -> "LimitsCatalogue":
        if data is None:
            return cls({}, origin)
        if not isinstance(data, Mapping):
            raise ValueError(f"{origin}: top level must map provider -> models, got {type(data).__name__}")
        sections: dict[str, dict[str, Limits]] = {}
        for provider, entries in data.items():
            if not isinstance(entries, Mapping):
                raise ValueError(f"{origin}: section {provider!r} must map model -> limits")
            section = sections.setdefault(_norm_provider(provider), {})
            for pattern, raw in entries.items():
                section[str(pattern)] = parse_limits_entry(raw, f"{origin}: {provider}/{pattern}")
        return cls(sections, origin)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "LimitsCatalogue":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, origin=str(path))

    # ------------------------------------------------------------------- lookup
    def lookup(self, provider: str, model: str) -> Optional[Limits]:
        """Exact name first, then the most specific glob; provider section, then ``*``."""
        model_l = model.lower()
        for section_name in (_norm_provider(provider), "*"):
            entries = self._sections.get(section_name)
            if not entries:
                continue
            hit = self._match(entries, model_l)
            if hit is not None:
                pattern, limits = hit
                return replace(limits, source=f"{self.origin} [{section_name}/{pattern}]")
        return None

    @staticmethod
    def _match(entries: Mapping[str, Limits], model_l: str) -> Optional[tuple[str, Limits]]:
        if model_l in entries:
            return model_l, entries[model_l]
        best: Optional[tuple[str, Limits]] = None
        for pattern, limits in entries.items():
            if fnmatch.fnmatchcase(model_l, pattern):
                if best is None or _specificity(pattern) > _specificity(best[0]):
                    best = (pattern, limits)
        return best

    def __bool__(self) -> bool:
        return any(self._sections.values())


def resolve_limits(
    provider: str,
    model: str,
    catalogue: Optional[LimitsCatalogue],
    provider_params: Mapping[str, Any],
) -> Limits:
    """
    Limits for one model, most specific source first, merged field by field:

      1. the catalogue entry for this model            (model-specific)
      2. rpm_limit / rpd_limit / tpm_limit in the      (provider-wide default,
         provider's block of llm_config                 what the old config used)
      3. nothing                                       (no pre-stop; reactive only)
    """
    provider_fields = {
        k: provider_params[k] for k in ("rpm_limit", "rpd_limit", "tpm_limit") if provider_params.get(k) is not None
    }
    base = replace(
        parse_limits_entry(provider_fields, f"llm_config[{provider}]"),
        source=f"llm_config provider defaults [{provider}]",
    )
    hit = catalogue.lookup(provider, model) if catalogue else None
    if hit is None:
        return base if not base.is_empty else Limits()

    merged = Limits(
        rpm=hit.rpm if hit.rpm is not None else base.rpm,
        rpd=hit.rpd if hit.rpd is not None else base.rpd,
        tpm=hit.tpm if hit.tpm is not None else base.tpm,
        daily_reset=hit.daily_reset,
        source=hit.source,
    )
    return merged


# --------------------------------------------------------------------- daily reset
@functools.lru_cache(maxsize=1)
def _pacific_tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/Los_Angeles")
    except Exception:  # no tz database (e.g. Windows without the `tzdata` package)
        logger.warning(
            "Time zone data for America/Los_Angeles not found (pip install tzdata); "
            "falling back to a fixed UTC-8 offset for 'midnight_pt' daily resets."
        )
        return timezone(timedelta(hours=-8), "PST")


def seconds_until_daily_reset(mode: str, wall_now: datetime) -> Optional[float]:
    """
    Seconds from `wall_now` (timezone-aware) to the provider's next daily reset,
    or None for 'rolling' (no fixed reset instant).
    """
    if mode == "rolling":
        return None
    if wall_now.tzinfo is None:
        raise ValueError("wall_now must be timezone-aware")
    if mode == "midnight_utc":
        now_utc = wall_now.astimezone(timezone.utc)
        nxt = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (nxt - now_utc).total_seconds()
    if mode == "midnight_pt":
        zone = _pacific_tz()
        local = wall_now.astimezone(zone)
        nxt_local = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        # Subtract in UTC: aware datetimes sharing a tzinfo subtract as naive wall-clock
        # time, which is an hour off on DST-change days.
        return (nxt_local.astimezone(timezone.utc) - wall_now.astimezone(timezone.utc)).total_seconds()
    raise ValueError(f"unknown daily_reset mode {mode!r}")
