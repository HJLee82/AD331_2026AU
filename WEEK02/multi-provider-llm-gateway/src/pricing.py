"""Static model registry and USD pricing.

All rates are USD per 1,000,000 tokens. Costs are computed with ``Decimal`` to
avoid float drift, then rounded once at the edge.

Keep this table in sync with the providers' published pricing pages; it is the
single source of truth for cost telemetry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Final, Mapping, Optional

from schemas import Provider

_PER_MILLION: Final = Decimal(1_000_000)
_COST_QUANTUM: Final = Decimal("0.00000001")  # 1e-8 USD


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cached_input_per_mtok: Optional[Decimal] = None  # OpenAI cached input / Anthropic cache read
    cache_write_per_mtok: Optional[Decimal] = None   # Anthropic cache write (5-minute TTL)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    provider: Provider
    api_model: str  # ID sent to the API when the caller uses the bare registry name
    pricing: ModelPricing


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    requested: str
    pricing_key: str
    provider: Provider
    api_model: str
    pricing: ModelPricing


D = Decimal

MODEL_REGISTRY: Final[Mapping[str, ModelSpec]] = MappingProxyType(
    {
        # ---- OpenAI ------------------------------------------------------
        "gpt-4o-mini": ModelSpec(
            provider="openai",
            api_model="gpt-4o-mini",
            pricing=ModelPricing(D("0.15"), D("0.60"), cached_input_per_mtok=D("0.075")),
        ),
        "gpt-4o": ModelSpec(
            provider="openai",
            api_model="gpt-4o",
            pricing=ModelPricing(D("2.50"), D("10.00"), cached_input_per_mtok=D("1.25")),
        ),
        # ---- Anthropic ---------------------------------------------------
        "claude-3-5-sonnet": ModelSpec(
            provider="anthropic",
            api_model="claude-3-5-sonnet-latest",
            pricing=ModelPricing(D("3.00"), D("15.00"), cached_input_per_mtok=D("0.30"), cache_write_per_mtok=D("3.75")),
        ),
        "claude-3-5-haiku": ModelSpec(
            provider="anthropic",
            api_model="claude-3-5-haiku-latest",
            pricing=ModelPricing(D("0.80"), D("4.00"), cached_input_per_mtok=D("0.08"), cache_write_per_mtok=D("1.00")),
        ),
    }
)

# Accept pinned snapshots of registered models, e.g. "gpt-4o-2024-08-06",
# "claude-3-5-haiku-20241022", "claude-3-5-sonnet-latest". Anything else
# (e.g. "gpt-4o-audio-preview") is rejected rather than silently mis-priced.
_SNAPSHOT_SUFFIX: Final = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|latest)$")


def resolve_model(model: str) -> ResolvedModel:
    """Map a user-supplied model name to its provider, API ID and pricing.

    Raises:
        KeyError: if the model (or its snapshot base) is not in the registry.
    """
    key = model.strip().lower()

    spec = MODEL_REGISTRY.get(key)
    if spec is not None:
        return ResolvedModel(model, key, spec.provider, spec.api_model, spec.pricing)

    match = _SNAPSHOT_SUFFIX.search(key)
    if match:
        base = key[: match.start()]
        spec = MODEL_REGISTRY.get(base)
        if spec is not None:
            # Caller pinned a specific snapshot: send it to the API verbatim.
            return ResolvedModel(model, base, spec.provider, key, spec.pricing)

    raise KeyError(model)


def supported_models() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def calculate_cost(
    pricing: ModelPricing,
    *,
    uncached_input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    """Return the request cost in USD.

    ``uncached_input_tokens`` must exclude cache reads/writes; those are billed
    separately at their own rates (falling back to the base input rate).
    """
    for name, value in (
        ("uncached_input_tokens", uncached_input_tokens),
        ("output_tokens", output_tokens),
        ("cache_read_tokens", cache_read_tokens),
        ("cache_write_tokens", cache_write_tokens),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    read_rate = pricing.cached_input_per_mtok if pricing.cached_input_per_mtok is not None else pricing.input_per_mtok
    write_rate = pricing.cache_write_per_mtok if pricing.cache_write_per_mtok is not None else pricing.input_per_mtok

    total = (
        D(uncached_input_tokens) * pricing.input_per_mtok
        + D(cache_read_tokens) * read_rate
        + D(cache_write_tokens) * write_rate
        + D(output_tokens) * pricing.output_per_mtok
    ) / _PER_MILLION

    return total.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)
