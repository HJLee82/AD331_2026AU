"""Provider-agnostic data contracts for the Unified LLM Gateway.

Every stream yields ``StreamChunk`` objects. Text chunks carry ``delta_text`` and
``is_final=False``; exactly one terminal chunk carries ``is_final=True`` plus a
``ChunkTelemetry`` payload. Consumers never see provider-specific event shapes.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Provider = Literal["openai", "anthropic"]
UsageSource = Literal["provider", "estimated"]


class ChunkTelemetry(BaseModel):
    """Token, latency and cost metrics attached to the final chunk of a stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0, description="Total prompt tokens, including any cached tokens.")
    output_tokens: int = Field(ge=0, description="Generated (completion) tokens.")
    cached_input_tokens: int = Field(default=0, ge=0, description="Prompt tokens served from cache (billed at the cached rate).")
    cache_write_tokens: int = Field(default=0, ge=0, description="Prompt tokens written to cache (Anthropic only).")

    latency_s: float = Field(ge=0, description="Wall-clock seconds from request start to stream end.")
    ttft_s: Optional[float] = Field(default=None, ge=0, description="Seconds until the first text token. None if no text was produced.")
    tokens_per_second: float = Field(ge=0, description="Decode throughput: output tokens / (end - first token).")

    cost_usd: float = Field(ge=0, description="Total request cost in USD, from the static pricing table.")

    usage_source: UsageSource = Field(description="'provider' if counts came from the API, 'estimated' if reconstructed locally.")
    prompt_tokens_estimate: Optional[int] = Field(default=None, ge=0, description="Local tiktoken estimate (OpenAI only), for verification.")
    finish_reason: Optional[str] = Field(default=None, description="Normalized stop reason reported by the provider.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StreamChunk(BaseModel):
    """A single normalized streaming event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_text: str = Field(default="", description="Newly generated text. Empty on the final chunk.")
    provider: Provider
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None

    @model_validator(mode="after")
    def _telemetry_only_on_final(self) -> "StreamChunk":
        if self.is_final and self.telemetry is None:
            raise ValueError("The final chunk must include telemetry.")
        if not self.is_final and self.telemetry is not None:
            raise ValueError("Telemetry is only allowed on the final chunk.")
        return self
