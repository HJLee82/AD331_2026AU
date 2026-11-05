"""Streaming protocol, token counting and telemetry tests (Tests 1–3)."""

from __future__ import annotations

import os

import pytest
import tiktoken
from dotenv import find_dotenv, load_dotenv

from src.gateway import UnifiedLLMGateway
from src.schemas import StreamChunk

from .conftest import EXPECTED_TEXT

MODELS = ["gpt-4o-mini", "claude-haiku-5-5"]


async def collect(gateway: UnifiedLLMGateway, model: str, prompt: str = "Say hello.") -> list[StreamChunk]:
    return [chunk async for chunk in gateway.stream(prompt, model=model)]


# --------------------------------------------------------------------------
# Test 1: Unified Stream Protocol
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", MODELS)
async def test_unified_stream_protocol(fake_gateway, model):
    chunks = await collect(fake_gateway, model)

    assert all(isinstance(c, StreamChunk) for c in chunks)
    finals = [c for c in chunks if c.is_final]
    assert len(finals) == 1 and chunks[-1].is_final, "exactly one final chunk, at the end"
    assert all(c.telemetry is None for c in chunks[:-1])

    text = "".join(c.delta_text for c in chunks if not c.is_final)
    assert text == EXPECTED_TEXT  # same text, whichever provider produced it
    assert {c.provider for c in chunks} == {"openai" if model.startswith("gpt") else "anthropic"}


# --------------------------------------------------------------------------
# Test 2: Token Count Validation (live, ~$0.00001 per run)
# --------------------------------------------------------------------------
FIFTY_WORDS = (
    "Streaming gateways normalize provider events into one schema so applications stay "
    "independent of vendors. Each request records latency, time to first token, throughput, "
    "and cost. Engineers compare models by measuring these numbers under identical prompts. "
    "Careful accounting prevents surprise bills and reveals which provider responds fastest "
    "for interactive workloads today."
)

load_dotenv(find_dotenv(usecwd=True))
has_openai_key = bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not has_openai_key, reason="needs OPENAI_API_KEY for a live call")
async def test_prompt_tokens_match_tiktoken_within_10_percent():
    assert len(FIFTY_WORDS.split()) == 50

    async with UnifiedLLMGateway() as gateway:
        chunks = [c async for c in gateway.stream(FIFTY_WORDS, model="gpt-4o-mini", temperature=0)]

    telemetry = chunks[-1].telemetry
    assert telemetry is not None and telemetry.usage_source == "provider"

    # The provider counts the whole chat message, not just the text:
    # 3 tokens per message + the role ("user") + the text + 3 tokens priming the reply.
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    local_count = 3 + len(enc.encode("user")) + len(enc.encode(FIFTY_WORDS)) + 3
    reported = telemetry.input_tokens
    assert abs(reported - local_count) <= 0.10 * local_count, (reported, local_count)
    assert telemetry.prompt_tokens_estimate == local_count  # gateway's own estimate uses the same formula


# --------------------------------------------------------------------------
# Test 3: Telemetry Sanity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", MODELS)
async def test_final_chunk_telemetry_is_sane(fake_gateway, model):
    final = (await collect(fake_gateway, model))[-1]
    t = final.telemetry

    assert final.is_final and t is not None
    assert t.total_cost_usd > 0
    assert t.tokens_per_second > 0
    assert t.ttft_ms is not None and t.ttft_ms > 0
    assert t.input_tokens > 0 and t.output_tokens > 0
    assert round(t.total_cost_usd, 6) == pytest.approx(t.total_cost_usd, abs=1e-6)


async def test_cost_matches_pricing_table(fake_gateway):
    """gpt-4o-mini: 12 prompt + 6 completion tokens at $0.15 / $0.60 per 1M."""
    t = (await collect(fake_gateway, "gpt-4o-mini"))[-1].telemetry
    assert t.total_cost_usd == pytest.approx(12 * 0.15 / 1e6 + 6 * 0.60 / 1e6)


async def test_claude_5_does_not_send_temperature():
    """Regression test for audit bug B5."""
    from .conftest import FakeAnthropic

    fake = FakeAnthropic()
    gateway = UnifiedLLMGateway(anthropic_client=fake)
    await collect(gateway, "claude-haiku-5-5")
    assert "extra_body" not in fake.calls[0] and "temperature" not in fake.calls[0]
