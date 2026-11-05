"""Shared test fakes.

The fakes mimic the *shape* of the OpenAI and Anthropic streaming events, so the
gateway's real parsing code runs without any network call or API cost.
They are injected through UnifiedLLMGateway(openai_client=..., anthropic_client=...).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any

import pytest

EXPECTED_TEXT = "Hello from the gateway."
PIECES = ["Hello ", "from ", "the ", "gateway."]


class FakeStream:
    """Async-iterable stream with the `close()` method both SDK streams have."""

    def __init__(self, events: list[Any], delay: float = 0.005) -> None:
        self._events = events
        self._delay = delay
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self._events:
            await asyncio.sleep(self._delay)  # tiny pause so TTFT and tok/s are > 0
            yield event

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------- OpenAI fakes
def openai_events(prompt_tokens: int = 12, completion_tokens: int = 6) -> list[Any]:
    events = [
        NS(model="gpt-4o-mini-2024-07-18", usage=None,
           choices=[NS(delta=NS(content=None), finish_reason=None)]),  # role-only first chunk
    ]
    events += [
        NS(model="gpt-4o-mini-2024-07-18", usage=None,
           choices=[NS(delta=NS(content=p), finish_reason=None)])
        for p in PIECES
    ]
    events.append(NS(model="gpt-4o-mini-2024-07-18", usage=None,
                     choices=[NS(delta=NS(content=None), finish_reason="stop")]))
    # Final usage chunk: empty choices (the case that crashes naive parsers)
    events.append(NS(model="gpt-4o-mini-2024-07-18", choices=[],
                     usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              prompt_tokens_details=None)))
    return events


class FakeOpenAI:
    def __init__(self, events: list[Any] | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._events = events if events is not None else openai_events()
        self._error = error
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return FakeStream(self._events)

    async def close(self) -> None:
        pass


# ------------------------------------------------------------- Anthropic fakes
def anthropic_events(input_tokens: int = 14, output_tokens: int = 7) -> list[Any]:
    events: list[Any] = [
        NS(type="message_start", message=NS(
            model="claude-haiku-5-5",
            usage=NS(input_tokens=input_tokens, output_tokens=1,
                     cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        NS(type="content_block_start"),
        NS(type="ping"),
    ]
    events += [NS(type="content_block_delta", delta=NS(type="text_delta", text=p)) for p in PIECES]
    events += [
        NS(type="content_block_stop"),
        NS(type="message_delta", usage=NS(output_tokens=output_tokens), delta=NS(stop_reason="end_turn")),
        NS(type="message_stop"),
    ]
    return events


class FakeAnthropic:
    def __init__(self, events: list[Any] | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._events = events if events is not None else anthropic_events()
        self._error = error
        self.messages = NS(create=self._create)

    async def _create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return FakeStream(self._events)

    async def close(self) -> None:
        pass


@pytest.fixture
def fake_gateway():
    """A gateway wired to fake clients: no network, no cost."""
    from src.gateway import UnifiedLLMGateway

    return UnifiedLLMGateway(openai_client=FakeOpenAI(), anthropic_client=FakeAnthropic())
