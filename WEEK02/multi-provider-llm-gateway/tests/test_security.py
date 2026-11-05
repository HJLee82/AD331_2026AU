"""Key handling and secret redaction tests (Tests 4–5). No network calls."""

from __future__ import annotations

import logging

import pytest

from src.gateway import AuthenticationError, GatewayError, UnifiedLLMGateway

from .conftest import FakeOpenAI

FAKE_KEY = "sk-proj-TESTSECRET1234567890abcdef"


async def drain(gateway: UnifiedLLMGateway, model: str) -> None:
    async for _ in gateway.stream("hi", model=model):
        pass


# --------------------------------------------------------------------------
# Test 4: Missing / malformed key -> descriptive AuthenticationError
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "env_var"),
    [("gpt-4o-mini", "OPENAI_API_KEY"), ("claude-haiku-5-5", "ANTHROPIC_API_KEY")],
)
async def test_missing_key_raises_authentication_error(monkeypatch, model, env_var):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    gateway = UnifiedLLMGateway()

    with pytest.raises(AuthenticationError) as exc_info:
        await drain(gateway, model)

    assert env_var in str(exc_info.value)  # tells the user exactly what to set


@pytest.mark.parametrize(
    ("model", "env_var", "bad_value"),
    [("gpt-4o-mini", "OPENAI_API_KEY", "abc123"), ("claude-haiku-5-5", "ANTHROPIC_API_KEY", "sk-proj-wrongprovider")],
)
async def test_malformed_key_raises_authentication_error(monkeypatch, model, env_var, bad_value):
    monkeypatch.setenv(env_var, bad_value)
    gateway = UnifiedLLMGateway()

    with pytest.raises(AuthenticationError) as exc_info:
        await drain(gateway, model)

    assert "malformed" in str(exc_info.value)
    assert bad_value not in str(exc_info.value)  # never echo the key back


# --------------------------------------------------------------------------
# Test 5: Secret redaction in exceptions, logs and repr
# --------------------------------------------------------------------------
async def test_exceptions_and_logs_never_contain_the_key(caplog):
    # Simulate an SDK failure whose message leaks the key (e.g. a proxy echoing headers).
    leaky = RuntimeError(f"connection reset; header Authorization: Bearer {FAKE_KEY}")
    gateway = UnifiedLLMGateway(openai_api_key=FAKE_KEY, openai_client=FakeOpenAI(error=leaky))

    with caplog.at_level(logging.DEBUG), pytest.raises(GatewayError) as exc_info:
        await drain(gateway, "gpt-4o-mini")

    err = exc_info.value
    for text in (str(err), repr(err), caplog.text):
        assert FAKE_KEY not in text
        assert FAKE_KEY[:12] not in text  # not even a recognizable fragment
    assert err.__cause__ is None and err.__suppress_context__  # `raise ... from None`: no chained traceback


def test_gateway_repr_hides_keys():
    gateway = UnifiedLLMGateway(openai_api_key=FAKE_KEY, anthropic_api_key="sk-ant-" + "x" * 20)
    text = repr(gateway) + str(gateway)
    assert FAKE_KEY not in text and "sk-ant-" not in text
    assert "configured" in text
