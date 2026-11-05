"""Unified async streaming gateway for OpenAI and Anthropic.

Usage:
    async with UnifiedLLMGateway() as gw:
        async for chunk in gw.stream("Hello!", model="gpt-4o-mini"):
            if chunk.is_final:
                print(chunk.telemetry)
            else:
                print(chunk.delta_text, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from functools import lru_cache
from typing import Any, Final, Optional

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr

from .pricing import ResolvedModel, calculate_cost, resolve_model, supported_models
from .schemas import ChunkTelemetry, Provider, StreamChunk, UsageSource

__all__ = [
    "UnifiedLLMGateway",
    "GatewayError",
    "UnsupportedModelError",
    "InvalidRequestError",
    "GatewayProviderError",
    "RateLimitExceededError",
    "AuthenticationError",
    "ProviderConnectionError",
    "ProviderTimeoutError",
    "ProviderBadRequestError",
]

logger = logging.getLogger("llm_gateway")


def _transport_exception_types() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    """Collect (timeout, transport) exception classes from httpx and/or httpx2.

    Older openai/anthropic SDKs use ``httpx``; newer releases use ``httpx2``.
    Mid-stream network failures can surface as raw transport errors rather than
    SDK ``APIConnectionError``s, so both families are caught.
    """
    timeouts: list[type[BaseException]] = []
    transports: list[type[BaseException]] = []
    for name in ("httpx", "httpx2"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        timeouts.append(mod.TimeoutException)
        transports.append(mod.TransportError)
    return tuple(timeouts), tuple(transports)


_TIMEOUT_ERRORS, _TRANSPORT_ERRORS = _transport_exception_types()


# ============================================================================
# Domain exceptions
# ============================================================================


class GatewayError(Exception):
    """Base class for every error raised by the gateway."""


class UnsupportedModelError(GatewayError):
    """The requested model is not in the pricing registry."""


class InvalidRequestError(GatewayError):
    """The request failed local validation before reaching any provider."""


class GatewayProviderError(GatewayError):
    """A provider call failed. Safe to log: never contains credentials."""

    def __init__(
        self,
        message: str,
        *,
        provider: Provider,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable

    def __str__(self) -> str:
        parts = [f"[{self.provider}] {self.message}"]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class RateLimitExceededError(GatewayProviderError):
    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(GatewayProviderError):
    """Missing, invalid, or insufficiently-permissioned API key."""


class ProviderConnectionError(GatewayProviderError):
    """Network-level failure reaching the provider."""


class ProviderTimeoutError(ProviderConnectionError):
    """The provider did not respond within the configured timeout."""


class ProviderBadRequestError(GatewayProviderError):
    """The provider rejected the request (400/404/422), e.g. an unknown or retired model."""


# ============================================================================
# Secret redaction
# ============================================================================

_SECRET_PATTERN: Final = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]+|x-api-key[\"':=\s]+[A-Za-z0-9._\-]+)",
    re.IGNORECASE,
)
_MAX_ERROR_MESSAGE_LEN: Final = 300


def _redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _SECRET_PATTERN.sub("[REDACTED]", text)
    return text[:_MAX_ERROR_MESSAGE_LEN]


def _to_secret(value: str | SecretStr | None, env_var: str) -> Optional[SecretStr]:
    if isinstance(value, SecretStr):
        return value if value.get_secret_value() else None
    raw = value if value is not None else os.environ.get(env_var)
    return SecretStr(raw) if raw else None


# ============================================================================
# tiktoken helpers (OpenAI prompt verification)
# ============================================================================

# OpenAI cookbook heuristic for chat models: 3 tokens per message wrapper,
# plus the role string, plus 3 tokens priming the assistant reply.
_TOKENS_PER_MESSAGE: Final = 3
_REPLY_PRIMING_TOKENS: Final = 3
_ESTIMATE_TOLERANCE: Final = 0.02  # 2% (plus a small absolute slack)


@lru_cache(maxsize=16)
def _encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")  # gpt-4o family encoding


def _count_chat_prompt_tokens_sync(prompt: str, model: str) -> int:
    enc = _encoding_for(model)
    return _TOKENS_PER_MESSAGE + len(enc.encode("user")) + len(enc.encode(prompt)) + _REPLY_PRIMING_TOKENS


def _count_text_tokens_sync(text: str, model: str) -> int:
    return len(_encoding_for(model).encode(text))


async def _estimate_openai_prompt_tokens(prompt: str, model: str) -> Optional[int]:
    # Encoding load can hit disk/network on first use; keep it off the event loop.
    try:
        return await asyncio.to_thread(_count_chat_prompt_tokens_sync, prompt, model)
    except Exception as exc:  # tiktoken must never break a request
        logger.warning("tiktoken prompt estimate unavailable: %s", type(exc).__name__)
        return None


# ============================================================================
# Gateway
# ============================================================================

_TEMPERATURE_RANGE: Final[dict[Provider, tuple[float, float]]] = {
    "openai": (0.0, 2.0),
    "anthropic": (0.0, 1.0),
}


class UnifiedLLMGateway:
    """Async, streaming, provider-agnostic LLM client with cost telemetry."""

    def __init__(
        self,
        *,
        openai_api_key: str | SecretStr | None = None,
        anthropic_api_key: str | SecretStr | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        anthropic_max_tokens: int = 4096,
        openai_client: AsyncOpenAI | None = None,
        anthropic_client: AsyncAnthropic | None = None,
    ) -> None:
        """
        Args:
            openai_api_key / anthropic_api_key: Explicit keys; fall back to
                OPENAI_API_KEY / ANTHROPIC_API_KEY environment variables.
            timeout: Per-request timeout in seconds.
            max_retries: SDK-level retries with backoff for 429 / 5xx / network errors.
            anthropic_max_tokens: Required by the Anthropic Messages API.
            openai_client / anthropic_client: Inject pre-built clients (testing, proxies).
                Injected clients are not closed by ``aclose()``.
        """
        if anthropic_max_tokens <= 0:
            raise ValueError("anthropic_max_tokens must be positive")

        self._openai_key = _to_secret(openai_api_key, "OPENAI_API_KEY")
        self._anthropic_key = _to_secret(anthropic_api_key, "ANTHROPIC_API_KEY")
        self._timeout = timeout
        self._max_retries = max_retries
        self._anthropic_max_tokens = anthropic_max_tokens

        self._openai: AsyncOpenAI | None = openai_client
        self._anthropic: AsyncAnthropic | None = anthropic_client
        self._owns_openai = openai_client is None
        self._owns_anthropic = anthropic_client is None

    def __repr__(self) -> str:  # never expose keys
        return (
            f"{type(self).__name__}(openai={'configured' if self._openai or self._openai_key else 'missing'}, "
            f"anthropic={'configured' if self._anthropic or self._anthropic_key else 'missing'})"
        )

    async def __aenter__(self) -> "UnifiedLLMGateway":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._openai is not None and self._owns_openai:
            await self._openai.close()
            self._openai = None
        if self._anthropic is not None and self._owns_anthropic:
            await self._anthropic.close()
            self._anthropic = None

    # ------------------------------------------------------------------ public

    async def stream(
        self, prompt: str, model: str, temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion as normalized ``StreamChunk`` objects.

        Yields zero or more text chunks (``is_final=False``) followed by exactly
        one final chunk (``is_final=True``) carrying ``ChunkTelemetry``.

        Raises:
            InvalidRequestError, UnsupportedModelError, GatewayProviderError (and subclasses).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("prompt must be a non-empty string")

        try:
            resolved = resolve_model(model)
        except KeyError:
            raise UnsupportedModelError(
                f"Model {model!r} is not supported. Supported: {', '.join(supported_models())}"
            ) from None

        low, high = _TEMPERATURE_RANGE[resolved.provider]
        if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and low <= temperature <= high):
            raise InvalidRequestError(
                f"temperature must be between {low} and {high} for provider {resolved.provider!r}"
            )

        if resolved.provider == "openai":
            inner = self._stream_openai(prompt, resolved, float(temperature))
        else:
            inner = self._stream_anthropic(prompt, resolved, float(temperature))

        async with aclosing(inner) as chunks:
            async for chunk in chunks:
                yield chunk

    # ------------------------------------------------------------- providers

    async def _stream_openai(
        self, prompt: str, resolved: ResolvedModel, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        prompt_estimate = await _estimate_openai_prompt_tokens(prompt, resolved.api_model)
        reported_model = resolved.api_model
        usage: Any = None
        finish_reason: Optional[str] = None
        output_parts: list[str] = []
        first_token_at: Optional[float] = None
        started = time.perf_counter()

        async with self._provider_errors("openai", resolved.api_model):
            client = self._get_openai_client()
            stream = await client.chat.completions.create(
                model=resolved.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            try:
                async for event in stream:
                    if getattr(event, "model", None):
                        reported_model = event.model
                    if getattr(event, "usage", None) is not None:
                        usage = event.usage  # arrives on the last event, with empty choices
                    if not event.choices:
                        continue
                    choice = event.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    text = choice.delta.content if choice.delta is not None else None
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_parts.append(text)
                        yield StreamChunk(delta_text=text, provider="openai", model=reported_model)
            finally:
                await stream.close()

        ended = time.perf_counter()

        source: UsageSource
        if usage is not None:
            input_tokens = int(usage.prompt_tokens or 0)
            output_tokens = int(usage.completion_tokens or 0)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0)
            source = "provider"
            self._verify_prompt_estimate(resolved.api_model, input_tokens, prompt_estimate)
        else:
            logger.warning("OpenAI stream ended without usage; falling back to tiktoken estimates")
            input_tokens = prompt_estimate or 0
            try:
                output_tokens = await asyncio.to_thread(
                    _count_text_tokens_sync, "".join(output_parts), resolved.api_model
                )
            except Exception:
                output_tokens = 0
            cached = 0
            source = "estimated"

        yield self._final_chunk(
            provider="openai",
            model=reported_model,
            resolved=resolved,
            started=started,
            first_token_at=first_token_at,
            ended=ended,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            cache_write_tokens=0,
            uncached_input_tokens=max(input_tokens - cached, 0),
            usage_source=source,
            prompt_estimate=prompt_estimate,
            finish_reason=finish_reason,
        )

    async def _stream_anthropic(
        self, prompt: str, resolved: ResolvedModel, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        reported_model = resolved.api_model
        input_tokens = cache_read = cache_write = output_tokens = 0
        saw_message_start = False
        stop_reason: Optional[str] = None
        first_token_at: Optional[float] = None
        started = time.perf_counter()

        async with self._provider_errors("anthropic", resolved.api_model):
            client = self._get_anthropic_client()
            request: dict[str, Any] = {
                "model": resolved.api_model,
                "max_tokens": self._anthropic_max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            # Newer Claude models (5.x) reject `temperature` with a 400; only send it when supported.
            if resolved.supports_temperature:
                request["extra_body"] = {"temperature": temperature}
            stream = await client.messages.create(**request)
            try:
                async for event in stream:
                    etype = event.type
                    if etype == "message_start":
                        saw_message_start = True
                        message = event.message
                        reported_model = message.model or reported_model
                        u = message.usage
                        input_tokens = int(u.input_tokens or 0)
                        cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
                        cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
                        output_tokens = int(getattr(u, "output_tokens", 0) or 0)
                    elif etype == "content_block_delta":
                        delta = event.delta
                        if getattr(delta, "type", None) == "text_delta" and delta.text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            yield StreamChunk(delta_text=delta.text, provider="anthropic", model=reported_model)
                    elif etype == "message_delta":
                        # usage.output_tokens on message_delta is cumulative, not incremental.
                        if event.usage is not None:
                            output_tokens = max(output_tokens, int(event.usage.output_tokens or 0))
                            late_input = getattr(event.usage, "input_tokens", None)
                            if late_input:
                                input_tokens = int(late_input)
                        if event.delta.stop_reason:
                            stop_reason = event.delta.stop_reason
                    # ping / content_block_start / content_block_stop / message_stop: nothing to emit
            finally:
                await stream.close()

        ended = time.perf_counter()

        # Anthropic's input_tokens excludes cache reads/writes; report the full prompt size.
        yield self._final_chunk(
            provider="anthropic",
            model=reported_model,
            resolved=resolved,
            started=started,
            first_token_at=first_token_at,
            ended=ended,
            input_tokens=input_tokens + cache_read + cache_write,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            uncached_input_tokens=input_tokens,
            usage_source="provider" if saw_message_start else "estimated",
            prompt_estimate=None,
            finish_reason=stop_reason,
        )

    # ---------------------------------------------------------------- helpers

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai is None:
            if not self._openai_key.get_secret_value().startswith("sk-"):
                raise AuthenticationError("OPENAI_API_KEY looks malformed (expected prefix 'sk-').", provider="openai")
            if self._openai_key is None:
                raise AuthenticationError("No OpenAI API key configured (set OPENAI_API_KEY).", provider="openai")
            self._openai = AsyncOpenAI(
                api_key=self._openai_key.get_secret_value(),
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._openai

    def _get_anthropic_client(self) -> AsyncAnthropic:
        if self._anthropic is None:
            if not self._anthropic_key.get_secret_value().startswith("sk-ant-"):
                raise AuthenticationError("ANTHROPIC_API_KEY looks malformed (expected prefix 'sk-ant-').", provider="anthropic")
            if self._anthropic_key is None:
                raise AuthenticationError("No Anthropic API key configured (set ANTHROPIC_API_KEY).", provider="anthropic")
            self._anthropic = AsyncAnthropic(
                api_key=self._anthropic_key.get_secret_value(),
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._anthropic

    def _secrets(self) -> tuple[str, ...]:
        return tuple(k.get_secret_value() for k in (self._openai_key, self._anthropic_key) if k is not None)

    @staticmethod
    def _verify_prompt_estimate(model: str, actual: int, estimate: Optional[int]) -> None:
        if estimate is None:
            return
        if abs(actual - estimate) > max(4, int(actual * _ESTIMATE_TOLERANCE)):
            logger.warning(
                "Prompt token mismatch for %s: provider=%d tiktoken=%d (billing uses provider count)",
                model, actual, estimate,
            )

    @staticmethod
    def _final_chunk(
        *,
        provider: Provider,
        model: str,
        resolved: ResolvedModel,
        started: float,
        first_token_at: Optional[float],
        ended: float,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        uncached_input_tokens: int,
        usage_source: UsageSource,
        prompt_estimate: Optional[int],
        finish_reason: Optional[str],
    ) -> StreamChunk:
        latency = max(ended - started, 0.0)
        ttft = (first_token_at - started) if first_token_at is not None else None
        decode_window = (ended - first_token_at) if first_token_at is not None else 0.0
        tps = output_tokens / decode_window if decode_window > 0 else 0.0

        cost = calculate_cost(
            resolved.pricing,
            uncached_input_tokens=uncached_input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

        telemetry = ChunkTelemetry(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            latency_s=round(latency, 4),
            ttft_ms=round(ttft * 1000, 1) if ttft is not None else None, 
            tokens_per_second=round(tps, 2),
            total_cost_usd=float(cost),
            usage_source=usage_source,
            prompt_tokens_estimate=prompt_estimate,
            finish_reason=finish_reason,
        )
        logger.info(
            "llm_request_complete provider=%s model=%s in=%d out=%d ttft=%s latency=%.3fs cost=$%s",
            provider, model, input_tokens, output_tokens,
            f"{ttft:.3f}s" if ttft is not None else "n/a", latency, cost,
        )
        return StreamChunk(delta_text="", provider=provider, model=model, is_final=True, telemetry=telemetry)

    @asynccontextmanager
    async def _provider_errors(self, provider: Provider, model: str) -> AsyncIterator[None]:
        """Translate SDK/transport exceptions into domain exceptions, without leaking secrets."""
        sdk: Any = openai if provider == "openai" else anthropic
        try:
            yield
        except GatewayError:
            raise
        except sdk.APIError as exc:
            err = _translate_sdk_error(exc, provider, sdk, self._secrets())
            logger.warning(
                "llm_provider_error provider=%s model=%s type=%s status=%s request_id=%s msg=%s",
                provider, model, type(err).__name__, err.status_code, err.request_id, err.message,
            )
            raise err from None
        except _TIMEOUT_ERRORS:
            logger.warning("llm_provider_timeout provider=%s model=%s", provider, model)
            raise ProviderTimeoutError("Timed out while streaming from provider.", provider=provider, retryable=True) from None
        except _TRANSPORT_ERRORS as exc:
            logger.warning("llm_provider_connection_error provider=%s model=%s type=%s", provider, model, type(exc).__name__)
            raise ProviderConnectionError("Connection to provider failed mid-stream.", provider=provider, retryable=True) from None
        except Exception as exc:
            # Last line of defence: no raw stack traces or messages escape.
            logger.error(
                "llm_gateway_unexpected_error provider=%s model=%s type=%s msg=%s",
                provider, model, type(exc).__name__, _redact(str(exc), self._secrets()),
            )
            raise GatewayProviderError("Unexpected error while streaming.", provider=provider) from None


def _retry_after_seconds(exc: Any) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after-ms")
    if raw:
        try:
            return float(raw) / 1000.0
        except ValueError:
            pass
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return None  # HTTP-date form; let the caller use its own backoff
    return None


def _translate_sdk_error(exc: Any, provider: Provider, sdk: Any, secrets: tuple[str, ...]) -> GatewayProviderError:
    status: Optional[int] = getattr(exc, "status_code", None)
    request_id: Optional[str] = getattr(exc, "request_id", None)
    safe_msg = _redact(str(getattr(exc, "message", "") or type(exc).__name__), secrets)
    common = {"provider": provider, "status_code": status, "request_id": request_id}

    if isinstance(exc, sdk.RateLimitError):
        return RateLimitExceededError(
            "Provider rate limit exceeded.", retry_after=_retry_after_seconds(exc), **common
        )
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        # Deliberately do not echo the provider message: it may include a key fragment.
        return AuthenticationError("Authentication with provider failed; check the API key and its permissions.", **common)
    if isinstance(exc, sdk.APITimeoutError):
        return ProviderTimeoutError("Provider request timed out.", retryable=True, **common)
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderConnectionError("Could not connect to provider.", retryable=True, **common)
    if isinstance(exc, (sdk.BadRequestError, sdk.NotFoundError, sdk.UnprocessableEntityError)):
        return ProviderBadRequestError(f"Provider rejected the request: {safe_msg}", **common)
    if isinstance(exc, sdk.APIStatusError):
        retryable = status is not None and (status >= 500 or status == 529)
        return GatewayProviderError(f"Provider returned an error: {safe_msg}", retryable=retryable, **common)
    return GatewayProviderError(f"Provider stream error: {safe_msg}", **common)


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  

    async def _demo() -> None:
        logging.basicConfig(level=logging.INFO)
        model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
        async with UnifiedLLMGateway() as gw:
            try:
                async for chunk in gw.stream("Explain TCP slow start in two sentences.", model=model):
                    if chunk.is_final:
                        print("\n\n" + chunk.telemetry.model_dump_json(indent=2))  # type: ignore[union-attr]
                    else:
                        print(chunk.delta_text, end="", flush=True)
            except RateLimitExceededError as e:
                print(f"Rate limited; retry after {e.retry_after}s")
            except GatewayError as e:
                print(f"Gateway error: {e}")

    asyncio.run(_demo())
