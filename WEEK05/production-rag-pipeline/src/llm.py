"""Thin, typed wrapper around the OpenAI Chat Completions API."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unusable output."""


@runtime_checkable
class ChatLLM(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


class OpenAIChatLLM:
    """OpenAI-backed ``ChatLLM``. Retries/timeouts are delegated to the SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        self._model = model
        self._temperature = temperature

    def complete(self, *, system: str, user: str) -> str:
        from openai import OpenAIError

        kwargs: dict[str, object] = {}
        if self._temperature is not None:
            # Reasoning-class models reject `temperature`; only send it if configured.
            kwargs["temperature"] = self._temperature
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,  # type: ignore[arg-type]
            )
        except OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        if not response.choices:
            raise LLMError("OpenAI returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("OpenAI returned an empty message")
        return content.strip()
