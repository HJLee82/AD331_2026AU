"""Environment-driven configuration (12-factor). All tunables live here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

EmbeddingProvider = Literal["local", "openai"]
WebProvider = Literal["ddgs", "tavily"]

_DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "local": "all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
}


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float | None) -> float | None:
    raw = _env(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    # --- LLM -----------------------------------------------------------------
    openai_api_key: str
    openai_model: str
    openai_temperature: float | None = None  # None => omit (some models reject it)
    openai_timeout_s: float = 30.0
    openai_max_retries: int = 3

    # --- Corpus & index ------------------------------------------------------
    docs_dir: Path = field(default_factory=lambda: Path("docs"))
    chroma_dir: Path = field(default_factory=lambda: Path(".chroma"))
    collection_prefix: str = "nimbus_docs"
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = _DEFAULT_EMBEDDING_MODELS["local"]
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    # --- Stage 1 / Stage 2 ---------------------------------------------------
    retrieval_top_k: int = 20  # K1
    lexical_boost_k: int = 5  # reserved Stage-1 slots for exact identifier hits
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3  # K2
    confidence_threshold: float = 0.55

    # --- Web fallback --------------------------------------------------------
    web_provider: WebProvider = "ddgs"
    tavily_api_key: str | None = None
    web_top_k: int = 3
    web_timeout_s: float = 10.0

    # --- Generation ----------------------------------------------------------
    max_context_chars: int = 12_000
    strict_citations: bool = True

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required")
        if not self.openai_model:
            raise ConfigError("OPENAI_MODEL is required")
        if self.embedding_provider not in _DEFAULT_EMBEDDING_MODELS:
            raise ConfigError(f"EMBEDDING_PROVIDER must be one of {list(_DEFAULT_EMBEDDING_MODELS)}")
        if self.web_provider not in ("ddgs", "tavily"):
            raise ConfigError("WEB_SEARCH_PROVIDER must be 'ddgs' or 'tavily'")
        if self.web_provider == "tavily" and not self.tavily_api_key:
            raise ConfigError("TAVILY_API_KEY is required when WEB_SEARCH_PROVIDER=tavily")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ConfigError("CONFIDENCE_THRESHOLD must be in (0, 1)")
        if not 1 <= self.rerank_top_k <= self.retrieval_top_k:
            raise ConfigError("Require 1 <= RERANK_TOP_K <= RETRIEVAL_TOP_K")
        if self.lexical_boost_k < 0 or self.lexical_boost_k > self.retrieval_top_k:
            raise ConfigError("Require 0 <= LEXICAL_BOOST_K <= RETRIEVAL_TOP_K")
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ConfigError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_MAX_CHARS")

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        if env_file is not None:
            load_dotenv(env_file, override=False)

        provider = (_env("EMBEDDING_PROVIDER", "local") or "local").lower()
        return cls(
            openai_api_key=_env("OPENAI_API_KEY", "") or "",
            openai_model=_env("OPENAI_MODEL", "") or "",
            openai_temperature=_env_float("OPENAI_TEMPERATURE", None),
            openai_timeout_s=_env_float("OPENAI_TIMEOUT_S", 30.0) or 30.0,
            openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
            docs_dir=Path(_env("DOCS_DIR", "docs") or "docs"),
            chroma_dir=Path(_env("CHROMA_DIR", ".chroma") or ".chroma"),
            collection_prefix=_env("COLLECTION_PREFIX", "nimbus_docs") or "nimbus_docs",
            embedding_provider=provider,  # type: ignore[arg-type]
            embedding_model=_env("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODELS.get(provider, ""))
            or "",
            chunk_max_chars=_env_int("CHUNK_MAX_CHARS", 1200),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 200),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 20),
            lexical_boost_k=_env_int("LEXICAL_BOOST_K", 5),
            reranker_model=_env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2") or "",
            rerank_top_k=_env_int("RERANK_TOP_K", 3),
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.55) or 0.55,
            web_provider=(_env("WEB_SEARCH_PROVIDER", "ddgs") or "ddgs").lower(),  # type: ignore[arg-type]
            tavily_api_key=_env("TAVILY_API_KEY"),
            web_top_k=_env_int("WEB_TOP_K", 3),
            web_timeout_s=_env_float("WEB_TIMEOUT_S", 10.0) or 10.0,
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 12_000),
            strict_citations=_env_bool("STRICT_CITATIONS", True),
        )
