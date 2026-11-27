"""Web fallback: DuckDuckGo (``ddgs``) or Tavily -> ``RetrievedChunk(source="web")``."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from .config import Settings
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WebResult:
    title: str
    url: str
    snippet: str
    score: float | None = None


class WebSearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[WebResult]: ...


class DDGSProvider:
    name = "ddgs"

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    def search(self, query: str, max_results: int) -> list[WebResult]:
        from ddgs import DDGS

        rows: list[dict[str, Any]] = DDGS(timeout=int(self._timeout_s)).text(
            query, max_results=max_results
        ) or []
        return [
            WebResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
            for r in rows
        ]


class TavilyProvider:
    name = "tavily"

    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int) -> list[WebResult]:
        payload = self._client.search(query=query, max_results=max_results, search_depth="basic")
        return [
            WebResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=r.get("score"),
            )
            for r in payload.get("results", [])
        ]


def build_web_provider(settings: Settings) -> WebSearchProvider:
    if settings.web_provider == "tavily":
        assert settings.tavily_api_key  # validated in Settings
        return TavilyProvider(settings.tavily_api_key)
    return DDGSProvider(timeout_s=settings.web_timeout_s)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class WebFallback:
    def __init__(
        self,
        provider: WebSearchProvider,
        *,
        top_k: int = 3,
        max_retries: int = 2,
        backoff_s: float = 1.0,
        max_snippet_chars: int = 1500,
    ) -> None:
        self._provider = provider
        self._top_k = top_k
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._max_snippet_chars = max_snippet_chars

    def _search_with_retry(self, query: str) -> list[WebResult]:
        # Over-fetch slightly so filtering/dedup still leaves top_k results.
        for attempt in range(self._max_retries + 1):
            try:
                return self._provider.search(query, max_results=self._top_k + 2)
            except Exception:  # noqa: BLE001 - provider SDKs raise heterogeneous errors
                if attempt == self._max_retries:
                    logger.error("Web search (%s) failed", self._provider.name, exc_info=True)
                    return []
                delay = self._backoff_s * (2**attempt)
                logger.warning("Web search attempt %d failed; retrying in %.1fs", attempt + 1, delay)
                time.sleep(delay)
        return []

    def search(self, query: str) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for rank, result in enumerate(self._search_with_retry(query)):
            url, title = result.url.strip(), _clean(result.title)
            snippet = _clean(result.snippet)[: self._max_snippet_chars]
            if not snippet or not _valid_url(url) or url in seen:
                continue
            seen.add(url)
            score = result.score if result.score is not None else 1.0 / (rank + 1)
            chunks.append(
                RetrievedChunk(
                    chunk_id=url,
                    content=f"Title: {title or url}\nURL: {url}\n\n{snippet}",
                    score=float(score),
                    source="web",
                )
            )
            if len(chunks) >= self._top_k:
                break
        logger.info("Web fallback (%s): %d snippets", self._provider.name, len(chunks))
        return chunks
