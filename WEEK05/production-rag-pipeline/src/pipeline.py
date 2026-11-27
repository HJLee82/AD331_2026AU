"""Orchestrator: rewrite -> Stage 1 -> Stage 2 -> (internal | web) -> generate."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .generator import GroundedGenerator
from .schemas import ChatTurn, RAGResponse, RerankResult, RerankStatus, RetrievedChunk, RouteType

logger = logging.getLogger(__name__)

EMPTY_QUERY_ANSWER = "Please enter a question."


# Structural interfaces so every stage can be swapped or faked in tests.
class Rewriter(Protocol):
    def rewrite(self, query: str, history: Sequence[ChatTurn] | None = None) -> str: ...


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> RerankResult: ...
    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]: ...


class WebSearcher(Protocol):
    def search(self, query: str) -> list[RetrievedChunk]: ...


@dataclass(slots=True)
class PipelineTrace:
    original_query: str
    transformed_query: str = ""
    stage1_candidates: int = 0
    rerank_status: RerankStatus | None = None
    top_internal_confidence: float = 0.0
    web_results: int = 0
    top_web_relevance: float = 0.0
    route: RouteType | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Full lists for the UI visualizer.
    stage1_chunks: list[RetrievedChunk] = field(default_factory=list)  # score = cosine similarity
    stage2_ranked: list[RetrievedChunk] = field(default_factory=list)  # score = sigmoid confidence
    web_chunks: list[RetrievedChunk] = field(default_factory=list)
    context_chunks: list[RetrievedChunk] = field(default_factory=list)  # what the generator saw

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[stage] = round((time.perf_counter() - start) * 1000, 1)


class TwoStageRAGPipeline:
    def __init__(
        self,
        *,
        rewriter: Rewriter,
        retriever: Retriever,
        reranker: Reranker,
        web: WebSearcher,
        generator: GroundedGenerator,
    ) -> None:
        self._rewriter = rewriter
        self._retriever = retriever
        self._reranker = reranker
        self._web = web
        self._generator = generator

    @classmethod
    def from_settings(
        cls, settings: Settings, *, retriever: Retriever | None = None
    ) -> "TwoStageRAGPipeline":
        # Local imports keep heavy deps out of module import time.
        from .llm import OpenAIChatLLM
        from .reranker import CrossEncoderReranker
        from .retriever import ChromaRetriever
        from .rewriter import QueryRewriter
        from .web_fallback import WebFallback, build_web_provider

        llm = OpenAIChatLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            timeout_s=settings.openai_timeout_s,
            max_retries=settings.openai_max_retries,
        )
        return cls(
            rewriter=QueryRewriter(llm),
            retriever=retriever or ChromaRetriever(settings),
            reranker=CrossEncoderReranker(
                settings.reranker_model,
                threshold=settings.confidence_threshold,
                top_k=settings.rerank_top_k,
            ),
            web=WebFallback(build_web_provider(settings), top_k=settings.web_top_k),
            generator=GroundedGenerator(
                llm,
                max_context_chars=settings.max_context_chars,
                strict_citations=settings.strict_citations,
            ),
        )

    def run(self, query: str, history: Sequence[ChatTurn] | None = None) -> RAGResponse:
        return self.run_with_trace(query, history)[0]

    def run_with_trace(
        self, query: str, history: Sequence[ChatTurn] | None = None
    ) -> tuple[RAGResponse, PipelineTrace]:
        trace = PipelineTrace(original_query=query)

        # Empty or whitespace-only input: answer immediately, no LLM / search calls.
        if not (query or "").strip():
            trace.route = "INTERNAL_RAG"
            return (
                RAGResponse(
                    answer=EMPTY_QUERY_ANSWER,
                    citations=[],
                    route_taken="INTERNAL_RAG",
                    confidence_score=0.0,
                    transformed_query="",
                ),
                trace,
            )

        with trace.timed("rewrite"):
            transformed = self._rewriter.rewrite(query, history)
        trace.transformed_query = transformed

        with trace.timed("stage1_retrieve"):
            candidates = self._retriever.retrieve(transformed)
        trace.stage1_candidates = len(candidates)
        trace.stage1_chunks = list(candidates)

        with trace.timed("stage2_rerank"):
            result = self._reranker.rerank(transformed, candidates)
        trace.rerank_status = result.status
        trace.top_internal_confidence = result.top_confidence
        trace.stage2_ranked = list(result.ranked)

        if not result.needs_fallback:
            route: RouteType = "INTERNAL_RAG"
            chunks, confidence = result.chunks, result.top_confidence
        else:
            route = "WEB_FALLBACK"
            logger.info("Status %s -> web fallback", result.status.value)
            with trace.timed("web_fallback"):
                web_chunks = self._web.search(transformed)
                # Re-score web snippets with the same cross-encoder so the reported
                # confidence is on the same scale as the internal route.
                chunks = self._reranker.score(transformed, web_chunks)
            trace.web_results = len(chunks)
            trace.web_chunks = list(chunks)
            trace.top_web_relevance = chunks[0].score if chunks else 0.0
            # Report the internal confidence that triggered the fallback, so the
            # value shown next to the 0.55 threshold matches the route taken.
            confidence = result.top_confidence
        trace.route = route
        trace.context_chunks = list(chunks)

        with trace.timed("generate"):
            response = self._generator.generate(
                query=query,
                transformed_query=transformed,
                chunks=chunks,
                route=route,
                confidence=confidence,
            )

        logger.info(
            "route=%s confidence=%.3f timings=%s", route, response.confidence_score, trace.timings_ms
        )
        return response, trace
