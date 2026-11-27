"""Offline unit tests for rewriter, reranker, web fallback, generator and routing."""

from __future__ import annotations

import numpy as np
import pytest

from src.generator import INSUFFICIENT_ANSWER, GroundedGenerator, normalize_citations
from src.llm import LLMError
from src.pipeline import TwoStageRAGPipeline
from src.reranker import CrossEncoderReranker
from src.rewriter import QueryRewriter
from src.schemas import RerankStatus, RetrievedChunk
from src.text_utils import extract_identifiers
from src.web_fallback import WebFallback, WebResult


# ----------------------------------------------------------------- fakes
class FakeLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply, self.calls = reply, []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeCrossEncoder:
    """Logit = +4 if the keyword appears in the passage, else -4."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.array([4.0 if self.keyword in p else -4.0 for _, p in pairs])


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=None):
        return list(self.chunks)


class FakeProvider:
    name = "fake"

    def __init__(self, results):
        self.results = results

    def search(self, query, max_results):
        return self.results


def chunk(cid: str, text: str, source="internal_doc") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, content=text, score=0.5, source=source)


# --------------------------------------------------------------- tests
def test_extract_identifiers():
    ids = extract_identifiers("What is NIM-E4021 in tf-nimbus-v1 at p99? Also 10GB and the 3rd step.")
    assert ids == ["NIM-E4021", "tf-nimbus-v1", "p99"]


def test_rewriter_reattaches_dropped_identifier_and_cleans_output():
    llm = FakeLLM('Query: "canary health divergence meaning"')
    out = QueryRewriter(llm).rewrite("hey, what does NIM-E4021 mean?")
    assert out == "canary health divergence meaning NIM-E4021"


def test_rewriter_falls_back_on_llm_error():
    out = QueryRewriter(FakeLLM(LLMError("boom"))).rewrite("  refund   policy ")
    assert out == "refund policy"


def test_reranker_threshold_routing():
    rr = CrossEncoderReranker(model=FakeCrossEncoder("refund"), threshold=0.55, top_k=3)
    cands = [chunk(f"c{i}", "refund rules" if i < 4 else "other") for i in range(20)]
    ok = rr.rerank("refund", cands)
    assert ok.status is RerankStatus.OK and len(ok.chunks) == 3
    assert ok.top_confidence == pytest.approx(1 / (1 + np.exp(-4)))

    low = rr.rerank("refund", [chunk("c", "nothing relevant")])
    assert low.status is RerankStatus.LOW_CONFIDENCE and low.chunks == []
    assert rr.rerank("q", []).status is RerankStatus.NO_CANDIDATES


def test_web_fallback_filters_and_formats():
    provider = FakeProvider([
        WebResult("A", "https://a.example/x", "  snippet  a "),
        WebResult("dup", "https://a.example/x", "dup"),
        WebResult("bad", "javascript:alert(1)", "bad"),
        WebResult("empty", "https://e.example", ""),
        WebResult("B", "https://b.example", "snippet b"),
    ])
    chunks = WebFallback(provider, top_k=3).search("q")
    assert [c.chunk_id for c in chunks] == ["https://a.example/x", "https://b.example"]
    assert all(c.source == "web" for c in chunks)
    assert "snippet a" in chunks[0].content


def test_normalize_citations_skips_code_and_invalid():
    text, cited = normalize_citations("Use `arr[0]` here [1, 3] and [9].\n```\nx[2]\n```", 2)
    assert cited == [1]
    assert "`arr[0]`" in text and "x[2]" in text and "[9]" not in text


def test_generator_strict_mode_rejects_uncited_answer():
    gen = GroundedGenerator(FakeLLM("The answer is 42."), strict_citations=True)
    resp = gen.generate(query="q", transformed_query="q", chunks=[chunk("a", "x")],
                        route="INTERNAL_RAG", confidence=0.9)
    assert resp.answer == INSUFFICIENT_ANSWER and resp.citations == []


def test_pipeline_internal_and_fallback_routes():
    internal = [chunk("billing_policy.md::refund::001", "refund within 7 days")]
    web_results = [WebResult("W", "https://w.example", "general refund info")]

    def build(keyword: str) -> TwoStageRAGPipeline:
        return TwoStageRAGPipeline(
            rewriter=QueryRewriter(FakeLLM("refund policy")),
            retriever=FakeRetriever(internal),
            reranker=CrossEncoderReranker(model=FakeCrossEncoder(keyword), threshold=0.55),
            web=WebFallback(FakeProvider(web_results)),
            generator=GroundedGenerator(FakeLLM("Refunds are allowed within 7 days [1].")),
        )

    resp = build("7 days").run("can I get a refund?")
    assert resp.route_taken == "INTERNAL_RAG"
    assert resp.citations == ["[1] billing_policy.md::refund::001"]
    assert resp.transformed_query == "refund policy"

    resp = build("general").run("can I get a refund?")
    assert resp.route_taken == "WEB_FALLBACK"
    assert resp.citations == ["[1] https://w.example"]
    # Fallback route reports the internal score that triggered the fallback.
    assert resp.confidence_score < 0.55
