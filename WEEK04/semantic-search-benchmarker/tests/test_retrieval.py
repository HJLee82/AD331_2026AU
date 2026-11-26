"""Tests 3–4: retrieval behavior."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.benchmarker import load_corpus, load_eval_queries, matched_anchors
from src.chunking import DocumentChunk, StructureAwareChunker, chunk_corpus
from src.indexer import BM25Retriever, DenseRetriever, EmbeddingCache, OpenAIEmbedder, bm25_tokenize

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _chunk(i: int, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"c{i}", content=text, parent_doc="settings.md",
                         metadata={"section_anchors": [], "anchor_path": []})


def test_bm25_ranks_rare_identifier_first():
    """Test 3: a rare exact identifier is retrieved at rank 1 by BM25."""
    chunks = [
        _chunk(0, "Configure the server settings and environment for deployment."),
        _chunk(1, "Set CUSTOM_CONFIG_FLAG_XYZ to true to enable the experimental cache."),
        _chunk(2, "The config file holds custom settings such as flags and timeouts."),
        _chunk(3, "Use environment variables to override default configuration flags."),
    ]
    bm25 = BM25Retriever()
    bm25.index(chunks)
    results = bm25.search("What does CUSTOM_CONFIG_FLAG_XYZ do?", k=3)
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY", "").startswith("sk-"),
                    reason="needs OPENAI_API_KEY for real embeddings")
def test_dense_finds_paraphrase_with_no_shared_keywords():
    """Test 4: a conceptual query that shares no meaningful words with the
    answer section is still found in the top-5 by dense search."""
    docs = load_corpus(ROOT / "corpus")
    chunks = chunk_corpus(StructureAwareChunker(), docs)
    q = next(q for q in load_eval_queries(ROOT / "data" / "eval_queries.json") if q.query_id == "q12")

    gold_text = " ".join(
        c.content for c in chunks
        if c.parent_doc == q.source_file and set(c.metadata["section_anchors"]) & set(q.relevant_sections)
    )
    content_words = {t for t in bm25_tokenize(q.query) if len(t) > 3}
    assert not content_words & set(bm25_tokenize(gold_text)), "query is not a true paraphrase"

    dense = DenseRetriever("test_dense_paraphrase",
                           OpenAIEmbedder(cache=EmbeddingCache(ROOT / ".cache" / "embeddings.sqlite")))
    dense.index(chunks)
    results = dense.search(q.query, k=5)
    assert any(matched_anchors(r, q) for r in results)
