"""Exercises the real ChromaDB code path with a deterministic offline embedding."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from src.config import Settings  # noqa: E402
from src.retriever import ChromaRetriever  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


class HashEmbedding(chromadb.EmbeddingFunction):
    """Bag-of-words hashing embedding: deterministic, offline, good enough for tests."""

    def __init__(self) -> None:
        pass

    def __call__(self, input):  # noqa: A002 - Chroma's required signature
        out = []
        for text in input:
            vec = [0.0] * 256
            for tok in text.lower().split():
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
            out.append(vec)
        return out

    @staticmethod
    def name() -> str:
        return "test-hash"


@pytest.fixture()
def retriever(tmp_path):
    settings = Settings(
        openai_api_key="test", openai_model="test", docs_dir=DOCS, chroma_dir=tmp_path
    )
    client = chromadb.PersistentClient(path=str(tmp_path))
    return ChromaRetriever(settings, embedding_function=HashEmbedding(), client=client)


def test_index_is_idempotent_and_retrieves_k1(retriever):
    n = retriever.index()
    assert n > 30
    assert retriever.index() == n  # fingerprint hit, no rebuild
    results = retriever.retrieve("worker node system requirements cpu ram disk")
    assert len(results) == 20
    assert all(r.source == "internal_doc" for r in results)


def test_identifier_hits_are_guaranteed_in_candidates(retriever):
    retriever.index()
    results = retriever.retrieve("what should I do about nim-e6602")
    assert any("NIM-E6602" in r.content for r in results)
