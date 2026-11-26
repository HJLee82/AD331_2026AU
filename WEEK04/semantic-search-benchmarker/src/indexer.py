"""
Dual indexing: dense vectors (OpenAI + ChromaDB) and sparse keywords (BM25).

Both retrievers implement the :class:`Retriever` protocol and return the same
:class:`SearchResult` model, so the benchmark (or a production hybrid/rerank
layer) can treat them interchangeably.

Design notes
------------
* Embeddings are computed client-side and handed to Chroma explicitly
  (no Chroma embedding function). That keeps the embedding model swappable
  via the :class:`Embedder` protocol and lets us L2-normalize ourselves.
* Chroma metadata only accepts scalars, so each chunk's full metadata is
  stored as one JSON string and restored on read.
* Document embeddings can be cached in SQLite (keyed by model + content hash)
  so re-running benchmarks doesn't re-bill the same chunks. Query embeddings
  are never cached — that would make latency numbers meaningless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import chromadb
import numpy as np
import tiktoken
from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi

from .chunking import DocumentChunk, get_encoding

__all__ = [
    "SearchResult",
    "Retriever",
    "Embedder",
    "OpenAIEmbedder",
    "EmbeddingCache",
    "DenseRetriever",
    "BM25Retriever",
    "bm25_tokenize",
    "l2_normalize",
]

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MAX_INPUT_TOKENS = 8191


# --------------------------------------------------------------------------- #
# Shared search interface
# --------------------------------------------------------------------------- #
class SearchResult(BaseModel):
    """One ranked hit, identical in shape for every retriever.

    ``score`` is retriever-specific and only comparable *within* one result
    list (cosine similarity for dense, raw BM25 score for sparse).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    parent_doc: str
    content: str
    score: float
    rank: int  # 1-indexed
    metadata: dict[str, Any]


@runtime_checkable
class Retriever(Protocol):
    """Index once, then answer top-k queries."""

    name: str

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        ...

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        ...


def _to_result(chunk: DocumentChunk, score: float, rank: int) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id,
        parent_doc=chunk.parent_doc,
        content=chunk.content,
        score=float(score),
        rank=rank,
        metadata=chunk.metadata,
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm (zero rows are left as zeros)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


@runtime_checkable
class Embedder(Protocol):
    """Text → unit-length vectors. ``model`` participates in cache keys."""

    model: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


class EmbeddingCache:
    """Tiny SQLite store for document embeddings keyed by (model, sha256)."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 900):  # stay under SQLite's variable limit
            batch = keys[i : i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            for k, blob in self._db.execute(q, batch):
                out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self._db.commit()


class OpenAIEmbedder:
    """OpenAI embeddings with batching, truncation, retries and L2 norm.

    Args:
        model: Embedding model name.
        batch_size: Inputs per API request (API max is 2048; smaller batches
            keep requests well under the per-request token cap).
        cache: Optional :class:`EmbeddingCache` for document embeddings.
        client: Optional pre-configured ``openai.OpenAI`` client.
        max_retries: Passed to the OpenAI client (exponential backoff on
            429/5xx is handled by the SDK).
        encoding: tiktoken encoding used for input truncation.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 128,
        cache: EmbeddingCache | None = None,
        client: Any | None = None,
        max_retries: int = 5,
        encoding: tiktoken.Encoding | None = None,
    ) -> None:
        from openai import OpenAI  # local import: module usable without openai for BM25-only runs

        self.model = model
        self.batch_size = batch_size
        self.cache = cache
        self._client = client or OpenAI(max_retries=max_retries)  # reads OPENAI_API_KEY
        self._enc = encoding or get_encoding()

    def _truncate(self, text: str) -> str:
        """Clip to the model's input limit instead of failing the batch."""
        toks = self._enc.encode(text, disallowed_special=())
        if len(toks) <= EMBEDDING_MAX_INPUT_TOKENS:
            return text
        log.warning("Truncating input from %d to %d tokens", len(toks), EMBEDDING_MAX_INPUT_TOKENS)
        return self._enc.decode(toks[:EMBEDDING_MAX_INPUT_TOKENS])

    def _call_api(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self._truncate(t) or " " for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            # The API may return items out of order; sort by index to be safe.
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return l2_normalize(np.array(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts, serving from cache where possible."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.cache is None:
            return self._call_api(texts)

        keys = [EmbeddingCache.key(self.model, t) for t in texts]
        hits = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in hits]
        if missing:
            fresh = self._call_api([texts[i] for i in missing])
            new = {keys[i]: fresh[j] for j, i in enumerate(missing)}
            self.cache.put_many(new)
            hits.update(new)
        log.info("Embeddings: %d cached, %d fetched", len(texts) - len(missing), len(missing))
        return l2_normalize(np.stack([hits[k] for k in keys]))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (never cached)."""
        return self._call_api([text])[0]


# --------------------------------------------------------------------------- #
# Dense retriever (ChromaDB)
# --------------------------------------------------------------------------- #
_COLLECTION_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class DenseRetriever:
    """Cosine-similarity retrieval over a ChromaDB HNSW collection.

    Args:
        name: Human-readable config name; also used to derive the collection name.
        embedder: Any :class:`Embedder`.
        client: Chroma client. Defaults to an in-memory ``EphemeralClient``;
            pass ``chromadb.PersistentClient(path)`` to persist.
        add_batch_size: Max records per ``collection.add`` call.

    After each :meth:`search`, ``last_timing_ms`` holds ``{"embed": .., "ann": ..}``
    so callers can separate network-bound embedding latency from index latency.
    """

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        client: Any | None = None,
        add_batch_size: int = 1000,
    ) -> None:
        self.name = name
        self.embedder = embedder
        self._client = client or chromadb.EphemeralClient()
        self._add_batch_size = add_batch_size
        self._collection_name = _COLLECTION_NAME_RE.sub("_", name)[:63].strip("_-") or "chunks"
        self._collection: Any | None = None
        self.last_timing_ms: dict[str, float] = {}

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """(Re)build the collection from scratch with the given chunks."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # collection didn't exist — Chroma raises varying types
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we always supply vectors ourselves
        )
        if not chunks:
            return

        vectors = self.embedder.embed_documents([c.content for c in chunks])
        for i in range(0, len(chunks), self._add_batch_size):
            batch = chunks[i : i + self._add_batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[i : i + len(batch)].tolist(),
                documents=[c.content for c in batch],
                metadatas=[
                    {"parent_doc": c.parent_doc, "meta_json": json.dumps(c.metadata)}
                    for c in batch
                ],
            )
        log.info("[%s] indexed %d chunks", self.name, len(chunks))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Embed ``query`` and return the top-``k`` chunks by cosine similarity."""
        if self._collection is None:
            raise RuntimeError(f"{self.name}: call index() before search()")
        n = min(k, self._collection.count())
        if n == 0:
            return []

        t0 = time.perf_counter()
        q_vec = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        res = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        t2 = time.perf_counter()
        self.last_timing_ms = {"embed": (t1 - t0) * 1e3, "ann": (t2 - t1) * 1e3}

        results: list[SearchResult] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]),
            start=1,
        ):
            results.append(
                SearchResult(
                    chunk_id=cid,
                    parent_doc=meta["parent_doc"],
                    content=doc,
                    score=1.0 - float(dist),  # cosine distance → similarity
                    rank=rank,
                    metadata=json.loads(meta["meta_json"]),
                )
            )
        return results


# --------------------------------------------------------------------------- #
# Sparse retriever (BM25)
# --------------------------------------------------------------------------- #
_NON_WORD_RE = re.compile(r"[^\w\s]+")  # punctuation (keeps '_' — see below)


def bm25_tokenize(text: str, split_identifiers: bool = True) -> list[str]:
    """Lower-case, strip punctuation, whitespace-split.

    Underscore is kept inside tokens so code identifiers like ``status_code``
    survive as exact-match terms. With ``split_identifiers`` the parts are
    *also* emitted (``status_code`` → ``status_code``, ``status``, ``code``) so
    natural-language queries ("status code") still match.
    """
    tokens: list[str] = []
    for tok in _NON_WORD_RE.sub(" ", text.lower()).split():
        tok = tok.strip("_")
        if not tok:
            continue
        tokens.append(tok)
        if split_identifiers and "_" in tok:
            tokens.extend(p for p in tok.split("_") if p)
    return tokens


class BM25Retriever:
    """Okapi BM25 keyword retrieval via ``rank_bm25.BM25Okapi``.

    Args:
        name: Config name for reports.
        k1, b: Standard BM25 parameters.
    """

    def __init__(self, name: str = "bm25", k1: float = 1.5, b: float = 0.75) -> None:
        self.name = name
        self.k1, self.b = k1, b
        self._bm25: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Tokenize and build the inverted statistics."""
        self._chunks = list(chunks)
        corpus = [bm25_tokenize(c.content) for c in self._chunks]
        # BM25Okapi divides by corpus size; guard the empty case.
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None
        log.info("[%s] indexed %d chunks", self.name, len(self._chunks))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Score every chunk and return the top-``k`` with a positive score."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(bm25_tokenize(query))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]  # O(n) selection, then sort k
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            _to_result(self._chunks[i], scores[i], rank)
            for rank, i in enumerate((i for i in top if scores[i] > 0), start=1)
        ]
