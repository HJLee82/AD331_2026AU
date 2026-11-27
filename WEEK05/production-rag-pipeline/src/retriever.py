"""Stage 1: ChromaDB dense retrieval (K1 candidates) with exact-identifier boosting."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .chunker import MarkdownChunk, MarkdownChunker
from .config import Settings
from .schemas import RetrievedChunk
from .text_utils import extract_identifiers, slugify

logger = logging.getLogger(__name__)

_ADD_BATCH_SIZE = 128


class RetrieverError(RuntimeError):
    pass


def build_embedding_function(settings: Settings) -> Any:
    from chromadb.utils import embedding_functions as ef

    if settings.embedding_provider == "openai":
        return ef.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key, model_name=settings.embedding_model
        )
    return ef.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


def _batched(items: Sequence[MarkdownChunk], size: int) -> Iterator[Sequence[MarkdownChunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _contains_filter(identifier: str) -> dict[str, Any]:
    """Case-variant ``$contains`` filter (Chroma's ``$contains`` is case-sensitive)."""
    variants = list(dict.fromkeys([identifier, identifier.upper(), identifier.lower()]))
    clauses = [{"$contains": v} for v in variants]
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


class ChromaRetriever:
    """Indexes ``docs/*.md`` and returns the top-K1 candidate chunks for a query.

    Indexing is idempotent: a fingerprint of (file contents, chunker params,
    embedding model) is stored on the collection and the index is rebuilt only
    when it changes. Collections are namespaced by embedding model so switching
    models can never mix vector dimensions.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        chunker: MarkdownChunker | None = None,
        embedding_function: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._chunker = chunker or MarkdownChunker(
            settings.chunk_max_chars, settings.chunk_overlap_chars
        )
        self._ef = embedding_function or build_embedding_function(settings)
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._client = client
        self._collection_name = (
            f"{settings.collection_prefix}__{slugify(settings.embedding_model, max_len=40)}"
        )[:63]
        self._collection: Any | None = None

    # ---------------------------------------------------------------- indexing
    def _doc_paths(self) -> list[Path]:
        paths = sorted(self._settings.docs_dir.glob("*.md"))
        if not paths:
            raise RetrieverError(f"No Markdown files found in {self._settings.docs_dir}")
        return paths

    def _fingerprint(self, paths: Sequence[Path]) -> str:
        h = hashlib.sha256()
        h.update(
            f"{self._settings.embedding_provider}|{self._settings.embedding_model}|"
            f"{self._chunker.max_chars}|{self._chunker.overlap_chars}".encode()
        )
        for path in paths:
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    def _existing_collection(self) -> Any | None:
        names = {c if isinstance(c, str) else c.name for c in self._client.list_collections()}
        if self._collection_name not in names:
            return None
        return self._client.get_collection(self._collection_name, embedding_function=self._ef)

    def index(self, *, force: bool = False) -> int:
        """Build (or reuse) the index. Returns the number of indexed chunks."""
        paths = self._doc_paths()
        fingerprint = self._fingerprint(paths)
        existing = self._existing_collection()

        if (
            existing is not None
            and not force
            and (existing.metadata or {}).get("corpus_fingerprint") == fingerprint
        ):
            self._collection = existing
            logger.info("Index up to date (%d chunks)", existing.count())
            return existing.count()

        chunks = [chunk for path in paths for chunk in self._chunker.chunk_file(path)]
        if existing is not None:
            self._client.delete_collection(self._collection_name)

        collection = self._client.create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
            metadata={
                "hnsw:space": "cosine",
                "corpus_fingerprint": fingerprint,
                "embedding_model": self._settings.embedding_model,
            },
        )
        for batch in _batched(chunks, _ADD_BATCH_SIZE):
            collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.content for c in batch],
                metadatas=[{"source_file": c.source_file, "section": c.section} for c in batch],
            )
        self._collection = collection
        logger.info("Indexed %d chunks from %d files", len(chunks), len(paths))
        return len(chunks)

    def _ensure_collection(self) -> Any:
        if self._collection is None:
            self.index()
        return self._collection

    # --------------------------------------------------------------- retrieval
    @staticmethod
    def _to_chunks(result: dict[str, Any]) -> list[RetrievedChunk]:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=cid, content=doc or "", score=1.0 - float(dist), source="internal_doc"
            )
            for cid, doc, dist in zip(ids, docs, dists, strict=True)
        ]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        collection = self._ensure_collection()
        total = collection.count()
        if total == 0:
            return []
        k = min(top_k or self._settings.retrieval_top_k, total)

        semantic = self._to_chunks(collection.query(query_texts=[query], n_results=k))

        lexical: list[RetrievedChunk] = []
        boost_k = min(self._settings.lexical_boost_k, k)
        if boost_k:
            for ident in extract_identifiers(query):
                try:
                    result = collection.query(
                        query_texts=[query],
                        n_results=boost_k,
                        where_document=_contains_filter(ident),
                    )
                    lexical.extend(self._to_chunks(result))
                except Exception:  # noqa: BLE001 - lexical boost is best-effort
                    logger.debug("Lexical boost failed for %r", ident, exc_info=True)

        merged: dict[str, RetrievedChunk] = {}
        for chunk in lexical:  # exact identifier hits get reserved slots ...
            if len(merged) >= boost_k:
                break
            merged.setdefault(chunk.chunk_id, chunk)
        for chunk in semantic:  # ... the remaining slots go to dense retrieval
            if len(merged) >= k:
                break
            merged.setdefault(chunk.chunk_id, chunk)

        logger.debug(
            "Stage 1: %d candidates (%d lexical, %d semantic)", len(merged), len(lexical), len(semantic)
        )
        return list(merged.values())
 