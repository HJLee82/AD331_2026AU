# AI Prompt Log

## Tool & Model
Claude (claude.ai / Claude Code), Nov 2026

## Initial Prompt
You are a Staff Information Retrieval and Search Engineer. I need a modular Python 3.11+ semantic search and evaluation pipeline that indexes technical Markdown documentation, evaluates multiple chunking strategies, and benchmarks dense vector retrieval against BM25 keyword search.

Context:
- Corpus: 12 FastAPI tutorial Markdown files in corpus/. Headers look like: ## Use `HTTPException` { #use-httpexception }
- Eval set: data/eval_queries.json, a list of objects with query_id, query, type ("exact" or "conceptual"), relevant_sections (list of header anchor IDs), source_file.
- Embeddings: OpenAI text-embedding-3-small (OPENAI_API_KEY in .env). Vector store: ChromaDB.

Requirements:

Ingestion & Chunking (src/chunking.py):
- Implement FixedSizeChunker(chunk_size: int, chunk_overlap: int) that splits text on tokens using tiktoken.
- Implement StructureAwareChunker() that splits on Markdown headers (#, ##, ###) while preserving code blocks (```) intact inside their parent sections.
- Each chunk must be a Pydantic model DocumentChunk with: chunk_id (str), content (str), metadata (dict with header hierarchy, line numbers), and parent_doc (str).

Dual Indexing (src/indexer.py):
- Dense Vector Store: Use ChromaDB to embed and store chunks using OpenAI text-embedding-3-small. Normalize embeddings to unit length for cosine distance.
- Sparse Keyword Index: Use rank_bm25.BM25Okapi with tokenized, lower-cased, punctuation-stripped text.

Benchmark Engine (src/benchmarker.py):
- Ingest data/eval_queries.json.
- Compute Hit Rate@K, Recall@K, MRR@K (average of 1 / rank of first relevant chunk), and p50/p95 query latency in ms.
- Benchmark 4 configurations on the same dataset:
  1. BM25 (baseline)
  2. Dense Vector (small chunks: 256 tokens)
  3. Dense Vector (large chunks: 1024 tokens)
  4. Dense Vector (structure-aware chunking)

Clean Code Standards:
- Typed functions, clear docstrings, and decoupled search interfaces returning a standardized SearchResult model.

Provide production-grade, well-commented code split into chunking.py, indexer.py, and benchmarker.py.

## Raw AI Output (unedited)
### chunking.py
```python
"""
Markdown ingestion and chunking strategies.

Two strategies share one data model (:class:`DocumentChunk`) and one source of
truth for document structure (:func:`parse_sections`):

* :class:`FixedSizeChunker` – sliding token window (tiktoken), structure-blind.
* :class:`StructureAwareChunker` – one chunk per Markdown section (``#``–``###``),
  never splitting a fenced code block.

Every chunk is annotated with the *same* structural metadata regardless of
strategy (header hierarchy, anchors, line span). This is what makes the
benchmark fair: relevance is judged against header anchor IDs, so even a
structure-blind fixed-size chunk must know which sections it overlaps.

Line numbers are 1-indexed and inclusive throughout.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DocumentChunk",
    "Section",
    "Chunker",
    "FixedSizeChunker",
    "StructureAwareChunker",
    "parse_sections",
    "slugify",
    "get_encoding",
    "DEFAULT_ENCODING_NAME",
]

# text-embedding-3-* models use cl100k_base, so token counts here match what
# the embedding API sees (and its 8191-token input limit).
DEFAULT_ENCODING_NAME = "cl100k_base"

# ATX header: up to 3 leading spaces, 1–6 '#', mandatory space, title, optional
# trailing '#'s, optional MkDocs/attr_list anchor: { #use-httpexception }
_HEADER_RE = re.compile(
    r"^ {0,3}(?P<hashes>#{1,6})[ \t]+(?P<title>.*?)"
    r"(?:[ \t]*\{[ \t]*#(?P<anchor>[\w\-.:]+)[^}]*\})?"
    r"[ \t#]*$"
)
# Opening/closing fence: ``` or ~~~ (3+), optionally indented up to 3 spaces.
_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
class DocumentChunk(BaseModel):
    """A retrievable unit of text plus provenance metadata.

    Attributes:
        chunk_id: Deterministic, globally unique ID (``<doc>::<strategy>::<n>``).
        content: Raw chunk text sent to the embedder / BM25 tokenizer.
        metadata: Structural provenance. Keys always present:
            ``start_line``, ``end_line``, ``header_path`` (titles, outer→inner),
            ``anchor_path`` (anchors, outer→inner), ``section_anchors``
            (anchors of every section the chunk overlaps), ``token_count``,
            ``chunker``.
        parent_doc: Source file name (e.g. ``handling-errors.md``).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_doc: str


@dataclass(frozen=True, slots=True)
class Section:
    """A Markdown section: a header line and everything up to the next header
    of any level. Level 0 denotes preamble text before the first header."""

    level: int
    title: str
    anchor: str | None
    start_line: int  # line of the header itself (1-indexed)
    end_line: int  # last line before the next header (inclusive)
    header_path: tuple[str, ...]
    anchor_path: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def get_encoding(name: str = DEFAULT_ENCODING_NAME) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding (loading BPE ranks is not free)."""
    return tiktoken.get_encoding(name)


def slugify(title: str) -> str:
    """MkDocs-style slug used when a header has no explicit ``{ #anchor }``.

    ``"Use `HTTPException`"`` → ``"use-httpexception"``.
    """
    text = unicodedata.normalize("NFKD", title)
    text = re.sub(r"[`*_~\[\]()]", "", text)  # drop inline markup
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)


def _fence_states(lines: Sequence[str]) -> list[bool]:
    """For each line, return True if it is *part of* a fenced code block
    (fence lines included). Handles ``` vs ~~~ and longer closing fences per
    CommonMark, so a ``# comment`` inside Python code is never a header."""
    states: list[bool] = []
    open_fence: str | None = None
    for line in lines:
        m = _FENCE_RE.match(line)
        if open_fence is None:
            if m:
                open_fence = m.group("fence")
                states.append(True)
            else:
                states.append(False)
        else:
            states.append(True)
            # Closing fence: same char, at least as long, nothing but whitespace after.
            if m and m.group("fence")[0] == open_fence[0] and len(m.group("fence")) >= len(
                open_fence
            ) and not line.strip()[len(m.group("fence")):].strip():
                open_fence = None
    return states


def parse_sections(text: str, max_level: int = 6) -> list[Section]:
    """Parse a Markdown document into a flat, ordered list of sections.

    Headers inside fenced code blocks are ignored. Sections are contiguous and
    cover every line of the document exactly once.

    Args:
        text: Markdown source.
        max_level: Deepest header level that starts a new section.

    Returns:
        Sections in document order; the first is a level-0 preamble only when
        the document has content before its first header.
    """
    lines = text.splitlines()
    in_code = _fence_states(lines)

    # (line_no, level, title, anchor)
    headers: list[tuple[int, int, str, str]] = []
    for i, line in enumerate(lines, start=1):
        if in_code[i - 1]:
            continue
        m = _HEADER_RE.match(line)
        if not m or len(m.group("hashes")) > max_level:
            continue
        title = m.group("title").strip()
        anchor = m.group("anchor") or slugify(title)
        headers.append((i, len(m.group("hashes")), title, anchor))

    sections: list[Section] = []
    n_lines = len(lines)
    if not headers or headers[0][0] > 1:
        first = headers[0][0] - 1 if headers else n_lines
        if first >= 1 and any(l.strip() for l in lines[:first]):
            sections.append(Section(0, "", None, 1, first, (), ()))

    # Stack of (level, title, anchor) giving the current ancestry.
    stack: list[tuple[int, str, str]] = []
    for idx, (line_no, level, title, anchor) in enumerate(headers):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title, anchor))
        end = headers[idx + 1][0] - 1 if idx + 1 < len(headers) else n_lines
        sections.append(
            Section(
                level=level,
                title=title,
                anchor=anchor,
                start_line=line_no,
                end_line=end,
                header_path=tuple(s[1] for s in stack),
                anchor_path=tuple(s[2] for s in stack),
            )
        )
    return sections


@dataclass
class _StructureIndex:
    """Line-range → structure lookup shared by all chunkers."""

    sections: list[Section]
    _starts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self._starts = [s.start_line for s in self.sections]

    def section_at(self, line: int) -> Section | None:
        """Innermost section containing ``line``."""
        i = bisect.bisect_right(self._starts, line) - 1
        return self.sections[i] if i >= 0 else None

    def annotate(self, start_line: int, end_line: int) -> dict[str, Any]:
        """Structural metadata for a chunk spanning [start_line, end_line]."""
        home = self.section_at(start_line)
        lo = max(bisect.bisect_right(self._starts, start_line) - 1, 0)
        hi = bisect.bisect_right(self._starts, end_line)
        overlapping = [s.anchor for s in self.sections[lo:hi] if s.anchor]
        return {
            "start_line": start_line,
            "end_line": end_line,
            "header_path": list(home.header_path) if home else [],
            "anchor_path": list(home.anchor_path) if home else [],
            "section_anchors": list(dict.fromkeys(overlapping)),  # dedupe, keep order
        }


# --------------------------------------------------------------------------- #
# Chunker interface
# --------------------------------------------------------------------------- #
@runtime_checkable
class Chunker(Protocol):
    """Anything that turns one document into chunks."""

    @property
    def name(self) -> str:  # used in chunk IDs and report labels
        ...

    def chunk(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        ...


def chunk_corpus(chunker: Chunker, docs: dict[str, str]) -> list[DocumentChunk]:
    """Apply ``chunker`` to every ``{filename: text}`` pair, in sorted order
    so chunk IDs are reproducible across runs."""
    out: list[DocumentChunk] = []
    for name in sorted(docs):
        out.extend(chunker.chunk(docs[name], parent_doc=name))
    return out


def _make_id(parent_doc: str, strategy: str, n: int) -> str:
    return f"{Path(parent_doc).stem}::{strategy}::{n:04d}"


# --------------------------------------------------------------------------- #
# Fixed-size (token window) chunker
# --------------------------------------------------------------------------- #
class FixedSizeChunker:
    """Sliding-window chunker over tiktoken tokens.

    Windows are ``chunk_size`` tokens long and advance by
    ``chunk_size - chunk_overlap``. Chunk text is sliced from the *original
    string* using tiktoken's per-token character offsets, so multi-byte
    characters are never split and content is byte-for-byte faithful.

    Args:
        chunk_size: Tokens per chunk (> 0).
        chunk_overlap: Tokens shared by consecutive chunks (0 ≤ overlap < size).
        encoding: Optional pre-built encoding (injectable for tests/offline use).
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        encoding: tiktoken.Encoding | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = encoding or get_encoding()

    @property
    def name(self) -> str:
        return f"fixed{self.chunk_size}o{self.chunk_overlap}"

    def chunk(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        """Split ``text`` into overlapping token windows."""
        tokens = self._enc.encode(text, disallowed_special=())
        if not tokens:
            return []
        _, char_offsets = self._enc.decode_with_offsets(tokens)
        char_offsets.append(len(text))  # sentinel: end of last token

        # Char offset → 1-indexed line number.
        line_starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
        to_line = lambda c: bisect.bisect_right(line_starts, c)  # noqa: E731

        structure = _StructureIndex(parse_sections(text))
        stride = self.chunk_size - self.chunk_overlap
        chunks: list[DocumentChunk] = []

        for n, start in enumerate(range(0, len(tokens), stride)):
            end = min(start + self.chunk_size, len(tokens))
            c0, c1 = char_offsets[start], char_offsets[end]
            content = text[c0:c1]
            if content.strip():
                # Don't let a trailing newline push end_line onto the next line.
                last_char = c1 - 1 if c1 > c0 else c0
                meta = structure.annotate(to_line(c0), to_line(last_char))
                meta.update(token_count=end - start, chunker=self.name)
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_id(parent_doc, self.name, n),
                        content=content,
                        metadata=meta,
                        parent_doc=parent_doc,
                    )
                )
            if end == len(tokens):
                break  # avoid a final window that is pure overlap
        return chunks


# --------------------------------------------------------------------------- #
# Structure-aware chunker
# --------------------------------------------------------------------------- #
class StructureAwareChunker:
    """One chunk per Markdown section split at ``#``, ``##`` and ``###``.

    Deeper headers (``####``+) stay inside their parent chunk. Fenced code
    blocks are atomic: headers inside them are ignored, and the optional
    oversize split never cuts through one.

    Args:
        split_levels: Deepest header level that starts a new chunk (default 3).
        max_tokens: If set, sections larger than this are packed into several
            chunks at block boundaries (blank lines outside code). A single
            block larger than the limit (e.g. a huge code sample) is kept whole.
        include_breadcrumb: Prepend ``"A > B > C"`` header context to content.
            Off by default so the benchmark measures pure structural chunking.
        skip_empty_sections: Drop sections consisting only of a header line
            (e.g. an H1 immediately followed by an H2).
        encoding: Optional pre-built tiktoken encoding.
    """

    def __init__(
        self,
        split_levels: int = 3,
        max_tokens: int | None = None,
        include_breadcrumb: bool = False,
        skip_empty_sections: bool = True,
        encoding: tiktoken.Encoding | None = None,
    ) -> None:
        if not 1 <= split_levels <= 6:
            raise ValueError("split_levels must be in [1, 6]")
        self.split_levels = split_levels
        self.max_tokens = max_tokens
        self.include_breadcrumb = include_breadcrumb
        self.skip_empty_sections = skip_empty_sections
        self._enc = encoding or get_encoding()

    @property
    def name(self) -> str:
        return "structure"

    def _count(self, s: str) -> int:
        return len(self._enc.encode(s, disallowed_special=()))

    def chunk(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        """Split ``text`` at section boundaries."""
        lines = text.splitlines()
        in_code = _fence_states(lines)
        # Annotation uses ALL header levels so #### anchors remain matchable.
        structure = _StructureIndex(parse_sections(text, max_level=6))
        split_sections = parse_sections(text, max_level=self.split_levels)

        chunks: list[DocumentChunk] = []
        for sec in split_sections:
            body_lines = lines[sec.start_line : sec.end_line]  # excludes header line
            if self.skip_empty_sections and sec.level > 0 and not any(
                l.strip() for l in body_lines
            ):
                continue

            for s, e in self._pack(sec, lines, in_code):
                content = "\n".join(lines[s - 1 : e]).strip("\n")
                if not content.strip():
                    continue
                # Continuation pieces repeat their header so they stay self-describing.
                if s != sec.start_line and sec.level > 0:
                    content = f"{lines[sec.start_line - 1]}\n\n{content}"
                if self.include_breadcrumb and sec.header_path:
                    content = f"[{' > '.join(sec.header_path)}]\n{content}"

                meta = structure.annotate(s, e)
                meta.update(
                    token_count=self._count(content),
                    chunker=self.name,
                    section_level=sec.level,
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_id(parent_doc, self.name, len(chunks)),
                        content=content,
                        metadata=meta,
                        parent_doc=parent_doc,
                    )
                )
        return chunks

    def _pack(
        self, sec: Section, lines: list[str], in_code: list[bool]
    ) -> list[tuple[int, int]]:
        """Return 1-indexed (start, end) line spans for one section.

        Without ``max_tokens`` the whole section is one span. Otherwise the
        section is cut into blocks — runs of non-blank prose lines, or a whole
        fenced code block — and blocks are greedily packed under the limit.
        """
        whole = [(sec.start_line, sec.end_line)]
        if self.max_tokens is None:
            return whole
        section_text = "\n".join(lines[sec.start_line - 1 : sec.end_line])
        if self._count(section_text) <= self.max_tokens:
            return whole

        # Build blocks as (start, end) line spans.
        blocks: list[tuple[int, int]] = []
        cur: int | None = None
        for ln in range(sec.start_line, sec.end_line + 1):
            blank = not lines[ln - 1].strip() and not in_code[ln - 1]
            if blank:
                if cur is not None:
                    blocks.append((cur, ln - 1))
                    cur = None
            elif cur is None:
                cur = ln
        if cur is not None:
            blocks.append((cur, sec.end_line))

        spans: list[tuple[int, int]] = []
        span_start, span_tokens = blocks[0][0], 0
        prev_end = blocks[0][0] - 1
        for b_start, b_end in blocks:
            b_tokens = self._count("\n".join(lines[b_start - 1 : b_end]))
            header_only = span_start == prev_end == sec.start_line  # never orphan a header
            if span_tokens and span_tokens + b_tokens > self.max_tokens and not header_only:
                spans.append((span_start, prev_end))
                span_start, span_tokens = b_start, 0
            span_tokens += b_tokens
            prev_end = b_end
        spans.append((span_start, prev_end))
        return spans

```
### indexer.py
```python
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

```
### benchmarker.py
```python
"""
Retrieval benchmark: BM25 vs dense retrieval under different chunking strategies.

Usage::

    python -m src.benchmarker                       # all 4 configs, K = 1,3,5,10
    python -m src.benchmarker --configs bm25        # offline, no API key needed
    python -m src.benchmarker --ks 5 10 --output results/run.json

Relevance model
---------------
Ground truth is a set of header anchors per query (``relevant_sections``) in a
given ``source_file``. A retrieved chunk is relevant to anchor ``a`` when it
comes from that file and either

* overlaps section ``a`` (``metadata["section_anchors"]``), or
* sits *inside* section ``a`` (``a`` in ``metadata["anchor_path"]``), so a
  ``###`` sub-section chunk counts for its ``##`` parent. Disable with
  ``--strict-anchors``.

Metrics (per query, then macro-averaged):

* Hit Rate@K – 1 if any of the top-K chunks is relevant.
* Recall@K   – fraction of the query's relevant anchors covered by the top-K.
* MRR@K      – 1 / rank of the first relevant chunk (0 if none in top-K).
* Latency    – end-to-end ``search()`` wall time, p50 / p95 in ms. For dense
  retrievers this includes the query-embedding API round-trip, which is
  reported separately as well.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from .chunking import (
    Chunker,
    DocumentChunk,
    FixedSizeChunker,
    StructureAwareChunker,
    chunk_corpus,
)
from .indexer import BM25Retriever, DenseRetriever, Embedder, Retriever, SearchResult

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class EvalQuery(BaseModel):
    """One labelled query from ``data/eval_queries.json``."""

    query_id: str
    query: str
    type: Literal["exact", "conceptual"]
    relevant_sections: list[str] = Field(min_length=1)
    source_file: str


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    """Load and validate the evaluation set (fails loudly on schema errors)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = [EvalQuery.model_validate(q) for q in raw]
    ids = [q.query_id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query_id values in eval set")
    return queries


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """Read every ``*.md`` file under ``corpus_dir`` → ``{filename: text}``."""
    files = sorted(Path(corpus_dir).rglob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {corpus_dir}")
    return {f.name: f.read_text(encoding="utf-8") for f in files}


def validate_labels(queries: Sequence[EvalQuery], chunks: Sequence[DocumentChunk]) -> list[str]:
    """Return warnings for labels that no chunk can ever match.

    A typo in an anchor silently caps Recall below 1.0 for *every* config, so
    this is checked up-front.
    """
    known: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        known[c.parent_doc].update(c.metadata["section_anchors"], c.metadata["anchor_path"])
    warnings = []
    for q in queries:
        doc = Path(q.source_file).name
        if doc not in known:
            warnings.append(f"{q.query_id}: source_file {q.source_file!r} not in corpus")
            continue
        for a in q.relevant_sections:
            if a not in known[doc]:
                warnings.append(f"{q.query_id}: anchor {a!r} not found in {doc}")
    return warnings


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def matched_anchors(result: SearchResult, query: EvalQuery, match_ancestors: bool = True) -> set[str]:
    """Relevant anchors of ``query`` that ``result`` satisfies (may be empty)."""
    if result.parent_doc != Path(query.source_file).name:
        return set()
    covered = set(result.metadata.get("section_anchors", []))
    if match_ancestors:
        covered |= set(result.metadata.get("anchor_path", []))
    return covered & set(query.relevant_sections)


@dataclass(frozen=True)
class QueryScore:
    hit: float
    recall: float
    rr: float


def score_query(
    results: Sequence[SearchResult], query: EvalQuery, k: int, match_ancestors: bool = True
) -> QueryScore:
    """Hit, Recall and reciprocal rank at cutoff ``k`` for one query."""
    relevant = set(query.relevant_sections)
    covered: set[str] = set()
    first_rank: int | None = None
    for r in results[:k]:
        m = matched_anchors(r, query, match_ancestors)
        if m and first_rank is None:
            first_rank = r.rank
        covered |= m
    return QueryScore(
        hit=1.0 if first_rank else 0.0,
        recall=len(covered) / len(relevant),
        rr=1.0 / first_rank if first_rank else 0.0,
    )


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


# --------------------------------------------------------------------------- #
# Benchmark orchestration
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkConfig:
    """A (chunker, retriever) pairing to evaluate."""

    name: str
    chunker: Chunker
    retriever_factory: Callable[[str], Retriever]


class ConfigReport(BaseModel):
    """Everything measured for one configuration."""

    name: str
    num_chunks: int
    avg_chunk_tokens: float
    index_time_s: float
    metrics: dict[str, dict[str, float]]  # {"all"|"exact"|"conceptual": {"hit@5": ..}}
    latency_ms: dict[str, float]
    per_query: list[dict]


class BenchmarkRunner:
    """Runs every config over the same corpus and query set.

    Args:
        docs: ``{filename: markdown}`` corpus.
        queries: Labelled eval queries.
        ks: Cutoffs to report. Each query is searched once at ``max(ks)``.
        match_ancestors: See module docstring.
    """

    def __init__(
        self,
        docs: dict[str, str],
        queries: Sequence[EvalQuery],
        ks: Sequence[int] = (1, 3, 5, 10),
        match_ancestors: bool = True,
    ) -> None:
        self.docs = docs
        self.queries = list(queries)
        self.ks = sorted(set(ks))
        self.match_ancestors = match_ancestors
        self._chunk_cache: dict[str, list[DocumentChunk]] = {}

    def _chunks_for(self, chunker: Chunker) -> list[DocumentChunk]:
        # BM25 and dense-structure share the same chunks; chunk once.
        if chunker.name not in self._chunk_cache:
            self._chunk_cache[chunker.name] = chunk_corpus(chunker, self.docs)
        return self._chunk_cache[chunker.name]

    def run_config(self, cfg: BenchmarkConfig) -> ConfigReport:
        chunks = self._chunks_for(cfg.chunker)
        for w in validate_labels(self.queries, chunks):
            log.warning("[%s] %s", cfg.name, w)

        retriever = cfg.retriever_factory(cfg.name)
        t0 = time.perf_counter()
        retriever.index(chunks)
        index_time = time.perf_counter() - t0

        max_k = self.ks[-1]
        retriever.search(self.queries[0].query, k=max_k)  # warm-up (connections, caches)

        latencies: list[float] = []
        embed_lat: list[float] = []
        per_query: list[dict] = []
        sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for q in self.queries:
            t = time.perf_counter()
            results = retriever.search(q.query, k=max_k)
            latencies.append((time.perf_counter() - t) * 1e3)
            if timing := getattr(retriever, "last_timing_ms", None):
                embed_lat.append(timing["embed"])

            row: dict = {"query_id": q.query_id, "type": q.type, "latency_ms": latencies[-1],
                         "top_ids": [r.chunk_id for r in results[:5]]}
            for k in self.ks:
                s = score_query(results, q, k, self.match_ancestors)
                for bucket in ("all", q.type):
                    sums[bucket][f"hit@{k}"].append(s.hit)
                    sums[bucket][f"recall@{k}"].append(s.recall)
                    sums[bucket][f"mrr@{k}"].append(s.rr)
                row.update({f"hit@{k}": s.hit, f"recall@{k}": s.recall, f"mrr@{k}": s.rr})
            per_query.append(row)

        latency = {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)}
        if embed_lat:
            latency |= {"embed_p50": percentile(embed_lat, 50),
                        "search_only_p50": percentile([a - b for a, b in zip(latencies, embed_lat)], 50)}

        return ConfigReport(
            name=cfg.name,
            num_chunks=len(chunks),
            avg_chunk_tokens=statistics.fmean(c.metadata["token_count"] for c in chunks) if chunks else 0.0,
            index_time_s=index_time,
            metrics={b: {m: statistics.fmean(v) for m, v in d.items()} for b, d in sums.items()},
            latency_ms=latency,
            per_query=per_query,
        )

    def run(self, configs: Sequence[BenchmarkConfig]) -> list[ConfigReport]:
        reports = []
        for cfg in configs:
            log.info("Running %s ...", cfg.name)
            reports.append(self.run_config(cfg))
        return reports


# --------------------------------------------------------------------------- #
# Default experiment
# --------------------------------------------------------------------------- #
def build_default_configs(
    embedder_factory: Callable[[], Embedder] | None,
    chroma_client: object | None = None,
    encoding: object | None = None,
) -> list[BenchmarkConfig]:
    """The four required configurations.

    BM25 runs on structure-aware chunks so the sparse/dense comparison isn't
    confounded by chunk size. Fixed-size overlap is 12.5% of chunk size.
    ``embedder_factory`` may be None for BM25-only runs.
    """
    structure = StructureAwareChunker(encoding=encoding)
    _embedder: list[Embedder] = []

    def dense(name: str) -> Retriever:
        if embedder_factory is None:
            raise RuntimeError("Dense configs need an embedder (is OPENAI_API_KEY set?)")
        if not _embedder:
            _embedder.append(embedder_factory())  # one client/cache shared by all dense configs
        return DenseRetriever(name, _embedder[0], client=chroma_client)

    return [
        BenchmarkConfig("bm25", structure, lambda n: BM25Retriever(n)),
        BenchmarkConfig("dense_fixed_256", FixedSizeChunker(256, 32, encoding=encoding), dense),
        BenchmarkConfig("dense_fixed_1024", FixedSizeChunker(1024, 128, encoding=encoding), dense),
        BenchmarkConfig("dense_structure", structure, dense),
    ]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(reports: Sequence[ConfigReport], ks: Sequence[int]) -> str:
    """Markdown tables: overall metrics, then MRR/Hit split by query type."""
    lines: list[str] = []
    cols = [f"{m}@{k}" for k in ks for m in ("hit", "recall", "mrr")]
    header = ["config", "chunks", "avg tok", *cols, "p50 ms", "p95 ms"]
    lines += ["## Overall", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in reports:
        m = r.metrics["all"]
        lines.append("| " + " | ".join([
            r.name, str(r.num_chunks), f"{r.avg_chunk_tokens:.0f}",
            *(f"{m[c]:.3f}" for c in cols),
            f"{r.latency_ms['p50']:.1f}", f"{r.latency_ms['p95']:.1f}",
        ]) + " |")

    k = ks[-1] if 5 not in ks else 5
    lines += ["", f"## By query type (@{k})", "",
              "| config | exact hit | exact mrr | conceptual hit | conceptual mrr |",
              "|---|---|---|---|---|"]
    for r in reports:
        e, c = r.metrics.get("exact", {}), r.metrics.get("conceptual", {})
        fmt = lambda d, key: f"{d[key]:.3f}" if key in d else "–"  # noqa: E731
        lines.append(f"| {r.name} | {fmt(e, f'hit@{k}')} | {fmt(e, f'mrr@{k}')} | "
                     f"{fmt(c, f'hit@{k}')} | {fmt(c, f'mrr@{k}')} |")

    dense = [r for r in reports if "embed_p50" in r.latency_ms]
    if dense:
        lines += ["", "## Dense latency breakdown (p50 ms)", "",
                  "| config | query embed | vector search |", "|---|---|---|"]
        lines += [f"| {r.name} | {r.latency_ms['embed_p50']:.1f} | "
                  f"{r.latency_ms['search_only_p50']:.2f} |" for r in dense]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--queries", default="data/eval_queries.json")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--configs", nargs="+", help="Subset of config names to run")
    p.add_argument("--output", default="results/benchmark.json")
    p.add_argument("--persist-dir", help="Persist Chroma here (default: in-memory)")
    p.add_argument("--cache", default=".cache/embeddings.sqlite",
                   help="Document-embedding cache ('' to disable)")
    p.add_argument("--strict-anchors", action="store_true",
                   help="Don't credit sub-section chunks to their parent anchor")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv
    load_dotenv()

    def embedder_factory() -> Embedder:
        from .indexer import EmbeddingCache, OpenAIEmbedder
        return OpenAIEmbedder(cache=EmbeddingCache(args.cache) if args.cache else None)

    chroma_client = None
    if args.persist_dir:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=args.persist_dir)

    configs = build_default_configs(embedder_factory, chroma_client)
    if args.configs:
        unknown = set(args.configs) - {c.name for c in configs}
        if unknown:
            p.error(f"Unknown configs: {sorted(unknown)}")
        configs = [c for c in configs if c.name in args.configs]

    runner = BenchmarkRunner(
        load_corpus(args.corpus), load_eval_queries(args.queries),
        ks=args.ks, match_ancestors=not args.strict_anchors,
    )
    reports = runner.run(configs)

    print(format_report(reports, runner.ks))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.model_dump() for r in reports], indent=2), encoding="utf-8")
    print(f"\nFull results (incl. per-query rows): {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## Audit Findings
