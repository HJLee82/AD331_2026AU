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
_FENCE_LINE_RE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})", re.MULTILINE)


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

    def annotate(
        self, start_line: int, end_line: int, min_coverage: float = 0.5
    ) -> dict[str, Any]:
        """Structural metadata for a chunk spanning [start_line, end_line].

        A section is credited only if the overlap is at least ``min_coverage``
        of the section (chunk holds most of it) OR of the chunk (chunk is
        mostly about it). Merely touching a section is not enough, so large
        chunks cannot collect credit for every section they brush against.
        """
        lo = max(bisect.bisect_right(self._starts, start_line) - 1, 0)
        hi = bisect.bisect_right(self._starts, end_line)
        chunk_len = end_line - start_line + 1
        credited: list[tuple[int, Section]] = []
        for s in self.sections[lo:hi]:
            overlap = min(end_line, s.end_line) - max(start_line, s.start_line) + 1
            if overlap <= 0:
                continue
            sec_len = s.end_line - s.start_line + 1
            if overlap / sec_len >= min_coverage or overlap / chunk_len >= min_coverage:
                credited.append((overlap, s))
        overlapping = [s.anchor for _, s in credited if s.anchor]
        # Ancestry: keep the section the chunk starts in if it earned credit;
        # otherwise use the section the chunk is mostly about.
        first = self.section_at(start_line)
        if credited and all(s is not first for _, s in credited):
            first = max(credited, key=lambda t: t[0])[1]
        home = first
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
                meta.update(
                    token_count=end - start,
                    chunker=self.name,
                    # Odd number of ``` fences = this window cut a code block in half.
                    broken_fence=_FENCE_LINE_RE.findall(content).__len__() % 2 == 1,
                )
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
