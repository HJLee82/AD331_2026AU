"""Structure-aware Markdown chunker.

Guarantees:
  * Splits on ``#``..``######`` headers and records the full header path
    (e.g. ``3. Kubernetes Clusters > 3.3 System Requirements``) as a breadcrumb.
  * ``#`` lines *inside* fenced code blocks (bash comments!) are never headers.
  * Fenced code blocks and tables are atomic. If one exceeds the size budget it
    is split on line boundaries and re-wrapped (fences re-opened / table header
    repeated) so every chunk stays valid Markdown.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .text_utils import slugify

_HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_HR_RE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_SPLIT_LEVELS: tuple[tuple[str, str], ...] = (
    (r"\n", "\n"),
    (r"(?<=[.!?])\s+", " "),
    (r"\s+", " "),
)


@dataclass(frozen=True, slots=True)
class MarkdownChunk:
    chunk_id: str
    source_file: str
    header_path: tuple[str, ...]
    body: str

    @property
    def section(self) -> str:
        return " > ".join(self.header_path) if self.header_path else "(preamble)"

    @property
    def content(self) -> str:
        """Text that is embedded and shown to the LLM (breadcrumb + body)."""
        return f"Source: {self.source_file}\nSection: {self.section}\n\n{self.body}"


def _is_fence_close(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and set(stripped) == {fence[0]}


def _split_to_fit(text: str, budget: int, level: int = 0) -> list[str]:
    """Recursively split ``text`` into pieces of at most ``budget`` characters."""
    if len(text) <= budget:
        return [text]
    if level >= len(_SPLIT_LEVELS):
        return [text[i : i + budget] for i in range(0, len(text), budget)]

    pattern, joiner = _SPLIT_LEVELS[level]
    parts = [p for p in re.split(pattern, text) if p]
    if len(parts) <= 1:
        return _split_to_fit(text, budget, level + 1)

    out: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{joiner}{part}" if current else part
        if len(candidate) <= budget:
            current = candidate
            continue
        if current:
            out.append(current)
        if len(part) <= budget:
            current = part
        else:
            out.extend(_split_to_fit(part, budget, level + 1))
            current = ""
    if current:
        out.append(current)
    return out


class MarkdownChunker:
    def __init__(self, max_chars: int = 1200, overlap_chars: int = 200) -> None:
        if max_chars < 200:
            raise ValueError("max_chars must be >= 200")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    # ------------------------------------------------------------------ public
    def chunk_file(self, path: Path) -> list[MarkdownChunk]:
        return self.chunk_text(path.read_text(encoding="utf-8"), source_file=path.name)

    def chunk_text(self, text: str, source_file: str) -> list[MarkdownChunk]:
        chunks: list[MarkdownChunk] = []
        for header_path, lines in self._split_sections(text):
            blocks = self._split_blocks(lines)
            for body in self._pack(blocks):
                slug = slugify(header_path[-1]) if header_path else "preamble"
                chunk_id = f"{source_file}::{slug}::{len(chunks):03d}"
                chunks.append(MarkdownChunk(chunk_id, source_file, header_path, body))
        return chunks

    # ----------------------------------------------------------------- parsing
    @staticmethod
    def _split_sections(text: str) -> list[tuple[tuple[str, ...], list[str]]]:
        sections: list[tuple[tuple[str, ...], list[str]]] = []
        path: list[tuple[int, str]] = []
        buffer: list[str] = []
        fence: str | None = None

        def flush() -> None:
            if any(line.strip() for line in buffer):
                sections.append((tuple(title for _, title in path), buffer.copy()))
            buffer.clear()

        for line in text.splitlines():
            if fence is not None:
                buffer.append(line)
                if _is_fence_close(line, fence):
                    fence = None
                continue
            fence_match = _FENCE_OPEN_RE.match(line)
            if fence_match:
                fence = fence_match.group(1)
                buffer.append(line)
                continue
            header = _HEADER_RE.match(line)
            if header:
                flush()
                level = len(header.group(1))
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, header.group(2).strip()))
                continue
            buffer.append(line)
        flush()
        return sections

    @staticmethod
    def _split_blocks(lines: list[str]) -> list[str]:
        """Group lines into paragraphs; fenced blocks are kept whole."""
        blocks: list[str] = []
        current: list[str] = []
        fence: str | None = None

        def emit() -> None:
            if current:
                blocks.append("\n".join(current).strip("\n"))
                current.clear()

        for line in lines:
            if fence is not None:
                current.append(line)
                if _is_fence_close(line, fence):
                    fence = None
                    emit()
                continue
            fence_match = _FENCE_OPEN_RE.match(line)
            if fence_match:
                emit()
                fence = fence_match.group(1)
                current.append(line)
                continue
            if not line.strip() or _HR_RE.match(line):
                emit()
                continue
            current.append(line)
        emit()
        return [b for b in blocks if b.strip()]

    # ----------------------------------------------------------------- packing
    def _pack(self, blocks: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        size = 0
        for block in (piece for b in blocks for piece in self._explode(b)):
            added = len(block) + (2 if current else 0)
            if current and size + added > self.max_chars:
                chunks.append("\n\n".join(current))
                tail = current[-1]
                keep_tail = (
                    len(tail) <= self.overlap_chars and len(tail) + 2 + len(block) <= self.max_chars
                )
                current = [tail] if keep_tail else []
                size = len(tail) if keep_tail else 0
                added = len(block) + (2 if current else 0)
            current.append(block)
            size += added
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _explode(self, block: str) -> Iterator[str]:
        if len(block) <= self.max_chars:
            yield block
            return
        lines = block.split("\n")
        fence_match = _FENCE_OPEN_RE.match(lines[0])
        if fence_match:
            yield from self._split_code(lines, fence_match.group(1))
        elif all(line.lstrip().startswith("|") for line in lines):
            yield from self._split_table(lines)
        else:
            yield from _split_to_fit(block, self.max_chars)

    def _split_code(self, lines: list[str], fence: str) -> Iterator[str]:
        opener = lines[0]
        closed = len(lines) > 1 and _is_fence_close(lines[-1], fence)
        closer = lines[-1] if closed else fence
        inner = "\n".join(lines[1:-1] if closed else lines[1:])
        budget = max(1, self.max_chars - len(opener) - len(closer) - 2)
        for piece in _split_to_fit(inner, budget):
            yield f"{opener}\n{piece}\n{closer}"

    def _split_table(self, lines: list[str]) -> Iterator[str]:
        has_header = len(lines) >= 2 and bool(_TABLE_SEP_RE.match(lines[1]))
        header = "\n".join(lines[:2]) if has_header else ""
        rows = "\n".join(lines[2:] if has_header else lines)
        budget = max(1, self.max_chars - len(header) - 1)
        for piece in _split_to_fit(rows, budget):
            yield f"{header}\n{piece}" if header else piece
