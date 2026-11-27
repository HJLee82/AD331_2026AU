from pathlib import Path

from src.chunker import MarkdownChunker

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _balanced_fences(text: str) -> bool:
    return sum(1 for ln in text.splitlines() if ln.strip().startswith("```")) % 2 == 0


def test_bash_comment_inside_fence_is_not_a_header():
    md = "# Title\n\n## Setup\n\n```bash\n# Install the CLI\nnimctl version\n```\n"
    chunks = MarkdownChunker().chunk_text(md, "x.md")
    assert len(chunks) == 1
    assert chunks[0].header_path == ("Title", "Setup")
    assert "# Install the CLI" in chunks[0].body


def test_header_path_tracks_nesting():
    md = "# A\n\n## B\n\ntext b\n\n### C\n\ntext c\n\n## D\n\ntext d\n"
    paths = [c.header_path for c in MarkdownChunker().chunk_text(md, "x.md")]
    assert paths == [("A", "B"), ("A", "B", "C"), ("A", "D")]


def test_oversized_table_repeats_header_and_code_stays_fenced():
    rows = "\n".join(f"| row{i} | {'x' * 40} |" for i in range(60))
    code = "\n".join(f"echo line{i}" for i in range(200))
    md = f"# T\n\n| Name | Value |\n|---|---|\n{rows}\n\n```bash\n{code}\n```\n"
    chunks = MarkdownChunker(max_chars=600, overlap_chars=100).chunk_text(md, "x.md")
    tables = [c for c in chunks if "| row" in c.body]
    assert len(tables) > 1
    assert all(c.body.startswith("| Name | Value |\n|---|---|") for c in tables)
    assert all(_balanced_fences(c.body) for c in chunks)
    assert all(len(c.body) <= 600 for c in chunks)


def test_real_corpus_invariants():
    chunker = MarkdownChunker()
    chunks = [c for p in sorted(DOCS.glob("*.md")) for c in chunker.chunk_file(p)]
    assert len(chunks) > 30
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(_balanced_fences(c.body) for c in chunks)
    sysreq = [c for c in chunks if "3.3 System Requirements" in c.section]
    assert any("32 GiB" in c.body for c in sysreq)
    assert any("NIM-E4021" in c.content for c in chunks if c.source_file == "deployment_runbook.md")
