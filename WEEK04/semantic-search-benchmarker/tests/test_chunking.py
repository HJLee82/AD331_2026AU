"""Tests 1–2: chunk boundaries."""

from src.chunking import FixedSizeChunker, StructureAwareChunker, get_encoding

DOC_WITH_CODE = """# Debugging { #debugging }

Intro text about running the app.

## Call uvicorn { #call-uvicorn }

Run the server like this:

```Python
import uvicorn

# Some more code
app = make_app()

## not a header either
uvicorn.run(app)
```

Text after the code block.

## Run your debugger { #run-your-debugger }

Set a breakpoint and step through.
"""


def test_fixed_chunk_overlap_integrity():
    """Test 1: consecutive chunks share exactly `overlap` tokens, and stitching
    the chunks back together (minus overlaps) reproduces the text with no drops."""
    enc = get_encoding()
    text = " ".join(f"word{i} value_{i}," for i in range(400))
    size, overlap = 64, 16
    chunks = FixedSizeChunker(size, overlap).chunk(text, "doc.md")
    assert len(chunks) > 3

    tokens = enc.encode(text, disallowed_special=())
    _, offsets = enc.decode_with_offsets(tokens)
    offsets.append(len(text))
    stride = size - overlap

    rebuilt = chunks[0].content
    for i in range(len(chunks) - 1):
        start = i * stride
        shared = text[offsets[start + stride]: offsets[start + size]]
        assert len(enc.encode(shared)) == overlap
        assert chunks[i].content.endswith(shared)
        assert chunks[i + 1].content.startswith(shared)
        rebuilt += chunks[i + 1].content[len(shared):]
    assert rebuilt == text


def test_structure_chunker_keeps_code_fences_whole():
    """Test 2: a code block is never split, even when lines inside it start
    with '#' (Python comments look like Markdown headers)."""
    chunks = StructureAwareChunker().chunk(DOC_WITH_CODE, "debugging.md")
    titles = [c.metadata["header_path"][-1] for c in chunks]

    assert "Some more code" not in titles
    assert "not a header either" not in titles
    for c in chunks:
        assert c.content.count("```") % 2 == 0, f"split fence in {c.chunk_id}"

    code_chunk = next(c for c in chunks if "import uvicorn" in c.content)
    assert "uvicorn.run(app)" in code_chunk.content
    assert code_chunk.metadata["anchor_path"][-1] == "call-uvicorn"
