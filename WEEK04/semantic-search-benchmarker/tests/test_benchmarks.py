"""Test 5: metric math on hand-made rankings with known answers."""

import pytest

from src.benchmarker import EvalQuery, score_query
from src.indexer import SearchResult

QUERY = EvalQuery(query_id="t", query="x", type="exact",
                  relevant_sections=["a", "b"], source_file="doc.md")


def ranking(*anchors):
    """Build results where position i covers the given anchor (None = irrelevant)."""
    return [
        SearchResult(chunk_id=f"r{i}", parent_doc="doc.md", content="", score=1.0 - i / 10,
                     rank=i, metadata={"section_anchors": [a] if a else [], "anchor_path": []})
        for i, a in enumerate(anchors, start=1)
    ]


def test_metric_math():
    """Test 5: Hit, Recall and MRR for hits, misses, out-of-K hits and ties."""
    # First relevant at rank 3 → RR = 1/3, one of two anchors found.
    s = score_query(ranking(None, None, "a", None, None), QUERY, k=5)
    assert (s.hit, s.recall, s.rr) == (1.0, 0.5, pytest.approx(1 / 3))

    # Both anchors found, first at rank 1.
    s = score_query(ranking("a", None, "b"), QUERY, k=5)
    assert (s.hit, s.recall, s.rr) == (1.0, 1.0, 1.0)

    # Relevant item exists but only at rank 4, cutoff K=3 → no credit at all.
    s = score_query(ranking(None, None, None, "a"), QUERY, k=3)
    assert (s.hit, s.recall, s.rr) == (0.0, 0.0, 0.0)

    # Total miss → MRR must be 0.0, not a division-by-zero error.
    s = score_query(ranking(None, None, None), QUERY, k=5)
    assert (s.hit, s.recall, s.rr) == (0.0, 0.0, 0.0)

    # Empty result list (e.g. BM25 with no matching words).
    s = score_query([], QUERY, k=5)
    assert (s.hit, s.recall, s.rr) == (0.0, 0.0, 0.0)

    # Tie: two relevant chunks with equal scores at ranks 2 and 3 → only the first counts for RR.
    tied = ranking(None, "a", "a")
    tied[2] = tied[2].model_copy(update={"score": tied[1].score})
    s = score_query(tied, QUERY, k=5)
    assert s.rr == pytest.approx(1 / 2)

    # Mean over queries: MRR = mean of (1, 1/3, 0) = 4/9.
    rrs = [score_query(ranking(*r), QUERY, k=5).rr for r in (("a",), (None, None, "b"), (None,))]
    assert sum(rrs) / len(rrs) == pytest.approx(4 / 9)
