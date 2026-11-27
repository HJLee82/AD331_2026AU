"""Assignment Tests 2 and 3: sigmoid logit normalization and threshold gating (mocked model)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.reranker import CrossEncoderReranker
from src.schemas import RerankStatus, RetrievedChunk
from src.text_utils import sigmoid


def _logit(p: float) -> float:
    """Inverse sigmoid: the raw logit that sigmoid maps to probability p."""
    return math.log(p / (1.0 - p))


class FixedLogitCrossEncoder:
    """Returns pre-set raw logits in order, like CrossEncoder.predict with no activation."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.array(self.logits[: len(pairs)])


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=f"doc.md::s{i}::{i:03d}", content=f"text {i}", score=0.0,
                       source="internal_doc")
        for i in range(n)
    ]


# Test 2 ---------------------------------------------------------------------
@pytest.mark.parametrize("logit", [-1000.0, -10.0, -0.5, 0.0, 0.2, 10.0, 1000.0])
def test_2_sigmoid_output_is_bounded(logit):
    value = sigmoid(logit)
    assert 0.0 <= value <= 1.0
    assert not math.isnan(value)


def test_2_sigmoid_known_values_and_order():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(0.2) == pytest.approx(0.5498, abs=1e-4)  # why 0.55 ≈ logit 0.2
    assert sigmoid(-10.0) < sigmoid(-0.5) < sigmoid(0.5) < sigmoid(10.0)


def test_2_reranker_scores_are_normalized_from_raw_logits():
    raw = [-11.2, -3.0, 0.0, 4.5, 9.8]  # typical ms-marco range, NOT probabilities
    reranker = CrossEncoderReranker(model=FixedLogitCrossEncoder(raw), threshold=0.55)
    scored = reranker.score("q", _chunks(len(raw)))
    assert all(0.0 <= c.score <= 1.0 for c in scored)
    assert scored[0].score == pytest.approx(sigmoid(9.8))


# Test 3 ---------------------------------------------------------------------
def test_3_top_score_042_triggers_low_confidence():
    logits = [_logit(0.42), _logit(0.30), _logit(0.10)]
    result = CrossEncoderReranker(model=FixedLogitCrossEncoder(logits), threshold=0.55).rerank(
        "q", _chunks(3)
    )
    assert result.top_confidence == pytest.approx(0.42)
    assert result.status is RerankStatus.LOW_CONFIDENCE
    assert result.needs_fallback
    assert result.chunks == []


def test_3_top_score_081_passes_the_gate():
    logits = [_logit(0.20), _logit(0.81), _logit(0.60), _logit(0.05)]
    result = CrossEncoderReranker(
        model=FixedLogitCrossEncoder(logits), threshold=0.55, top_k=3
    ).rerank("q", _chunks(4))
    assert result.top_confidence == pytest.approx(0.81)
    assert result.status is RerankStatus.OK
    assert not result.needs_fallback
    assert [c.score for c in result.chunks] == pytest.approx([0.81, 0.60, 0.20])
