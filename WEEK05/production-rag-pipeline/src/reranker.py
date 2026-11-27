"""Stage 2: cross-encoder re-ranking, confidence scoring and routing signal."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Any

from .schemas import RerankResult, RerankStatus, RetrievedChunk
from .text_utils import sigmoid

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-scores candidates with a cross-encoder and decides INTERNAL vs FALLBACK.

    Confidence = sigmoid(raw logit). Raw logits are requested explicitly because
    the default activation applied by ``CrossEncoder.predict`` differs between
    models and sentence-transformers versions; normalising ourselves keeps the
    0.55 threshold meaningful (sigmoid(0.2) ~= 0.55 for ms-marco models).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        threshold: float = 0.55,
        top_k: int = 3,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=max_length, device=device)
        self._model = model
        self.threshold = threshold
        self.top_k = top_k
        self._batch_size = batch_size
        params = inspect.signature(model.predict).parameters
        self._activation_kwarg = next(
            (name for name in ("activation_fn", "activation_fct") if name in params), None
        )

    def _logits(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        kwargs: dict[str, Any] = {"batch_size": self._batch_size, "show_progress_bar": False}
        if self._activation_kwarg:
            import torch

            kwargs[self._activation_kwarg] = torch.nn.Identity()
        raw = self._model.predict([(query, c.content) for c in chunks], **kwargs)
        flat = raw.reshape(-1).tolist() if hasattr(raw, "reshape") else list(raw)
        if len(flat) != len(chunks):
            raise RuntimeError(f"Cross-encoder returned {len(flat)} scores for {len(chunks)} pairs")
        return [float(x) for x in flat]

    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return all chunks re-scored (score = sigmoid confidence), sorted descending."""
        if not chunks:
            return []
        scored = [
            c.model_copy(update={"score": sigmoid(logit)})
            for c, logit in zip(chunks, self._logits(query, chunks), strict=True)
        ]
        return sorted(scored, key=lambda c: c.score, reverse=True)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> RerankResult:
        if not candidates:
            return RerankResult(status=RerankStatus.NO_CANDIDATES, top_confidence=0.0, candidates_scored=0)

        ranked = self.score(query, candidates)
        top = ranked[0].score
        if top >= self.threshold:
            status, chunks = RerankStatus.OK, ranked[: self.top_k]
        else:
            status, chunks = RerankStatus.LOW_CONFIDENCE, []

        logger.info(
            "Stage 2: %d scored, top=%.3f (%s), threshold=%.2f",
            len(ranked), top, ranked[0].chunk_id, self.threshold,
        )
        return RerankResult(
            status=status,
            top_confidence=top,
            chunks=chunks,
            candidates_scored=len(ranked),
            ranked=ranked,
        )
