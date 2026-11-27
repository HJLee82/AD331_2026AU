"""Typed data contracts shared by every pipeline stage."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["internal_doc", "web"]
RouteType = Literal["INTERNAL_RAG", "WEB_FALLBACK"]


class RetrievedChunk(BaseModel):
    """A unit of evidence passed between retrieval, re-ranking and generation.

    ``score`` semantics depend on the stage that produced the chunk:
      * Stage 1 (retriever): cosine similarity, a recall signal only.
      * Stage 2 (re-ranker): sigmoid-normalised cross-encoder relevance in [0, 1].
      * Web fallback: provider score (Tavily) or rank-based prior (DDGS) until re-scored.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    content: str
    score: float
    source: SourceType


class RAGResponse(BaseModel):
    """Final, user-facing pipeline output."""

    answer: str
    citations: List[str] = Field(default_factory=list)
    route_taken: RouteType
    confidence_score: float = Field(ge=0.0, le=1.0)
    transformed_query: str


class RerankStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_CANDIDATES = "NO_CANDIDATES"


class RerankResult(BaseModel):
    """Outcome of Stage 2; drives the routing decision."""

    status: RerankStatus
    top_confidence: float = Field(ge=0.0, le=1.0)
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    candidates_scored: int = Field(ge=0)
    ranked: List[RetrievedChunk] = Field(default_factory=list)  # all candidates, re-scored and sorted

    @property
    def needs_fallback(self) -> bool:
        return self.status is not RerankStatus.OK


class ChatTurn(BaseModel):
    """One prior conversation turn, used by the rewriter to resolve pronouns."""

    role: Literal["user", "assistant"]
    content: str
