"""Assignment Test 1: conversational pronoun resolution (real LLM)."""

from __future__ import annotations

import pytest

from src.rewriter import QueryRewriter
from src.schemas import ChatTurn


@pytest.mark.live
def test_1_conversational_pronoun_resolution(live_llm):
    history = [
        ChatTurn(role="user", content="Tell me about the Kubernetes Worker Nodes in Nimbus Cloud."),
        ChatTurn(
            role="assistant",
            content="Kubernetes Worker Nodes run in node pools such as np-general-v3 and "
            "np-memory-v2, and must pass the nodecheck-qv bootstrap check.",
        ),
    ]

    rewritten = QueryRewriter(live_llm).rewrite("What are its system requirements?", history)

    # "its" must be resolved to the subject from the history.
    assert "kubernetes worker node" in rewritten.lower(), rewritten
    assert "requirement" in rewritten.lower(), rewritten
