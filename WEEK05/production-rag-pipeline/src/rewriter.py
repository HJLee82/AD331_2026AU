"""Query Rewriter: conversational / underspecified input -> retrieval-optimised query."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from .llm import ChatLLM, LLMError
from .schemas import ChatTurn
from .text_utils import extract_identifiers

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 512
_HISTORY_TURNS = 6
_HISTORY_TURN_CHARS = 600
_PREFIX_RE = re.compile(r"^(?:rewritten\s+)?(?:search\s+)?query\s*:\s*", re.IGNORECASE)

REWRITER_SYSTEM_PROMPT = """\
You rewrite user messages into standalone search queries for Nimbus Cloud's internal \
knowledge base (cloud architecture, deployment runbook, billing policy).

Rules:
1. Output ONLY the rewritten query on a single line. No quotes, labels or explanations.
2. Remove conversational filler (greetings, "can you tell me", "I was wondering", thanks).
3. Resolve pronouns and vague references ("it", "that", "the second one") using the \
conversation history. If the history does not resolve them, leave them as they are.
4. Copy identifiers EXACTLY as written: error codes (e.g. NIM-E4021), CLI commands and \
flags (e.g. nimctl deploy --canary 10), module, service, VPC, plan and cluster names, \
numbers and units.
5. Never answer the question and never add facts or terms that are not in the message \
or the history.
6. Keep the user's language. If the message is already a good query, return it unchanged.
"""


def _format_history(history: Sequence[ChatTurn]) -> str:
    recent = list(history)[-_HISTORY_TURNS:]
    if not recent:
        return "(none)"
    return "\n".join(f"{t.role}: {t.content[:_HISTORY_TURN_CHARS]}" for t in recent)


def _clean(raw: str) -> str:
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = _PREFIX_RE.sub("", line).strip().strip("\"'`").strip()
    return re.sub(r"\s+", " ", line)[:MAX_QUERY_CHARS]


class QueryRewriter:
    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm

    def rewrite(self, query: str, history: Sequence[ChatTurn] | None = None) -> str:
        original = re.sub(r"\s+", " ", query or "").strip()
        if not original:
            # Return early instead of raising, so empty UI input can't crash the app.
            return ""

        user_prompt = (
            f"Conversation history:\n{_format_history(history or [])}\n\n"
            f"Latest user message:\n<<<{original}>>>"
        )
        try:
            rewritten = _clean(self._llm.complete(system=REWRITER_SYSTEM_PROMPT, user=user_prompt))
        except LLMError:
            logger.warning("Query rewrite failed; falling back to the original query", exc_info=True)
            return original[:MAX_QUERY_CHARS]

        if not rewritten:
            return original[:MAX_QUERY_CHARS]

        # Safety net: identifiers are the highest-value retrieval tokens. Re-attach any
        # the LLM dropped or altered.
        missing = [i for i in extract_identifiers(original) if i.lower() not in rewritten.lower()]
        if missing:
            logger.info("Rewriter dropped identifiers %s; re-attaching", missing)
            rewritten = f"{rewritten} {' '.join(missing)}"

        logger.debug("Rewrote %r -> %r", original, rewritten)
        return rewritten[:MAX_QUERY_CHARS]
