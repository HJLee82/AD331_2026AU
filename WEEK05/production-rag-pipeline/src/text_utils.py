"""Small, dependency-free text helpers shared across stages."""

from __future__ import annotations

import math
import re
import unicodedata

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
_SEP_RE = re.compile(r"[-_.]")
_UNIT_RE = re.compile(r"^\d+(?:st|nd|rd|th|s|m|h|d|ms|kb|mb|gb|tb|k)$", re.IGNORECASE)


def extract_identifiers(text: str, *, limit: int = 5) -> list[str]:
    """Return exact-match identifiers such as ``NIM-E4021``, ``tf-nimbus-v1``, ``p99``.

    Dense embeddings are notoriously weak at matching opaque identifiers, so
    these tokens are used for (a) lexical boosting in Stage 1 and (b) verifying
    the query rewriter did not drop or mangle them.
    """
    found: dict[str, str] = {}
    for token in _TOKEN_RE.findall(text):
        if len(token) < 3 or _UNIT_RE.match(token):
            continue
        has_digit = any(c.isdigit() for c in token)
        has_alpha = any(c.isalpha() for c in token)
        seps = len(_SEP_RE.findall(token))
        if (has_digit and has_alpha) or seps >= 2 or (has_digit and seps >= 1):
            found.setdefault(token.lower(), token)
            if len(found) >= limit:
                break
    return list(found.values())


def slugify(text: str, *, max_len: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "section"


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
