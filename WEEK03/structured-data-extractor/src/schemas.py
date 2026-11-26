"""
Pydantic v2 data contracts for invoice extraction.

Design principle
----------------
Two layers of enforcement:

* **Wire schema** (``to_strict_json_schema``) – what we send to OpenAI's
  constrained decoder. It guarantees *shape* (keys, types, enums) and is
  deliberately relaxed: numeric bounds, string formats and length limits are
  stripped, because unsupported keywords make the API reject the whole request.
* **Pydantic models** (below) – the source of truth. They enforce *semantics*
  (quantity > 0, ISO dates, arithmetic consistency). Violations surface as
  ``ValidationError`` and drive the reflection-repair loop in ``extractor.py``.
"""

from __future__ import annotations

import copy
import math
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Absolute tolerance (in currency units) for the totals cross-check.
TOTAL_TOLERANCE: float = 0.02

Currency = Literal["USD", "EUR", "GBP"]

ExecutionStatus = Literal[
    "SUCCESS",                  # valid on the first model response
    "REPAIRED_VIA_REFLECTION",  # valid after >= 1 corrective re-prompt
    "REPAIRED_VIA_HEURISTIC",   # valid only after json_repair + normalisation
    "DLQ_FAILED",               # unrecoverable; record written to the DLQ
]


class _StrictBase(BaseModel):
    """Shared config: reject unknown keys, NaN/inf, and trim strings."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class LineItem(_StrictBase):
    description: str = Field(
        min_length=1,
        description="Item description as printed. Merge wrapped continuation lines.",
    )
    quantity: int = Field(gt=0, description="Billed quantity (positive integer).")
    unit_price: float = Field(
        gt=0, description="Price per unit as a plain number, e.g. 1249.5 (no symbols)."
    )
    line_total: float = Field(
        description="Line total exactly as printed on the document, as a plain number."
    )


class InvoiceExtraction(_StrictBase):
    invoice_number: str = Field(min_length=1, description="Invoice identifier as printed.")
    vendor_name: str = Field(min_length=1, description="Name of the issuing vendor.")
    invoice_date: date = Field(
        description="Invoice ISSUE date in ISO 8601 format YYYY-MM-DD "
        "(not due date, ship date or project date)."
    )
    line_items: list[LineItem] = Field(
        min_length=1, description="Billable rows only; exclude subtotal/tax/total rows."
    )
    subtotal: float = Field(description="Subtotal as printed.")
    tax_amount: float = Field(description="Tax AMOUNT as printed (not the rate).")
    total_amount: float = Field(description="Grand total / amount due as printed.")
    currency: Currency = Field(description="ISO 4217 code the totals are expressed in.")

    @model_validator(mode="after")
    def _check_totals(self) -> "InvoiceExtraction":
        """sum(line_total) + tax_amount must equal total_amount (± TOTAL_TOLERANCE).

        The error message contains the computed numbers so the reflection
        prompt can tell the model exactly *where* the arithmetic breaks.
        """
        items_sum = math.fsum(item.line_total for item in self.line_items)
        expected = items_sum + self.tax_amount
        diff = self.total_amount - expected
        # Small epsilon so a diff of exactly 0.02 isn't rejected by float noise.
        if abs(diff) > TOTAL_TOLERANCE + 1e-9:
            raise ValueError(
                f"Arithmetic mismatch: sum(line_items[*].line_total)={items_sum:.2f} "
                f"+ tax_amount={self.tax_amount:.2f} = {expected:.2f}, but "
                f"total_amount={self.total_amount:.2f} (difference {diff:+.2f}, "
                f"tolerance ±{TOTAL_TOLERANCE:.2f})"
            )
        return self


class ExtractionResult(BaseModel):
    """Outcome envelope returned by ``extract_with_retry``. Never raises."""

    validated_data: Optional[InvoiceExtraction] = None
    attempts_used: int = Field(ge=0, description="Number of LLM calls made.")
    error_log: list[str] = Field(default_factory=list)
    execution_status: ExecutionStatus
    # Extensions beyond the minimal contract – useful for triage / replay.
    partial_data: Optional[dict[str, Any]] = None
    dlq_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def format_validation_errors(exc: ValidationError) -> list[str]:
    """Flatten a ValidationError into ``"<loc>: <msg> (got: <input>)"`` lines.

    ``loc`` tuples like ``('line_items', 2, 'quantity')`` become
    ``line_items[2].quantity``; model-level errors get ``<root>``.
    """
    lines: list[str] = []
    for err in exc.errors(include_url=False):
        loc = _format_loc(err.get("loc", ()))
        msg = err.get("msg", "invalid")
        line = f"{loc}: {msg}"
        if "input" in err and err.get("type") != "model_type" and loc != "<root>":
            shown = repr(err["input"])
            line += f" (got: {shown[:120]}{'…' if len(shown) > 120 else ''})"
        lines.append(line)
    return lines


def _format_loc(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out or "<root>"


# Keywords removed from the wire schema; Pydantic enforces them locally instead.
_RELAXED_KEYWORDS = frozenset(
    {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "minItems", "maxItems", "pattern", "format", "default",
    }
)
# Keys whose values are maps of *names* (field names / def names), not schemas.
_NAME_MAP_KEYS = frozenset({"properties", "$defs", "definitions"})


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Derive an OpenAI ``strict: true`` compatible JSON schema from a model.

    Strict mode requires every object to declare ``additionalProperties: false``
    and to list *all* properties in ``required``.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node: Any, is_name_map: bool = False) -> None:
        if isinstance(node, dict):
            if is_name_map:
                for child in node.values():
                    walk(child)
                return
            for kw in _RELAXED_KEYWORDS:
                node.pop(kw, None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for key, child in node.items():
                walk(child, is_name_map=key in _NAME_MAP_KEYS)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(schema)
    return schema