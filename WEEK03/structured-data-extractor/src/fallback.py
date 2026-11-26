"""
Stage 3 – heuristic fallback and Dead-Letter Queue (DLQ).

Runs only after the LLM retry budget is spent (or the loop stopped early).
It never calls the model again and never *invents* data:

1. ``json_repair`` the last raw payload (truncated JSON, trailing commas, …).
2. Normalise loosely-typed values ("$1,249.50" → 1249.5, "€" → "EUR",
   "12th of Oct, 2024" → "2024-10-12").
3. Re-validate against ``InvoiceExtraction``. Pass → REPAIRED_VIA_HEURISTIC.
4. Otherwise salvage whatever fields validate individually (partial
   extraction) and write a DLQ record for human review / replay.

Arithmetic is never "fixed" here: a document whose printed totals don't add
up is a business exception and belongs in the DLQ, not in the ledger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import json_repair
from dateutil import parser as date_parser
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from .schemas import InvoiceExtraction, LineItem, format_validation_errors

logger = logging.getLogger(__name__)

_INVOICE_MONEY_FIELDS = frozenset({"subtotal", "tax_amount", "total_amount"})
_ITEM_MONEY_FIELDS = frozenset({"unit_price", "line_total"})

_CURRENCY_ALIASES = {
    "$": "USD", "US$": "USD", "USD": "USD", "DOLLAR": "USD", "DOLLARS": "USD",
    "€": "EUR", "EUR": "EUR", "EURO": "EUR", "EUROS": "EUR",
    "£": "GBP", "GBP": "GBP", "POUND": "GBP", "POUNDS": "GBP",
}


@dataclass(slots=True)
class FallbackOutcome:
    invoice: Optional[InvoiceExtraction]
    partial: Optional[dict[str, Any]]
    dlq_path: Optional[Path]


# --------------------------------------------------------------------------- #
# Value normalisers
# --------------------------------------------------------------------------- #

def parse_money(value: Any) -> Any:
    """Best-effort conversion of a money string to float; else return as-is.

    Handles symbols/codes, thousands separators, decimal commas and
    accounting negatives:
        "$1,249.50" → 1249.5    "USD 300" → 300.0    "€ 89,00" → 89.0
        "1.234,56"  → 1234.56   "(15.00)" → -15.0
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    digits = re.sub(r"[^\d.,]", "", s)
    if not re.search(r"\d", digits):
        return value

    if "," in digits and "." in digits:
        # Whichever separator appears last is the decimal separator.
        if digits.rfind(",") > digits.rfind("."):
            digits = digits.replace(".", "").replace(",", ".")
        else:
            digits = digits.replace(",", "")
    elif "," in digits:
        # "89,00" → decimal comma; "1,249" / "1,000,000" → thousands.
        digits = digits.replace(",", ".") if re.search(r",\d{1,2}$", digits) else digits.replace(",", "")
    try:
        number = float(digits)
    except ValueError:
        return value
    return -number if negative else number


def parse_loose_date(value: Any) -> Any:
    """Convert free-form dates ("12th of Oct, 2024", "10/03/24") to ISO strings.

    Ambiguous numeric dates are read month-first (US convention), matching the
    extraction prompt. Returns the input unchanged if it can't be parsed.
    """
    if not isinstance(value, str):
        return value
    try:
        date.fromisoformat(value.strip())
        return value.strip()
    except ValueError:
        pass
    try:
        return date_parser.parse(value, fuzzy=True, dayfirst=False).date().isoformat()
    except (ValueError, OverflowError):
        return value


def _parse_quantity(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", value)
        try:
            as_float = float(cleaned)
        except ValueError:
            return value
        return int(as_float) if as_float.is_integer() else as_float
    return value


def normalize_payload(obj: dict[str, Any]) -> dict[str, Any]:
    """Coerce loosely-typed values into the shapes ``InvoiceExtraction`` expects."""
    out = dict(obj)
    for name in _INVOICE_MONEY_FIELDS & out.keys():
        out[name] = parse_money(out[name])
    if "invoice_date" in out:
        out["invoice_date"] = parse_loose_date(out["invoice_date"])
    if isinstance(out.get("currency"), str):
        key = out["currency"].strip().upper()
        out["currency"] = _CURRENCY_ALIASES.get(key, key)
    if isinstance(out.get("line_items"), list):
        items = []
        for item in out["line_items"]:
            if isinstance(item, dict):
                item = dict(item)
                for name in _ITEM_MONEY_FIELDS & item.keys():
                    item[name] = parse_money(item[name])
                if "quantity" in item:
                    item["quantity"] = _parse_quantity(item["quantity"])
            items.append(item)
        out["line_items"] = items
    return out


# --------------------------------------------------------------------------- #
# Partial extraction
# --------------------------------------------------------------------------- #

def extract_partial(obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate each field independently and keep the ones that pass.

    The cross-field totals check is skipped by design – that's what failed.
    Returns ``(partial_json_ready_dict, notes)``.
    """
    partial: dict[str, Any] = {}
    notes: list[str] = []

    for name, field in InvoiceExtraction.model_fields.items():
        if name == "line_items":
            continue
        if name not in obj:
            notes.append(f"partial: {name} missing")
            continue
        # Re-attach Field constraints (gt, min_length, …) stored in metadata.
        annotation = (
            Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
        )
        adapter = TypeAdapter(annotation)
        try:
            partial[name] = adapter.validate_python(obj[name])
        except ValidationError as exc:
            notes.append(f"partial: {name} rejected – {exc.errors()[0]['msg']}")

    raw_items = obj.get("line_items")
    if isinstance(raw_items, list):
        good: list[LineItem] = []
        for idx, item in enumerate(raw_items):
            try:
                good.append(LineItem.model_validate(item))
            except ValidationError as exc:
                for line in format_validation_errors(exc):
                    notes.append(f"partial: line_items[{idx}].{line}")
        partial["line_items"] = good
        notes.append(f"partial: kept {len(good)}/{len(raw_items)} line items")

    return to_jsonable_python(partial), notes


# --------------------------------------------------------------------------- #
# DLQ
# --------------------------------------------------------------------------- #

def write_dlq_record(
    dlq_dir: Path,
    *,
    raw_text: str,
    last_payload: Optional[str],
    error_log: list[str],
    partial: Optional[dict[str, Any]],
    attempts_used: int,
) -> Path:
    """Write one self-contained JSON record per failure, atomically.

    One file per record (instead of an appended JSONL) avoids interleaved
    writes between concurrent workers; ``os.replace`` makes it atomic.
    NOTE: records contain the full source text – treat the DLQ directory as
    sensitive (PII/financial data) and apply your retention policy to it.
    """
    dlq_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    record_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    record = {
        "dlq_failed": True,
        "record_id": record_id,
        "created_at": now.isoformat(),
        "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "attempts_used": attempts_used,
        "error_log": error_log,
        "last_payload_raw": last_payload,
        "partial_extraction": partial,
        "raw_text": raw_text,
    }
    final_path = dlq_dir / f"{record_id}.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)
    return final_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_heuristic_fallback(
    *,
    raw_text: str,
    last_payload: Optional[str],
    error_log: list[str],
    attempts_used: int,
    dlq_dir: Path,
) -> FallbackOutcome:
    """Stage 3 entry point. Mutates ``error_log`` in place; never raises."""
    repaired: Optional[dict[str, Any]] = None

    if last_payload:
        try:
            candidate = json_repair.loads(last_payload)
        except Exception as exc:  # json_repair is best-effort; never let it crash us
            error_log.append(f"[heuristic] json_repair crashed: {exc!r}")
            candidate = None
        if isinstance(candidate, dict) and candidate:
            repaired = normalize_payload(candidate)
        else:
            error_log.append("[heuristic] json_repair could not recover a JSON object")
    else:
        error_log.append("[heuristic] no model payload available to repair")

    partial: Optional[dict[str, Any]] = None
    if repaired is not None:
        try:
            invoice = InvoiceExtraction.model_validate(repaired)
            error_log.append("[heuristic] payload validated after json_repair + normalisation")
            return FallbackOutcome(invoice=invoice, partial=None, dlq_path=None)
        except ValidationError as exc:
            error_log.extend(f"[heuristic] {line}" for line in format_validation_errors(exc))
            partial, notes = extract_partial(repaired)
            error_log.extend(notes)

    try:
        path = write_dlq_record(
            dlq_dir,
            raw_text=raw_text,
            last_payload=last_payload,
            error_log=list(error_log),
            partial=partial,
            attempts_used=attempts_used,
        )
        error_log.append(f"[dlq] record written to {path}")
        logger.warning("Invoice extraction sent to DLQ: %s", path)
    except OSError as exc:
        # Disk/permission failure must not take the pipeline down; surface loudly.
        path = None
        error_log.append(f"[dlq] FAILED to write DLQ record: {exc!r}")
        logger.critical("DLQ write failed – record lost unless caller persists result: %r", exc)

    return FallbackOutcome(invoice=None, partial=partial, dlq_path=path)