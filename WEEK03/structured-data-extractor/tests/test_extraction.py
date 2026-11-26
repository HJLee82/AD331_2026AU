"""Tests 1–3: golden extraction, schema constraints, cross-field reconciliation."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.extractor import load_document_text
from src.schemas import InvoiceExtraction

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PDF = ROOT / "samples" / "invoice_golden.pdf"


# --------------------------------------------------------------------------- #
# Test 1: Golden document extraction
# --------------------------------------------------------------------------- #

def test_golden_document_passes_on_first_attempt(make_extractor, good_invoice):
    """A well-formed PDF is read and validated on attempt 1 with zero errors."""
    raw_text = load_document_text(GOLDEN_PDF)
    assert raw_text.strip(), "golden PDF produced no text (saved as an image?)"

    extractor, client = make_extractor([good_invoice])
    result = extractor.extract_with_retry(raw_text)

    assert result.execution_status == "SUCCESS"
    assert result.attempts_used == 1
    assert result.error_log == []
    assert result.validated_data.invoice_number == "INV-2024-0042"
    assert len(client.calls) == 1


@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API")
def test_golden_document_live_api():
    """Optional: the same golden PDF against the real Anthropic API (costs a few cents)."""
    from dotenv import load_dotenv

    from src.extractor import extract_with_retry

    load_dotenv(ROOT / ".env")
    result = extract_with_retry(load_document_text(GOLDEN_PDF))
    assert result.execution_status == "SUCCESS", result.error_log


# --------------------------------------------------------------------------- #
# Test 2: Schema constraint assertions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "mutate, bad_field",
    [
        (lambda d: d["line_items"][0].update(quantity=0), "quantity"),
        (lambda d: d["line_items"][0].update(quantity=-3), "quantity"),
        (lambda d: d.update(total_amount=-231.65), "total_amount"),
        (lambda d: d.update(subtotal=-1), "subtotal"),
    ],
    ids=["quantity_zero", "quantity_negative", "negative_total", "negative_subtotal"],
)
def test_schema_rejects_invalid_values(good_invoice, mutate, bad_field):
    mutate(good_invoice)
    with pytest.raises(ValidationError) as exc_info:
        InvoiceExtraction.model_validate(good_invoice)
    failed_fields = {str(part) for err in exc_info.value.errors() for part in err["loc"]}
    assert bad_field in failed_fields


# --------------------------------------------------------------------------- #
# Test 3: Cross-field reconciliation
# --------------------------------------------------------------------------- #

def test_totals_mismatch_fails_validation(good_invoice):
    """Line items + tax do not add up to total_amount → rejected."""
    good_invoice["total_amount"] = 246.65  # off by $15, like invoice_messy.pdf
    with pytest.raises(ValidationError, match="total_amount"):
        InvoiceExtraction.model_validate(good_invoice)


def test_subtotal_mismatch_fails_validation(good_invoice):
    """Subtotal that doesn't equal the sum of line totals → rejected."""
    good_invoice["subtotal"] = 200.00
    good_invoice["total_amount"] = 218.15  # keep subtotal + tax == total so only subtotal is wrong
    with pytest.raises(ValidationError, match="subtotal"):
        InvoiceExtraction.model_validate(good_invoice)


def test_messy_strings_are_cleaned(good_invoice):
    """Before-validators turn messy strings into clean types."""
    good_invoice.update(invoice_date="12th of Oct, 2024", total_amount="$231.65", currency="$")
    good_invoice["line_items"][1]["line_total"] = "$128.50"
    inv = InvoiceExtraction.model_validate(good_invoice)
    assert inv.invoice_date.isoformat() == "2024-10-12"
    assert inv.total_amount == 231.65
    assert inv.currency == "USD"