"""Tests 4–5: reflection self-healing and DLQ routing on hard failures."""

import json

from src.extractor import _redact

NONSENSE_TEXT = "Meeting notes. Lorem ipsum dolor sit amet. Bring snacks. Not an invoice."

NONSENSE_OUTPUTS = [
    {"invoice_number": "", "vendor_name": "", "invoice_date": "sometime", "line_items": [],
     "subtotal": "N/A", "tax_amount": -1, "total_amount": "lots", "currency": "BTC"},
    {"invoice_number": "unknown", "vendor_name": "notes", "invoice_date": "TBD", "line_items": [],
     "subtotal": 0, "tax_amount": 0, "total_amount": "n/a", "currency": "USD"},
    {"invoice_number": "none", "vendor_name": "notes", "invoice_date": "level 2",
     "line_items": [{"description": "snacks", "quantity": 0, "unit_price": 0, "line_total": 0}],
     "subtotal": 0, "tax_amount": 0, "total_amount": 0, "currency": "USD"},
    {"invoice_number": "none", "vendor_name": "notes", "invoice_date": "parking", "line_items": [],
     "subtotal": None, "tax_amount": None, "total_amount": None, "currency": "???"},
]


# --------------------------------------------------------------------------- #
# Test 4: Reflection loop success
# --------------------------------------------------------------------------- #

def test_reflection_self_heals_on_attempt_2(make_extractor, good_invoice):
    """Bad first answer (tax rate instead of amount, missing currency) → fixed on attempt 2."""
    bad = {**good_invoice, "tax_amount": 8.5}
    del bad["currency"]
    extractor, client = make_extractor([bad, good_invoice])

    result = extractor.extract_with_retry("INVOICE INV-2024-0042 ...", max_retries=3)

    assert result.execution_status == "REPAIRED_VIA_REFLECTION"
    assert result.attempts_used == 2
    assert result.validated_data.tax_amount == 18.15

    # The corrective prompt must carry the exact Pydantic error locations.
    corrective_prompt = client.calls[1]["messages"][-1]["content"]
    assert "currency" in corrective_prompt
    # ...and the previous bad JSON is sent back as the assistant turn.
    assert json.loads(client.calls[1]["messages"][-2]["content"])["tax_amount"] == 8.5


# --------------------------------------------------------------------------- #
# Test 5: DLQ routing on hard failures
# --------------------------------------------------------------------------- #

def test_irrecoverable_input_goes_to_dlq(make_extractor, tmp_path):
    """Nonsense input → 1 + 3 retries → json_repair fails → DLQ_FAILED, record on disk, no crash."""
    extractor, client = make_extractor(NONSENSE_OUTPUTS)

    result = extractor.extract_with_retry(NONSENSE_TEXT, max_retries=3)

    assert result.execution_status == "DLQ_FAILED"
    assert result.attempts_used == 4
    assert len(client.calls) == 4
    assert result.validated_data is None

    dlq_files = list((tmp_path / "dlq").glob("*.json"))
    assert len(dlq_files) == 1
    record = json.loads(dlq_files[0].read_text())
    assert record["dlq_failed"] is True
    assert record["raw_text"] == NONSENSE_TEXT
    assert str(dlq_files[0]) == result.dlq_path


# --------------------------------------------------------------------------- #
# Extra: audit-fix regression tests
# --------------------------------------------------------------------------- #

def test_reflection_prompt_is_bounded(make_extractor, good_invoice):
    """Audit fix 1: retries must not resend a huge document in full."""
    huge_doc = "Line item filler text. " * 5_000  # ~115k chars
    bad = {**good_invoice, "tax_amount": 8.5}
    extractor, client = make_extractor([bad, good_invoice])

    extractor.extract_with_retry(huge_doc, max_retries=3)

    first_doc = client.calls[0]["messages"][0]["content"]
    retry_doc = client.calls[1]["messages"][0]["content"]
    assert len(retry_doc) < len(first_doc)
    assert len(retry_doc) < 7_000


def test_api_keys_are_redacted():
    """Audit fix 3: keys never reach error_log / the UI."""
    msg = _redact("Error 401: invalid x-api-key sk-ant-api03-AbCdEf123456 provided")
    assert "AbCdEf123456" not in msg
    assert "REDACTED" in msg