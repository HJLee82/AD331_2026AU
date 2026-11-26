"""Shared test helpers: a fake Anthropic client and known-good invoice data.

The fake client replays scripted tool outputs, so tests are fast, free and
deterministic (no API key, no network).
"""

import copy
from types import SimpleNamespace

import pytest

from src.extractor import InvoiceExtractor

GOOD_INVOICE = {
    "invoice_number": "INV-2024-0042",
    "vendor_name": "Northwind Office Supply",
    "invoice_date": "2024-10-12",
    "line_items": [
        {"description": "Copy paper (case)", "quantity": 10, "unit_price": 8.50, "line_total": 85.00},
        {"description": "Toner cartridge", "quantity": 2, "unit_price": 64.25, "line_total": 128.50},
    ],
    "subtotal": 213.50,
    "tax_amount": 18.15,
    "total_amount": 231.65,
    "currency": "USD",
}


class FakeClient:
    """Mimics `anthropic.Anthropic().messages.create(...)` with canned tool outputs."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []          # every kwargs dict sent to create()
        self.messages = self     # so client.messages.create(...) works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(type="tool_use", input=out)],
        )


@pytest.fixture
def good_invoice():
    return copy.deepcopy(GOOD_INVOICE)


@pytest.fixture
def make_extractor(tmp_path):
    """Build an extractor around a FakeClient; DLQ goes to a temp folder; no backoff sleep."""

    def _make(outputs):
        client = FakeClient(outputs)
        extractor = InvoiceExtractor(client, dlq_dir=tmp_path / "dlq", sleep=lambda _s: None)
        return extractor, client

    return _make