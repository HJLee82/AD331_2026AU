"""
Streamlit UI for the resilient invoice extractor.

Run from the project root:
    uv run streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

# `streamlit run src/app.py` puts src/ on sys.path, not the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.extractor import (  # noqa: E402
    InvoiceExtractor,
    text_from_html,
    text_from_pdf,
)

load_dotenv(ROOT / ".env")
DLQ_DIR = ROOT / "dlq"

st.set_page_config(page_title="Resilient Invoice Extractor", page_icon="🧾", layout="wide")


# --------------------------------------------------------------------------- #
# Clients and tracing
# --------------------------------------------------------------------------- #

@st.cache_resource
def get_client() -> anthropic.Anthropic:
    """One persistent, connection-pooled client for the whole app."""
    return anthropic.Anthropic(max_retries=3, timeout=60.0)


class TracingExtractor(InvoiceExtractor):
    """Records every message list sent to the model, so the UI can show the
    exact corrective prompts dispatched on each attempt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[list[dict[str, str]]] = []

    def _call_llm(self, messages):
        self.calls.append(messages)
        return super()._call_llm(messages)


class ScriptedClient:
    """Fake Anthropic client that replays canned tool outputs. No API call, no cost.
    Used only by the Simulate buttons so the demo is deterministic."""

    def __init__(self, outputs: list[dict]):
        self._outputs = list(outputs)
        self.messages = self  # so `client.messages.create(...)` works

    def create(self, **_kwargs):
        out = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(type="tool_use", input=out)],
        )


# --------------------------------------------------------------------------- #
# Simulation scenarios (clearly labeled as simulated in the UI)
# --------------------------------------------------------------------------- #

SIM_INVOICE_TEXT = """NORTHWIND OFFICE SUPPLY
1200 Pine St, Suite 400
Seattle, WA 98101

Invoice #: INV-2024-0042
Date: 12th of Oct, 2024

Description              Qty   Unit Price   Amount
Copy paper (case)         10      $8.50      $85.00
Toner cartridge           2      $64.25     $128.50

Subtotal                                    $213.50
Tax (8.5%)                                   $18.15
TOTAL DUE                                   $231.65"""

SIM_GOOD = {
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
# Classic transcription mistake: the model copied the tax RATE instead of the amount.
SIM_TAX_RATE_MISTAKE = {**SIM_GOOD, "tax_amount": 8.5}

SIM_NONSENSE_TEXT = """Meeting notes - Q3 offsite
Lorem ipsum dolor sit amet. Bring snacks. Parking is on level 2.
Action items: TBD. Nothing here is an invoice."""

SIM_NONSENSE_OUTPUTS = [
    {"invoice_number": "", "vendor_name": "", "invoice_date": "sometime in Q3", "line_items": [],
     "subtotal": "N/A", "tax_amount": -1, "total_amount": "lots", "currency": "BTC"},
    {"invoice_number": "unknown", "vendor_name": "Q3 offsite", "invoice_date": "TBD", "line_items": [],
     "subtotal": 0, "tax_amount": 0, "total_amount": "n/a", "currency": "USD"},
    {"invoice_number": "none", "vendor_name": "Meeting notes", "invoice_date": "level 2",
     "line_items": [{"description": "snacks", "quantity": 0, "unit_price": 0, "line_total": 0}],
     "subtotal": 0, "tax_amount": 0, "total_amount": 0, "currency": "USD"},
    {"invoice_number": "none", "vendor_name": "Meeting notes", "invoice_date": "parking",
     "line_items": [], "subtotal": None, "tax_amount": None, "total_amount": None, "currency": "???"},
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def read_upload(upload) -> str:
    name = upload.name.lower()
    data = upload.getvalue()
    if name.endswith(".pdf"):
        return text_from_pdf(data)
    text = data.decode("utf-8", errors="replace")
    return text_from_html(text) if name.endswith((".html", ".htm")) else text


def run_pipeline(raw_text: str, client, max_retries: int, label: str) -> None:
    extractor = TracingExtractor(client, dlq_dir=DLQ_DIR)
    result = extractor.extract_with_retry(raw_text, max_retries=max_retries)
    st.session_state["run"] = {
        "label": label,
        "raw_text": raw_text,
        "result": result,
        "calls": extractor.calls,
    }


def reconciliation(data: dict | None) -> tuple[str, str]:
    """Return (level, message) where level is 'ok' | 'bad' | 'unknown'."""
    if not data:
        return "unknown", "No data to reconcile."
    try:
        items = data.get("line_items") or []
        items_sum = round(sum(float(i["line_total"]) for i in items), 2)
        subtotal = float(data["subtotal"])
        tax = float(data["tax_amount"])
        total = float(data["total_amount"])
    except (KeyError, TypeError, ValueError):
        return "unknown", "Not enough valid fields to check the totals."
    checks = [
        (abs(items_sum - subtotal) <= 0.02, f"Line items {items_sum:,.2f} vs subtotal {subtotal:,.2f}"),
        (abs(subtotal + tax - total) <= 0.02, f"Subtotal + tax {subtotal + tax:,.2f} vs total {total:,.2f}"),
    ]
    level = "ok" if all(ok for ok, _ in checks) else "bad"
    msg = "  \n".join(f"{'✅' if ok else '❌'} {text}" for ok, text in checks)
    return level, msg


STATUS_STYLE = {
    "SUCCESS": ("success", "🟢 SUCCESS: valid on the first attempt"),
    "REPAIRED_VIA_REFLECTION": ("info", "🔵 REPAIRED VIA REFLECTION: self-corrected after a validation error"),
    "REPAIRED_VIA_HEURISTIC": ("warning", "🟠 REPAIRED VIA HEURISTIC: fixed by json_repair + normalization"),
    "DLQ_FAILED": ("error", "🔴 DLQ FAILED: sent to the dead-letter queue for human review"),
}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Settings")
    reflection_on = st.toggle("Reflection self-correction loop", value=True)
    max_retries = st.slider("Max reflection retries", 1, 3, 3, disabled=not reflection_on)
    effective_retries = max_retries if reflection_on else 0
    st.caption("Off = one attempt, then straight to json_repair / DLQ.")

    st.divider()
    st.header("🧪 Simulations")
    st.caption("Scripted model responses. No API call, no cost.")
    sim_reflection = st.button("Simulate Reflection Repair", use_container_width=True)
    sim_failure = st.button("Simulate Edge Case Failure", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main: input
# --------------------------------------------------------------------------- #

st.title("🧾 Resilient Invoice Extractor")
st.caption("Messy PDF / HTML → Claude tool use → Pydantic v2 validation → reflection → json_repair → DLQ")

tab_upload, tab_paste = st.tabs(["📄 Upload file", "📋 Paste text / HTML"])
with tab_upload:
    upload = st.file_uploader("PDF, HTML or TXT", type=["pdf", "html", "htm", "txt"])
with tab_paste:
    pasted = st.text_area("Raw invoice text or HTML", height=180)

run_clicked = st.button("▶️ Extract", type="primary")

if run_clicked:
    raw = read_upload(upload) if upload is not None else text_from_html(pasted) if "<" in pasted else pasted
    if not raw.strip():
        st.warning("Upload a file or paste some text first.")
    elif not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is not set. Add it to .env and restart the app.")
    else:
        with st.spinner("Extracting…"):
            run_pipeline(raw, get_client(), effective_retries, label=upload.name if upload else "pasted text")

if sim_reflection:
    run_pipeline(SIM_INVOICE_TEXT, ScriptedClient([SIM_TAX_RATE_MISTAKE, SIM_GOOD]), 3,
                 label="SIMULATED: tax rate copied instead of tax amount")
if sim_failure:
    run_pipeline(SIM_NONSENSE_TEXT, ScriptedClient(SIM_NONSENSE_OUTPUTS), 3,
                 label="SIMULATED: irrecoverable non-invoice input")


# --------------------------------------------------------------------------- #
# Main: results
# --------------------------------------------------------------------------- #

run = st.session_state.get("run")
if run:
    result = run["result"]
    st.divider()
    st.subheader(f"Result: {run['label']}")

    kind, text = STATUS_STYLE[result.execution_status]
    getattr(st, kind)(text)

    m1, m2, m3 = st.columns(3)
    m1.metric("Status", result.execution_status)
    m2.metric("Attempts used", result.attempts_used)
    m3.metric("Model calls", len(run["calls"]))

    data = (
        result.validated_data.model_dump(mode="json")
        if result.validated_data
        else result.partial_data
    )

    # Reconciliation badge
    level, msg = reconciliation(data)
    badge = {"ok": st.success, "bad": st.error, "unknown": st.warning}[level]
    badge(f"**Reconciliation:** {'Totals match' if level == 'ok' else 'Totals do NOT match' if level == 'bad' else 'Unknown'}  \n{msg}")

    # Side-by-side raw vs JSON
    left, right = st.columns(2)
    with left:
        st.markdown("**Raw document text**")
        st.code(run["raw_text"], language=None)
    with right:
        st.markdown("**Validated JSON**" if result.validated_data else "**Partial extraction (failed validation)**")
        if data:
            st.json(data)
        else:
            st.info("Nothing could be salvaged.")
        if result.dlq_path:
            st.caption(f"📥 DLQ record: `{Path(result.dlq_path).name}`")

    # Audit log
    st.subheader("🔍 Audit log")
    with st.expander("Error log (retries, validation traces, fallback)", expanded=True):
        if result.error_log:
            st.code("\n".join(result.error_log), language=None)
        else:
            st.write("No errors. Passed on the first attempt.")

    for i, messages in enumerate(run["calls"], start=1):
        is_reflection = len(messages) > 1
        title = f"Attempt {i}: {'reflection (corrective prompt)' if is_reflection else 'initial extraction'}"
        with st.expander(title):
            if is_reflection:
                st.markdown("**Previous (invalid) JSON sent back to the model**")
                st.code(messages[-2]["content"], language="json")
                st.markdown("**Corrective prompt dispatched**")
                st.code(messages[-1]["content"], language=None)
            else:
                st.markdown("**Prompt sent** (system prompt + document)")
                st.code(messages[0]["content"][:3000], language=None)