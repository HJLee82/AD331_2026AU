"""Streamlit UI for the Unified LLM Gateway.

Run from the project folder:
    uv run streamlit run src/app.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Iterator

import streamlit as st
from dotenv import find_dotenv, load_dotenv

# `streamlit run src/app.py` puts src/ on sys.path, not the project root.
# Add the root so `src` imports as a package (needed for the relative imports in gateway.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gateway import (  # noqa: E402
    AuthenticationError,
    GatewayError,
    RateLimitExceededError,
    UnifiedLLMGateway,
)
from src.pricing import resolve_model, supported_models  # noqa: E402
from src.schemas import StreamChunk  # noqa: E402

# ---------------------------------------------------------------------------
# Keys: local .env first; on Streamlit Cloud, fall back to its Secrets manager.
# Keys are only placed in the server's environment, never sent to the browser.
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv(usecwd=True))
try:
    for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.getenv(_key) and _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:  # no secrets.toml locally: that's fine
    pass


# ---------------------------------------------------------------------------
# One background event loop + one gateway for the whole server process.
# Streamlit reruns this script on every click, so without caching we would
# rebuild the HTTP clients each time (the "client instantiation" anti-pattern).
# ---------------------------------------------------------------------------
@st.cache_resource
def get_runtime() -> tuple[asyncio.AbstractEventLoop, UnifiedLLMGateway]:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="llm-gateway-loop").start()
    return loop, UnifiedLLMGateway()


def iterate_async(agen: AsyncGenerator[StreamChunk, None], loop: asyncio.AbstractEventLoop) -> Iterator[StreamChunk]:
    """Pull items from an async generator running on the background loop.

    Streamlit's script is synchronous; this bridge lets it consume the async
    stream chunk-by-chunk without blocking the event loop that does the I/O.
    """
    try:
        while True:
            future = asyncio.run_coroutine_threadsafe(agen.__anext__(), loop)
            try:
                yield future.result()
            except StopAsyncIteration:
                return
    finally:
        asyncio.run_coroutine_threadsafe(agen.aclose(), loop).result()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Provider LLM Gateway", page_icon="⚡", layout="wide")
st.title("⚡ Multi-Provider LLM Gateway")
st.caption("One async interface for OpenAI and Anthropic, with live token and cost telemetry.")

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Sidebar: provider, model, temperature --------------------------------
MODELS: dict[str, list[str]] = {"openai": [], "anthropic": []}
for name in supported_models():
    MODELS[resolve_model(name).provider].append(name)

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Provider", list(MODELS), format_func=lambda p: "OpenAI" if p == "openai" else "Anthropic")
    model = st.selectbox("Model", MODELS[provider])
    resolved = resolve_model(model)

    max_temp = 2.0 if provider == "openai" else 1.0
    temperature = st.slider(
        "Temperature",
        0.0,
        max_temp,
        min(0.7, max_temp),
        0.1,
        disabled=not resolved.supports_temperature,
        help="Claude 5.x models don't accept temperature, so it's disabled for them.",
    )
    if not resolved.supports_temperature:
        st.caption("ℹ️ This model ignores temperature.")

    p = resolved.pricing
    st.divider()
    st.markdown(f"**Price** (per 1M tokens)  \nInput ${p.input_per_mtok} · Output ${p.output_per_mtok}")

# ---- Main: prompt, streamed answer, live metrics --------------------------
prompt = st.text_area("Prompt", "Explain TCP slow start in two sentences.", height=110)
run = st.button("▶ Run", type="primary", disabled=not prompt.strip())

col_text, col_metrics = st.columns([3, 2], gap="large")
with col_text:
    st.subheader("Response")
    answer_box = st.empty()
with col_metrics:
    st.subheader("Telemetry")
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    ttft_box, tps_box = m1.empty(), m2.empty()
    tokens_box, cost_box = m3.empty(), m4.empty()
    status_box = st.empty()


def show_metrics(ttft: str, tps: str, tokens: str, cost: str) -> None:
    ttft_box.metric("TTFT", ttft)
    tps_box.metric("Speed", tps)
    tokens_box.metric("Tokens (prompt / completion)", tokens)
    cost_box.metric("Cost (USD)", cost)


show_metrics("–", "–", "–", "–")

if run:
    loop, gateway = get_runtime()
    text = ""
    chunks_seen = 0
    started = time.perf_counter()
    first_at: float | None = None

    try:
        stream = gateway.stream(prompt, model=model, temperature=float(temperature))
        for chunk in iterate_async(stream, loop):
            if chunk.is_final:
                t = chunk.telemetry
                assert t is not None
                answer_box.markdown(text)
                show_metrics(
                    f"{t.ttft_ms:,.0f} ms" if t.ttft_ms is not None else "–",
                    f"{t.tokens_per_second:,.1f} tok/s",
                    f"{t.input_tokens} / {t.output_tokens}",
                    f"${t.total_cost_usd:.6f}",
                )
                status_box.success(f"✅ Done in {t.latency_s:.2f}s · counts from {t.usage_source}")
                st.session_state.history.insert(0, {
                    "Model": chunk.model,
                    "TTFT (ms)": round(t.ttft_ms or 0),
                    "Tok/s": t.tokens_per_second,
                    "Prompt tok": t.input_tokens,
                    "Completion tok": t.output_tokens,
                    "Cost (USD)": f"{t.total_cost_usd:.6f}",
                })
            else:
                if first_at is None:
                    first_at = time.perf_counter()
                text += chunk.delta_text
                chunks_seen += 1
                answer_box.markdown(text + " ▌")
                # Live estimates while streaming; replaced by exact provider numbers on the final chunk.
                elapsed = time.perf_counter() - first_at
                show_metrics(
                    f"{(first_at - started) * 1000:,.0f} ms",
                    f"~{chunks_seen / elapsed:,.0f} chunks/s" if elapsed > 0 else "–",
                    "streaming…",
                    "calculating…",
                )
    except AuthenticationError as e:
        status_box.error(f"🔑 Authentication problem: {e}")
    except RateLimitExceededError as e:
        wait = f" Retry in {e.retry_after:.0f}s." if e.retry_after else ""
        status_box.warning(f"⏳ Rate limit reached.{wait}")
    except GatewayError as e:
        status_box.error(f"⚠️ {type(e).__name__}: {e}")

# ---- Run history: compare providers side by side --------------------------
if st.session_state.history:
    st.divider()
    st.subheader("📊 Run history")
    st.dataframe(st.session_state.history, width="stretch", hide_index=True)
