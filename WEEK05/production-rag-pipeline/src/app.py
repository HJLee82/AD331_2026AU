"""Streamlit UI for the Nimbus Cloud two-stage RAG pipeline.

Run from the project root:
    uv run streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import ConfigError, Settings  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.pipeline import PipelineTrace, TwoStageRAGPipeline  # noqa: E402
from src.schemas import ChatTurn, RAGResponse, RetrievedChunk  # noqa: E402

STAGE2_SHOWN = 5  # rows in the Stage 2 table (assignment: top 5)

st.set_page_config(page_title="Nimbus RAG", page_icon="☁️", layout="wide")


# --------------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading models and index…")
def load_pipeline() -> tuple[TwoStageRAGPipeline, Settings]:
    from src.retriever import ChromaRetriever

    settings = Settings.from_env(ROOT / ".env")
    retriever = ChromaRetriever(settings)
    retriever.index()  # no-op when the corpus fingerprint is unchanged
    return TwoStageRAGPipeline.from_settings(settings, retriever=retriever), settings


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # [{role, content, response?, trace?}]
    st.session_state.setdefault("history", [])  # list[ChatTurn] for the rewriter


# ------------------------------------------------------------------- helpers
def short_ref(chunk_id: str) -> str:
    """'deployment_runbook.md::6-error-code-reference::015' -> 'deployment_runbook.md › 6-error-code-reference'."""
    if chunk_id.startswith(("http://", "https://")):
        return chunk_id
    parts = chunk_id.split("::")
    return " › ".join(parts[:2]) if len(parts) >= 2 else chunk_id


def route_badge(route: str) -> str:
    if route == "INTERNAL_RAG":
        return ":green-background[🟢 **INTERNAL_RAG** · High Confidence]"
    return ":orange-background[🟠 **WEB_FALLBACK** · Low Confidence Triggered]"


def confidence_meter(score: float, threshold: float) -> None:
    pct = max(0.0, min(1.0, score)) * 100
    color = "#1d9a52" if score >= threshold else "#e07b00"
    st.markdown(
        f"""
<div style="position:relative;height:18px;border-radius:9px;background:rgba(128,128,128,.2);margin:6px 0 4px">
  <div style="width:{pct:.1f}%;height:100%;border-radius:9px;background:{color}"></div>
  <div title="threshold" style="position:absolute;top:-4px;left:{threshold * 100:.1f}%;width:2px;height:26px;background:currentColor"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.75">
  <span>0.0</span><span>threshold {threshold:.2f}</span><span>1.0</span>
</div>""",
        unsafe_allow_html=True,
    )
    verdict = "≥ threshold → internal docs" if score >= threshold else "< threshold → web fallback"
    st.caption(f"Top cross-encoder confidence **{score:.3f}** ({verdict})")


def stage1_table(chunks: list[RetrievedChunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": i,
                "chunk": short_ref(c.chunk_id),
                "cosine similarity": round(c.score, 4),
                "cosine distance": round(1.0 - c.score, 4),
            }
            for i, c in enumerate(chunks, start=1)
        ]
    )


def stage2_table(ranked: list[RetrievedChunk], stage1: list[RetrievedChunk], k: int) -> pd.DataFrame:
    before = {c.chunk_id: i for i, c in enumerate(stage1, start=1)}
    rows = []
    for new_rank, c in enumerate(ranked[:k], start=1):
        old_rank = before.get(c.chunk_id)
        if old_rank is None:
            shift = "new"
        elif old_rank > new_rank:
            shift = f"▲ {old_rank - new_rank}"
        elif old_rank < new_rank:
            shift = f"▼ {new_rank - old_rank}"
        else:
            shift = "—"
        rows.append(
            {
                "rank": new_rank,
                "stage 1 rank": old_rank,
                "shift": shift,
                "chunk": short_ref(c.chunk_id),
                "cross-encoder score": round(c.score, 4),
            }
        )
    return pd.DataFrame(rows)


def render_citations(response: RAGResponse, trace: PipelineTrace) -> None:
    if not response.citations:
        return
    st.markdown("**Sources**")
    cols = st.columns(min(len(response.citations), 3))
    for i, citation in enumerate(response.citations):
        number = citation.split("]", 1)[0].lstrip("[")
        idx = int(number) - 1 if number.isdigit() else -1
        chunk = trace.context_chunks[idx] if 0 <= idx < len(trace.context_chunks) else None
        label = citation if chunk is None else f"[{number}] {short_ref(chunk.chunk_id)}"
        with cols[i % len(cols)].popover(label, width="stretch"):
            if chunk is None:
                st.write(citation)
            elif chunk.source == "web":
                st.markdown(f"🌐 [{chunk.chunk_id}]({chunk.chunk_id})")
                st.markdown(chunk.content)
            else:
                st.markdown(f"📄 `{chunk.chunk_id}`")
                st.markdown(chunk.content)


def render_details(trace: PipelineTrace, settings: Settings) -> None:
    with st.expander(f"Stage 1 · Bi-encoder candidates (top {len(trace.stage1_chunks)})"):
        if trace.stage1_chunks:
            st.dataframe(stage1_table(trace.stage1_chunks), hide_index=True, width="stretch")
        else:
            st.write("No candidates.")

    with st.expander(f"Stage 2 · Cross-encoder re-ranked (top {STAGE2_SHOWN})", expanded=True):
        if trace.stage2_ranked:
            st.dataframe(
                stage2_table(trace.stage2_ranked, trace.stage1_chunks, STAGE2_SHOWN),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"Scores are sigmoid(logit). The top {settings.rerank_top_k} are sent to the "
                f"generator when the top score is ≥ {settings.confidence_threshold:.2f}."
            )
        else:
            st.write("Nothing to re-rank.")

    if trace.route == "WEB_FALLBACK":
        with st.expander(f"Web fallback results ({len(trace.web_chunks)})", expanded=True):
            if trace.web_chunks:
                for i, c in enumerate(trace.web_chunks, start=1):
                    st.markdown(f"**[{i}]** [{c.chunk_id}]({c.chunk_id}) · relevance `{c.score:.3f}`")
            else:
                st.write("Web search returned no usable results.")


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.header("☁️ Nimbus RAG")
        st.toggle(
            "Conversational follow-ups",
            key="conversational",
            value=True,
            help="On: earlier turns are sent to the query rewriter so pronouns like "
            "'it' or 'that' get resolved. Off: every question is standalone.",
        )
        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.subheader("Pipeline state")
        last = next(
            (m for m in reversed(st.session_state.messages) if m.get("trace") is not None), None
        )
        if last is None:
            st.caption("Ask a question to see the pipeline telemetry.")
            return

        response: RAGResponse = last["response"]
        trace: PipelineTrace = last["trace"]
        st.markdown("**Transformed query**")
        st.code(response.transformed_query or "(empty)", language=None, wrap_lines=True)
        st.markdown("**Route**")
        st.markdown(route_badge(response.route_taken))
        st.markdown("**Confidence**")
        confidence_meter(response.confidence_score, settings.confidence_threshold)
        if trace.route == "WEB_FALLBACK":
            st.caption(f"Best web snippet relevance: {trace.top_web_relevance:.3f}")
        if trace.timings_ms:
            st.markdown("**Latency (ms)**")
            st.dataframe(
                pd.DataFrame([{"stage": k, "ms": v} for k, v in trace.timings_ms.items()]),
                hide_index=True,
                width="stretch",
            )


def render_assistant(message: dict, settings: Settings) -> None:
    if "error" in message:
        st.error(message["error"])
        return
    response: RAGResponse = message["response"]
    trace: PipelineTrace = message["trace"]
    st.markdown(route_badge(response.route_taken))
    st.markdown(response.answer)
    render_citations(response, trace)
    render_details(trace, settings)


# ---------------------------------------------------------------------- main
def main() -> None:
    init_state()
    try:
        pipeline, settings = load_pipeline()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}. Check your .env file.")
        st.stop()

    render_sidebar(settings)

    st.title("Nimbus Cloud Knowledge Assistant")
    st.caption(
        "Two-stage RAG: query rewrite → bi-encoder (K1=20) → cross-encoder re-rank → "
        f"confidence gate ({settings.confidence_threshold:.2f}) → internal docs or web fallback."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message, settings)

    question = st.chat_input("Ask about Nimbus architecture, deployments, or billing…")
    if question is None:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.history if st.session_state.get("conversational", True) else []
    with st.chat_message("assistant"):
        with st.spinner("Rewriting, retrieving, re-ranking…"):
            try:
                response, trace = pipeline.run_with_trace(question, history)
                message = {"role": "assistant", "content": response.answer,
                           "response": response, "trace": trace}
            except LLMError as exc:
                message = {"role": "assistant", "content": "", "error": f"LLM error: {exc}"}
            except Exception as exc:  # noqa: BLE001 - keep the UI alive
                message = {"role": "assistant", "content": "",
                           "error": f"Unexpected error ({type(exc).__name__}): {exc}"}
        render_assistant(message, settings)

    st.session_state.messages.append(message)
    if "response" in message:
        st.session_state.history += [
            ChatTurn(role="user", content=question),
            ChatTurn(role="assistant", content=message["response"].answer),
        ]
    st.rerun()  # refresh the sidebar with this run's telemetry


main()
