"""
SearchWorkbench: Streamlit dashboard for the semantic search benchmark.

Run from the project root:
    uv run streamlit run src/app.py

Tab 1 (Live Search): BM25 vs dense results side by side for any query.
Tab 2 (Benchmark): runs all 4 configs over data/eval_queries.json and
shows metrics, charts, and a BM25-vs-dense failure analysis.
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `streamlit run src/app.py` can import src.*

from dotenv import load_dotenv  # noqa: E402

from src.benchmarker import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    build_default_configs,
    load_corpus,
    load_eval_queries,
)
from src.chunking import chunk_corpus  # noqa: E402
from src.indexer import EmbeddingCache, OpenAIEmbedder, bm25_tokenize  # noqa: E402

load_dotenv(ROOT / ".env")

CORPUS_DIR = ROOT / "corpus"
QUERIES_PATH = ROOT / "data" / "eval_queries.json"
CACHE_PATH = ROOT / ".cache" / "embeddings.sqlite"
KS = (1, 3, 5, 10)
LABELS = {
    "bm25": "BM25 (keyword)",
    "dense_fixed_256": "Dense · fixed 256",
    "dense_fixed_1024": "Dense · fixed 1024",
    "dense_structure": "Dense · structure-aware",
}

st.set_page_config(page_title="SearchWorkbench", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Build indexes once per server process
# --------------------------------------------------------------------------- #
def has_api_key() -> bool:
    return os.getenv("OPENAI_API_KEY", "").startswith("sk-")


@st.cache_resource(show_spinner="Chunking corpus and building indexes…")
def build_workbench(use_dense: bool):
    """Return (configs, retrievers, chunks_by_config). Dense indexing uses the
    embedding cache, so only the first run ever calls the embeddings API."""
    docs = load_corpus(CORPUS_DIR)
    factory = (lambda: OpenAIEmbedder(cache=EmbeddingCache(CACHE_PATH))) if use_dense else None
    configs = build_default_configs(factory)
    if not use_dense:
        configs = [c for c in configs if c.name == "bm25"]

    retrievers, chunks = {}, {}
    for cfg in configs:
        cs = chunk_corpus(cfg.chunker, docs)
        r = cfg.retriever_factory(cfg.name)
        r.index(cs)
        retrievers[cfg.name], chunks[cfg.name] = r, cs
    return docs, configs, retrievers, chunks


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def highlight(text: str, query: str, max_chars: int = 700) -> str:
    """HTML-escape a snippet and <mark> every word that also appears in the query
    (using the same tokenizer BM25 uses, so highlights match what BM25 scored)."""
    terms = {t for t in bm25_tokenize(query) if len(t) > 2}
    snippet = text if len(text) <= max_chars else text[:max_chars] + " …"
    out = html.escape(snippet)
    if terms:
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + r")\b",
                             re.IGNORECASE)
        out = pattern.sub(r"<mark>\1</mark>", out)
    return out


def render_results(results, query: str, score_label: str) -> None:
    if not results:
        st.info("No results (BM25 returns nothing when no query word appears in the corpus).")
        return
    for r in results:
        section = " › ".join(r.metadata.get("header_path", [])[-2:]) or "(preamble)"
        with st.container(border=True):
            st.markdown(f"**#{r.rank}** · `{r.parent_doc}` · {html.escape(section)}")
            st.caption(f"{score_label}: **{r.score:.3f}** · chunk `{r.chunk_id}`")
            st.markdown(
                f"<div style='font-size:0.85rem;white-space:pre-wrap;"
                f"font-family:ui-monospace,monospace'>{highlight(r.content, query)}</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🔎 SearchWorkbench")
st.caption("FastAPI docs · BM25 vs OpenAI text-embedding-3-small + ChromaDB")

use_dense = has_api_key()
if not use_dense:
    st.warning("OPENAI_API_KEY not found in .env, so only BM25 is available.")

docs, configs, retrievers, chunks = build_workbench(use_dense)
queries = load_eval_queries(QUERIES_PATH)

tab_search, tab_bench = st.tabs(["🔍 Live Search Comparator", "📊 Benchmark Dashboard"])

# ---------------- Tab 1: live search ---------------- #
with tab_search:
    examples = {f"{q.query_id} ({q.type}): {q.query}": q.query for q in queries}
    c1, c2, c3 = st.columns([3, 1, 2])
    pick = c1.selectbox("Try an eval query, or type your own below",
                        ["(custom)"] + list(examples), index=0)
    k = c2.slider("K", 1, 10, 5)
    dense_names = [n for n in retrievers if n != "bm25"]
    dense_name = c3.selectbox("Dense config", dense_names, index=len(dense_names) - 1,
                              format_func=LABELS.get) if dense_names else None
    query = st.text_input("Query", value="" if pick == "(custom)" else examples[pick],
                          placeholder="e.g. How do I raise an HTTPException with a 404?")

    if query.strip():
        left, right = st.columns(2)
        with left:
            st.subheader("BM25 keyword search")
            res = retrievers["bm25"].search(query, k=k)
            render_results(res, query, "BM25 score")
        with right:
            st.subheader(LABELS.get(dense_name, "Dense search"))
            if dense_name:
                r = retrievers[dense_name]
                res = r.search(query, k=k)
                t = r.last_timing_ms
                st.caption(f"⏱ query embedding {t['embed']:.0f} ms · vector search {t['ann']:.1f} ms")
                render_results(res, query, "cosine similarity")
            else:
                st.info("Add OPENAI_API_KEY to .env to enable dense search.")

# ---------------- Tab 2: benchmark ---------------- #
with tab_bench:
    st.markdown(f"Runs **{len(configs)} configs** over **{len(queries)} labeled queries** "
                f"from `data/eval_queries.json`.")
    if st.button("▶️ Run Benchmark", type="primary"):
        runner = BenchmarkRunner(docs, queries, ks=KS)
        # Reuse the already-built retrievers (re-indexing hits the embedding cache).
        cfgs = [BenchmarkConfig(c.name, c.chunker, lambda n, r=retrievers[c.name]: r) for c in configs]
        with st.spinner("Running benchmark…"):
            st.session_state["reports"] = runner.run(cfgs)

    reports = st.session_state.get("reports")
    if reports:
        bench_k = st.radio("Cutoff K", [3, 5, 10], index=1, horizontal=True)

        rows = []
        for r in reports:
            m = r.metrics["all"]
            rows.append({
                "Config": LABELS.get(r.name, r.name),
                "Chunks": r.num_chunks,
                "Avg tokens": round(r.avg_chunk_tokens),
                "Split code fences": r.broken_fence_chunks,
                f"Hit@{bench_k}": m[f"hit@{bench_k}"],
                f"Recall@{bench_k}": m[f"recall@{bench_k}"],
                f"MRR@{bench_k}": m[f"mrr@{bench_k}"],
                "p50 ms": r.latency_ms["p50"],
                "p95 ms": r.latency_ms["p95"],
            })
        table = pd.DataFrame(rows).set_index("Config")
        st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if "@" in c}
                                        | {"p50 ms": "{:.1f}", "p95 ms": "{:.1f}"}),
                     width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Hit@{bench_k} and MRR@{bench_k}**")
            st.bar_chart(table[[f"Hit@{bench_k}", f"MRR@{bench_k}"]], stack=False)
        with c2:
            st.markdown("**p50 latency (ms)**")
            st.bar_chart(table[["p50 ms"]])

        st.markdown(f"**By query type (Hit@{bench_k})**")
        by_type = pd.DataFrame({
            LABELS.get(r.name, r.name): {
                "exact": r.metrics.get("exact", {}).get(f"hit@{bench_k}"),
                "conceptual": r.metrics.get("conceptual", {}).get(f"hit@{bench_k}"),
            } for r in reports
        }).T
        st.bar_chart(by_type, stack=False)

        # ---- Failure analysis: BM25 vs the dense config of your choice ---- #
        st.subheader("🔬 Failure analysis")
        dense_reports = [r for r in reports if r.name != "bm25"]
        if dense_reports:
            rival_name = st.selectbox("Compare BM25 against", [r.name for r in dense_reports],
                                      index=len(dense_reports) - 1, format_func=LABELS.get)
            bm = {row["query_id"]: row for row in next(r for r in reports if r.name == "bm25").per_query}
            dn = {row["query_id"]: row for row in next(r for r in reports if r.name == rival_name).per_query}
            qmap = {q.query_id: q for q in queries}
            key = f"mrr@{bench_k}"

            gold_chunks = chunks["bm25"]  # structure-aware chunks carry section anchors

            def explain(q, bm_rr: float, dn_rr: float) -> str:
                gold = " ".join(
                    c.content for c in gold_chunks
                    if c.parent_doc == q.source_file
                    and set(c.metadata["section_anchors"]) & set(q.relevant_sections)
                )
                gold_terms = set(bm25_tokenize(gold))
                shared = sorted(t for t in set(bm25_tokenize(q.query)) if len(t) > 3 and t in gold_terms)
                if bm_rr > dn_rr:
                    return ("Exact term match: BM25 ranks the answer higher because it shares rare "
                            f"literal words with it ({', '.join(shared[:4]) or 'none'}).")
                return (f"Paraphrase: only {len(shared)} meaningful word(s) shared with the answer "
                        f"({', '.join(shared[:4]) or 'none'}), so BM25 has little to match; "
                        "the embedding matches the meaning instead.")

            wins = {"BM25 wins": [], f"{LABELS[rival_name]} wins": []}
            for qid, q in qmap.items():
                b, d = bm[qid][key], dn[qid][key]
                if b == d:
                    continue
                row = {"Query": f"{qid} · {q.query}", "Type": q.type,
                       f"BM25 {key}": round(b, 3), f"Dense {key}": round(d, 3),
                       "Why": explain(q, b, d)}
                wins["BM25 wins" if b > d else f"{LABELS[rival_name]} wins"].append(row)

            for title, items in wins.items():
                st.markdown(f"**{title}** ({len(items)})")
                if items:
                    st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
                else:
                    st.caption("None at this K.")
        else:
            st.caption("Dense configs need OPENAI_API_KEY.")
    else:
        st.info("Click **Run Benchmark** to compute Hit Rate, Recall, MRR and latency.")
