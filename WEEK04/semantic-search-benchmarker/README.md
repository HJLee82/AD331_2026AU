# 🔎 Semantic Search Benchmarker

A search engine over the FastAPI tutorial docs that compares **BM25 keyword search** with **dense vector search** across three chunking strategies, and measures which one works best with Hit Rate, Recall, MRR and latency.

**Stack:** Python 3.11 · uv · Pydantic v2 · tiktoken · ChromaDB · OpenAI `text-embedding-3-small` · `rank-bm25` · Streamlit · pytest

---

## Architecture

```mermaid
flowchart LR
    A[corpus/*.md<br/>12 FastAPI docs] --> B{Chunking engines}
    B --> B1[FixedSizeChunker<br/>256 tok / 32 overlap]
    B --> B2[FixedSizeChunker<br/>1024 tok / 128 overlap]
    B --> B3[StructureAwareChunker<br/># ## ### headers,<br/>code fences kept whole]
    B1 & B2 & B3 --> C[OpenAI embeddings<br/>L2-normalized]
    C --> D[(ChromaDB<br/>cosine HNSW)]
    B3 --> E[(BM25Okapi<br/>shared tokenizer)]
    Q[Query] --> F[Unified search interface<br/>SearchResult]
    D --> F
    E --> F
    F --> G[Benchmarker<br/>Hit@K · Recall@K · MRR@K · p50/p95]
    G --> H[Streamlit SearchWorkbench]
    EV[data/eval_queries.json<br/>15 labeled queries] --> G
```

| File | Role |
|---|---|
| `src/chunking.py` | `DocumentChunk` model, `FixedSizeChunker`, `StructureAwareChunker`, section/anchor parsing |
| `src/indexer.py` | `OpenAIEmbedder` (+ SQLite embedding cache), `DenseRetriever` (Chroma), `BM25Retriever` |
| `src/benchmarker.py` | Eval loading, relevance matching, metrics, 4-config benchmark, CLI |
| `src/app.py` | Streamlit UI: live search comparator + benchmark dashboard |
| `tests/` | Chunking, retrieval and metric-math tests |

---

## Setup

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/HJLee82/AD331_2026AU.git
cd AD331_2026AU/WEEK04/semantic-search-benchmarker
uv sync
cp .env.example .env 
```

`.env`:
```
OPENAI_API_KEY=sk-...
```
`.env` is in `.gitignore` and is never committed. BM25 works without a key; the dense configs need one.

---

## Run

```bash
# Web UI
uv run streamlit run src/app.py

# Benchmark from the terminal (all 4 configs)
uv run python -m src.benchmarker -v

# BM25 only, no API key needed
uv run python -m src.benchmarker --configs bm25

# Tests
uv run pytest -v
```

Document embeddings are cached in `.cache/embeddings.sqlite`, so only the first run calls the embeddings API. Query embeddings are never cached, so latency numbers stay honest.

---

## Corpus & evaluation set

**Corpus:** 12 Markdown files (≈15,500 words) from the [FastAPI tutorial docs](https://github.com/fastapi/fastapi/tree/master/docs/en/docs/tutorial) (MIT license): `first-steps`, `path-params`, `query-params`, `query-params-str-validations`, `body`, `extra-models`, `response-model`, `handling-errors`, `classes-as-dependencies`, `testing`, `debugging`, `oauth2-jwt`. I chose them because they have many exact identifiers (`HTTPException`, `Depends`, `response_model`, `TestClient`), many code blocks, and clear `##` sections.

> Note: FastAPI pulls many Python examples in from external files with lines like `{* ../../docs_src/... *}`. Those lines stay as-is in the corpus. The files still contain ~100 fenced code blocks (Python, JSON, console).

**Eval set (`data/eval_queries.json`):** 15 queries.
- **8 exact** queries use the real identifier, e.g. *"How do I override the RequestValidationError exception handler?"*
- **7 conceptual** queries describe the idea in different words, e.g. *"Two routes look alike and the wrong one keeps catching my requests"* → `order-matters`. I checked each one: none shares a key term with its answer section.

**Ground truth = header anchors, not chunk IDs.** Every FastAPI header ends with a stable ID like `{ #use-httpexception }`. Each query lists these anchors in `relevant_sections`, so the labels stay valid for any chunk size. A chunk counts as relevant only if it covers at least 50% of the labeled section, or if at least 50% of the chunk is that section (see Audit, Finding 1).

---

## Benchmark findings

K = 5, 15 queries. Dense latency includes the OpenAI query-embedding round trip.

| Config | Chunks | Avg tokens | Split code fences | Hit@5 | Recall@5 | MRR@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| BM25 (baseline) | 145 | 182 | 0 | 0.667 | 0.667 | 0.600 | 0.2 | 0.4 |
| Dense · fixed 256 | 123 | 244 | 22 | 0.800 | 0.767 | 0.569 | 232.3 | 926.4 |
| Dense · fixed 1024 | 33 | 881 | 2 | 0.867 | 0.867 | 0.756 | 193.5 | 269.9 |
| **Dense · structure-aware** | 145 | 182 | 0 | **0.933** | **0.867** | **0.756** | 199.7 | 259.6 |

**By query type (Hit@5 / MRR@5)**

| Config | Exact | Conceptual |
|---|---|---|
| BM25 | 1.000 / 0.938 | 0.286 / 0.214 |
| Dense · fixed 256 | 1.000 / 0.838 | 0.571 / 0.262 |
| Dense · fixed 1024 | 1.000 / 0.938 | 0.714 / 0.548 |
| Dense · structure-aware | 1.000 / 1.000 | 0.857 / 0.476 |

**Dense latency breakdown (p50):** query embedding ≈ 190–230 ms, Chroma vector search ≈ 2–3 ms.

### Reflection

Structure-aware chunking won overall (Hit@5 0.933) because FastAPI's docs are already organized as one topic per section: a chunk that matches a section boundary holds one complete idea and its code example, with nothing cut off and nothing unrelated mixed in. The fixed 256-token chunker shows the opposite failure, semantic fragmentation: it found the right neighborhood (Hit@10 = 1.0) but had the lowest MRR@5 (0.569) and cut 22 code blocks in half, so the chunk holding the answer often lacked the context to rank first. The 1024-token chunker had the best conceptual MRR (0.548), since a large chunk carries enough surrounding explanation to match a paraphrased question. But it mixes several sections in each chunk (vector dilution), so its overall Hit@5 stays below structure-aware.

BM25 was perfect at finding exact identifiers (Hit@5 1.0, MRR 0.938) and answers in 0.2 ms, but it found only 2 of 7 conceptual queries, because it can only match words the user actually typed. Dense search with OpenAI embeddings also found every exact query, so on this corpus BM25's advantage shows up in ranking (it ranks the answer higher than fixed-256) and in cost and speed, not in recall. The latency gap is almost entirely the embeddings API round trip: the vector search itself takes about 2 ms. For a production docs search, I would use structure-aware chunks with a hybrid of BM25 and dense search merged with reciprocal rank fusion. That keeps BM25's precision on identifiers and the embedding's recall on paraphrases.

---

## Tests

| # | Test | File |
|---|---|---|
| 1 | Consecutive fixed chunks share exactly the configured token overlap, and the chunks rebuild the original text with no dropped characters | `test_chunking.py` |
| 2 | Structure-aware chunker never splits a code fence, even when a Python `# comment` inside it looks like a header | `test_chunking.py` |
| 3 | BM25 ranks a rare identifier (`CUSTOM_CONFIG_FLAG_XYZ`) at rank 1 | `test_retrieval.py` |
| 4 | Dense search finds a paraphrased query that shares no meaningful words with its answer in the top 5 (needs `OPENAI_API_KEY`) | `test_retrieval.py` |
| 5 | Hit / Recall / MRR math: rank 3 → 1/3, out-of-K → 0, miss → 0, empty → 0, ties, mean MRR | `test_benchmarks.py` |

---

## AI audit summary

The first draft was generated with Claude from the assignment's sample prompt, adapted to my corpus, embedding model and label format. The prompt, the raw output, and full before/after code are in [`ai_prompt_log.md`](ai_prompt_log.md).

**Required audit points:** ground truth uses section anchors instead of chunk indexes ✅ · cosine distance correctly converted to similarity (`1 − distance`, verified with a 2-vector test) ✅ · the same tokenizer is used for BM25 corpus and queries ✅ · MRR math gives 0 past K and on misses, with no division by zero ✅ · code-fence splitting ⚠️ found in the fixed chunker.

**Bugs found and fixed:**
1. **Relevance credit was too generous to large chunks.** A chunk that touched a section by a single line counted as relevant, which inflated Hit Rate for 1024-token chunks. Fixed with a 50% coverage rule.
2. **The fixed chunker splits code blocks, and nothing measured it.** I kept the fixed chunker naive on purpose (it is the baseline) but added a `broken_fence` flag and a "split code fences" column (256 → 22 chunks, 1024 → 2, structure-aware → 0).
3. **The SQLite embedding cache crashed in Streamlit.** Streamlit reruns run on different threads, and SQLite rejects use from another thread by default. Fixed with `check_same_thread=False`.