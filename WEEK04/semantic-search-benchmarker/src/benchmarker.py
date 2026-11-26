"""
Retrieval benchmark: BM25 vs dense retrieval under different chunking strategies.

Usage::

    python -m src.benchmarker                       # all 4 configs, K = 1,3,5,10
    python -m src.benchmarker --configs bm25        # offline, no API key needed
    python -m src.benchmarker --ks 5 10 --output results/run.json

Relevance model
---------------
Ground truth is a set of header anchors per query (``relevant_sections``) in a
given ``source_file``. A retrieved chunk is relevant to anchor ``a`` when it
comes from that file and either

* overlaps section ``a`` (``metadata["section_anchors"]``), or
* sits *inside* section ``a`` (``a`` in ``metadata["anchor_path"]``), so a
  ``###`` sub-section chunk counts for its ``##`` parent. Disable with
  ``--strict-anchors``.

Metrics (per query, then macro-averaged):

* Hit Rate@K – 1 if any of the top-K chunks is relevant.
* Recall@K   – fraction of the query's relevant anchors covered by the top-K.
* MRR@K      – 1 / rank of the first relevant chunk (0 if none in top-K).
* Latency    – end-to-end ``search()`` wall time, p50 / p95 in ms. For dense
  retrievers this includes the query-embedding API round-trip, which is
  reported separately as well.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from .chunking import (
    Chunker,
    DocumentChunk,
    FixedSizeChunker,
    StructureAwareChunker,
    chunk_corpus,
)
from .indexer import BM25Retriever, DenseRetriever, Embedder, Retriever, SearchResult

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class EvalQuery(BaseModel):
    """One labelled query from ``data/eval_queries.json``."""

    query_id: str
    query: str
    type: Literal["exact", "conceptual"]
    relevant_sections: list[str] = Field(min_length=1)
    source_file: str


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    """Load and validate the evaluation set (fails loudly on schema errors)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = [EvalQuery.model_validate(q) for q in raw]
    ids = [q.query_id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query_id values in eval set")
    return queries


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """Read every ``*.md`` file under ``corpus_dir`` → ``{filename: text}``."""
    files = sorted(Path(corpus_dir).rglob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {corpus_dir}")
    return {f.name: f.read_text(encoding="utf-8") for f in files}


def validate_labels(queries: Sequence[EvalQuery], chunks: Sequence[DocumentChunk]) -> list[str]:
    """Return warnings for labels that no chunk can ever match.

    A typo in an anchor silently caps Recall below 1.0 for *every* config, so
    this is checked up-front.
    """
    known: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        known[c.parent_doc].update(c.metadata["section_anchors"], c.metadata["anchor_path"])
    warnings = []
    for q in queries:
        doc = Path(q.source_file).name
        if doc not in known:
            warnings.append(f"{q.query_id}: source_file {q.source_file!r} not in corpus")
            continue
        for a in q.relevant_sections:
            if a not in known[doc]:
                warnings.append(f"{q.query_id}: anchor {a!r} not found in {doc}")
    return warnings


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def matched_anchors(result: SearchResult, query: EvalQuery, match_ancestors: bool = True) -> set[str]:
    """Relevant anchors of ``query`` that ``result`` satisfies (may be empty)."""
    if result.parent_doc != Path(query.source_file).name:
        return set()
    covered = set(result.metadata.get("section_anchors", []))
    if match_ancestors:
        covered |= set(result.metadata.get("anchor_path", []))
    return covered & set(query.relevant_sections)


@dataclass(frozen=True)
class QueryScore:
    hit: float
    recall: float
    rr: float


def score_query(
    results: Sequence[SearchResult], query: EvalQuery, k: int, match_ancestors: bool = True
) -> QueryScore:
    """Hit, Recall and reciprocal rank at cutoff ``k`` for one query."""
    relevant = set(query.relevant_sections)
    covered: set[str] = set()
    first_rank: int | None = None
    for r in results[:k]:
        m = matched_anchors(r, query, match_ancestors)
        if m and first_rank is None:
            first_rank = r.rank
        covered |= m
    return QueryScore(
        hit=1.0 if first_rank else 0.0,
        recall=len(covered) / len(relevant),
        rr=1.0 / first_rank if first_rank else 0.0,
    )


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


# --------------------------------------------------------------------------- #
# Benchmark orchestration
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkConfig:
    """A (chunker, retriever) pairing to evaluate."""

    name: str
    chunker: Chunker
    retriever_factory: Callable[[str], Retriever]


class ConfigReport(BaseModel):
    """Everything measured for one configuration."""

    name: str
    num_chunks: int
    avg_chunk_tokens: float
    index_time_s: float
    broken_fence_chunks: int = 0  # chunks whose code block was cut in half
    metrics: dict[str, dict[str, float]]  # {"all"|"exact"|"conceptual": {"hit@5": ..}}
    latency_ms: dict[str, float]
    per_query: list[dict]


class BenchmarkRunner:
    """Runs every config over the same corpus and query set.

    Args:
        docs: ``{filename: markdown}`` corpus.
        queries: Labelled eval queries.
        ks: Cutoffs to report. Each query is searched once at ``max(ks)``.
        match_ancestors: See module docstring.
    """

    def __init__(
        self,
        docs: dict[str, str],
        queries: Sequence[EvalQuery],
        ks: Sequence[int] = (1, 3, 5, 10),
        match_ancestors: bool = True,
    ) -> None:
        self.docs = docs
        self.queries = list(queries)
        self.ks = sorted(set(ks))
        self.match_ancestors = match_ancestors
        self._chunk_cache: dict[str, list[DocumentChunk]] = {}

    def _chunks_for(self, chunker: Chunker) -> list[DocumentChunk]:
        # BM25 and dense-structure share the same chunks; chunk once.
        if chunker.name not in self._chunk_cache:
            self._chunk_cache[chunker.name] = chunk_corpus(chunker, self.docs)
        return self._chunk_cache[chunker.name]

    def run_config(self, cfg: BenchmarkConfig) -> ConfigReport:
        chunks = self._chunks_for(cfg.chunker)
        for w in validate_labels(self.queries, chunks):
            log.warning("[%s] %s", cfg.name, w)

        retriever = cfg.retriever_factory(cfg.name)
        t0 = time.perf_counter()
        retriever.index(chunks)
        index_time = time.perf_counter() - t0

        max_k = self.ks[-1]
        retriever.search(self.queries[0].query, k=max_k)  # warm-up (connections, caches)

        latencies: list[float] = []
        embed_lat: list[float] = []
        per_query: list[dict] = []
        sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for q in self.queries:
            t = time.perf_counter()
            results = retriever.search(q.query, k=max_k)
            latencies.append((time.perf_counter() - t) * 1e3)
            if timing := getattr(retriever, "last_timing_ms", None):
                embed_lat.append(timing["embed"])

            row: dict = {"query_id": q.query_id, "type": q.type, "latency_ms": latencies[-1],
                         "top_ids": [r.chunk_id for r in results[:5]]}
            for k in self.ks:
                s = score_query(results, q, k, self.match_ancestors)
                for bucket in ("all", q.type):
                    sums[bucket][f"hit@{k}"].append(s.hit)
                    sums[bucket][f"recall@{k}"].append(s.recall)
                    sums[bucket][f"mrr@{k}"].append(s.rr)
                row.update({f"hit@{k}": s.hit, f"recall@{k}": s.recall, f"mrr@{k}": s.rr})
            per_query.append(row)

        latency = {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)}
        if embed_lat:
            latency |= {"embed_p50": percentile(embed_lat, 50),
                        "search_only_p50": percentile([a - b for a, b in zip(latencies, embed_lat)], 50)}

        return ConfigReport(
            name=cfg.name,
            num_chunks=len(chunks),
            avg_chunk_tokens=statistics.fmean(c.metadata["token_count"] for c in chunks) if chunks else 0.0,
            index_time_s=index_time,
            broken_fence_chunks=sum(bool(c.metadata.get("broken_fence")) for c in chunks),
            metrics={b: {m: statistics.fmean(v) for m, v in d.items()} for b, d in sums.items()},
            latency_ms=latency,
            per_query=per_query,
        )

    def run(self, configs: Sequence[BenchmarkConfig]) -> list[ConfigReport]:
        reports = []
        for cfg in configs:
            log.info("Running %s ...", cfg.name)
            reports.append(self.run_config(cfg))
        return reports


# --------------------------------------------------------------------------- #
# Default experiment
# --------------------------------------------------------------------------- #
def build_default_configs(
    embedder_factory: Callable[[], Embedder] | None,
    chroma_client: object | None = None,
    encoding: object | None = None,
) -> list[BenchmarkConfig]:
    """The four required configurations.

    BM25 runs on structure-aware chunks so the sparse/dense comparison isn't
    confounded by chunk size. Fixed-size overlap is 12.5% of chunk size.
    ``embedder_factory`` may be None for BM25-only runs.
    """
    structure = StructureAwareChunker(encoding=encoding)
    _embedder: list[Embedder] = []

    def dense(name: str) -> Retriever:
        if embedder_factory is None:
            raise RuntimeError("Dense configs need an embedder (is OPENAI_API_KEY set?)")
        if not _embedder:
            _embedder.append(embedder_factory())  # one client/cache shared by all dense configs
        return DenseRetriever(name, _embedder[0], client=chroma_client)

    return [
        BenchmarkConfig("bm25", structure, lambda n: BM25Retriever(n)),
        BenchmarkConfig("dense_fixed_256", FixedSizeChunker(256, 32, encoding=encoding), dense),
        BenchmarkConfig("dense_fixed_1024", FixedSizeChunker(1024, 128, encoding=encoding), dense),
        BenchmarkConfig("dense_structure", structure, dense),
    ]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(reports: Sequence[ConfigReport], ks: Sequence[int]) -> str:
    """Markdown tables: overall metrics, then MRR/Hit split by query type."""
    lines: list[str] = []
    cols = [f"{m}@{k}" for k in ks for m in ("hit", "recall", "mrr")]
    header = ["config", "chunks", "avg tok", "split fences", *cols, "p50 ms", "p95 ms"]
    lines += ["## Overall", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in reports:
        m = r.metrics["all"]
        lines.append("| " + " | ".join([
            r.name, str(r.num_chunks), f"{r.avg_chunk_tokens:.0f}", str(r.broken_fence_chunks),
            *(f"{m[c]:.3f}" for c in cols),
            f"{r.latency_ms['p50']:.1f}", f"{r.latency_ms['p95']:.1f}",
        ]) + " |")

    k = ks[-1] if 5 not in ks else 5
    lines += ["", f"## By query type (@{k})", "",
              "| config | exact hit | exact mrr | conceptual hit | conceptual mrr |",
              "|---|---|---|---|---|"]
    for r in reports:
        e, c = r.metrics.get("exact", {}), r.metrics.get("conceptual", {})
        fmt = lambda d, key: f"{d[key]:.3f}" if key in d else "–"  # noqa: E731
        lines.append(f"| {r.name} | {fmt(e, f'hit@{k}')} | {fmt(e, f'mrr@{k}')} | "
                     f"{fmt(c, f'hit@{k}')} | {fmt(c, f'mrr@{k}')} |")

    dense = [r for r in reports if "embed_p50" in r.latency_ms]
    if dense:
        lines += ["", "## Dense latency breakdown (p50 ms)", "",
                  "| config | query embed | vector search |", "|---|---|---|"]
        lines += [f"| {r.name} | {r.latency_ms['embed_p50']:.1f} | "
                  f"{r.latency_ms['search_only_p50']:.2f} |" for r in dense]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--queries", default="data/eval_queries.json")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--configs", nargs="+", help="Subset of config names to run")
    p.add_argument("--output", default="results/benchmark.json")
    p.add_argument("--persist-dir", help="Persist Chroma here (default: in-memory)")
    p.add_argument("--cache", default=".cache/embeddings.sqlite",
                   help="Document-embedding cache ('' to disable)")
    p.add_argument("--strict-anchors", action="store_true",
                   help="Don't credit sub-section chunks to their parent anchor")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv
    load_dotenv()

    def embedder_factory() -> Embedder:
        from .indexer import EmbeddingCache, OpenAIEmbedder
        return OpenAIEmbedder(cache=EmbeddingCache(args.cache) if args.cache else None)

    chroma_client = None
    if args.persist_dir:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=args.persist_dir)

    configs = build_default_configs(embedder_factory, chroma_client)
    if args.configs:
        unknown = set(args.configs) - {c.name for c in configs}
        if unknown:
            p.error(f"Unknown configs: {sorted(unknown)}")
        configs = [c for c in configs if c.name in args.configs]

    runner = BenchmarkRunner(
        load_corpus(args.corpus), load_eval_queries(args.queries),
        ks=args.ks, match_ancestors=not args.strict_anchors,
    )
    reports = runner.run(configs)

    print(format_report(reports, runner.ks))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.model_dump() for r in reports], indent=2), encoding="utf-8")
    print(f"\nFull results (incl. per-query rows): {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
