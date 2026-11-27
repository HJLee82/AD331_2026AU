"""Confidence-threshold calibration for the Stage 2 gate.

Runs labelled in-domain and out-of-domain queries through Stage 1 (bi-encoder) and
Stage 2 (cross-encoder) only. No LLM or web calls, so it costs nothing.

    uv run python calibrate.py                 # prints tables, writes calibration_results.md
    uv run python calibrate.py --threshold 0.6 # evaluate a different chosen cutoff
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Answerable from docs/ (some use exact identifiers, some are paraphrased on purpose).
IN_DOMAIN = [
    "What does NIM-E4021 mean?",
    "What is the CIDR block of Nimbus VPC-Stage?",
    "What are the system requirements for Kubernetes worker nodes?",
    "How much does the Gale plan cost per month?",
    "What is the overage rate per vCPU-hour on the Tempest plan?",
    "How do I roll back a release after a full rollout?",
    "What happens when an account is 15 days past due?",
    "How long do customers have to dispute an invoice?",
    "What are the canary rollout stages and bake times?",
    "How do I fix a stale Terraform state lock?",
    "Which service handles usage metering and invoice generation?",
    "Can I get my money back if I cancel a yearly prepaid plan after a month?",
]

# Need the web. The last four are "near misses" that share vocabulary with the docs.
OUT_OF_DOMAIN = [
    "What caused the AWS us-east-1 outage in October 2025?",
    "What is the latest stable version of Kubernetes?",
    "How do I install Docker on Ubuntu?",
    "What is the difference between TCP and UDP?",
    "What is the weather in Seattle today?",
    "How do I cache dependencies in a GitHub Actions workflow?",
    "Who founded HashiCorp?",
    "What is the maximum execution timeout for AWS Lambda?",
    "How much does an AWS t3.medium instance cost per hour?",
    "What is the default idle timeout of an AWS Application Load Balancer?",
    "How do I write a Terraform module for an S3 bucket?",
    "What is Google Cloud's uptime SLA for Compute Engine?",
]

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]


@dataclass(frozen=True)
class Scored:
    query: str
    in_domain: bool
    top_score: float
    top_chunk: str


def score_queries(retriever, reranker, labelled: list[tuple[str, bool]]) -> list[Scored]:
    results = []
    for query, in_domain in labelled:
        candidates = retriever.retrieve(query)
        ranked = reranker.score(query, candidates)
        top = ranked[0] if ranked else None
        results.append(
            Scored(query, in_domain, top.score if top else 0.0, top.chunk_id if top else "-")
        )
    return results


def rates(results: list[Scored], threshold: float) -> dict[str, float]:
    """False fallback = in-domain sent to web. Missed fallback = out-of-domain kept internal."""
    ins = [r for r in results if r.in_domain]
    outs = [r for r in results if not r.in_domain]
    false_fb = sum(r.top_score < threshold for r in ins)
    missed_fb = sum(r.top_score >= threshold for r in outs)
    return {
        "false_fallback": false_fb,
        "false_fallback_rate": false_fb / len(ins) if ins else 0.0,
        "missed_fallback": missed_fb,
        "missed_fallback_rate": missed_fb / len(outs) if outs else 0.0,
        "accuracy": (len(results) - false_fb - missed_fb) / len(results) if results else 0.0,
    }


def score_table(results: list[Scored], threshold: float) -> str:
    lines = [
        "| Query | Label | Top score | Route @ {:.2f} | Correct? | Top chunk |".format(threshold),
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (not r.in_domain, -r.top_score)):
        route = "INTERNAL" if r.top_score >= threshold else "WEB"
        correct = (route == "INTERNAL") == r.in_domain
        lines.append(
            f"| {r.query} | {'in' if r.in_domain else 'out'} | {r.top_score:.4f} | {route} | "
            f"{'✅' if correct else '❌'} | `{r.top_chunk.split('::')[0]}` |"
        )
    return "\n".join(lines)


def threshold_table(results: list[Scored], thresholds: list[float], chosen: float) -> str:
    n_in = sum(r.in_domain for r in results)
    n_out = len(results) - n_in
    lines = [
        f"| Threshold | False fallback (in → web, of {n_in}) | "
        f"Missed fallback (out → internal, of {n_out}) | Routing accuracy |",
        "|---|---|---|---|",
    ]
    for t in thresholds:
        r = rates(results, t)
        mark = " **← chosen**" if abs(t - chosen) < 1e-9 else ""
        lines.append(
            f"| {t:.2f}{mark} | {r['false_fallback']} ({r['false_fallback_rate']:.0%}) | "
            f"{r['missed_fallback']} ({r['missed_fallback_rate']:.0%}) | {r['accuracy']:.0%} |"
        )
    return "\n".join(lines)


def separation_summary(results: list[Scored]) -> str:
    ins = sorted(r.top_score for r in results if r.in_domain)
    outs = sorted(r.top_score for r in results if not r.in_domain)
    lowest_in, highest_out = ins[0], outs[-1]
    gap = (
        f"clean gap between {highest_out:.4f} and {lowest_in:.4f}"
        if highest_out < lowest_in
        else f"overlap: highest out-of-domain {highest_out:.4f} ≥ lowest in-domain {lowest_in:.4f}"
    )
    return (
        f"- In-domain scores: min {ins[0]:.4f}, median {ins[len(ins) // 2]:.4f}, max {ins[-1]:.4f}\n"
        f"- Out-of-domain scores: min {outs[0]:.4f}, median {outs[len(outs) // 2]:.4f}, "
        f"max {outs[-1]:.4f}\n"
        f"- Separation: {gap}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cutoff to highlight (default: CONFIDENCE_THRESHOLD from .env)")
    parser.add_argument("--out", default="calibration_results.md")
    args = parser.parse_args()

    from src.config import Settings
    from src.reranker import CrossEncoderReranker
    from src.retriever import ChromaRetriever

    settings = Settings.from_env(ROOT / ".env")
    chosen = args.threshold if args.threshold is not None else settings.confidence_threshold
    retriever = ChromaRetriever(settings)
    retriever.index()
    reranker = CrossEncoderReranker(settings.reranker_model, threshold=chosen,
                                    top_k=settings.rerank_top_k)

    labelled = [(q, True) for q in IN_DOMAIN] + [(q, False) for q in OUT_OF_DOMAIN]
    results = score_queries(retriever, reranker, labelled)

    report = "\n\n".join([
        f"## Threshold Calibration ({len(IN_DOMAIN)} in-domain, {len(OUT_OF_DOMAIN)} out-of-domain)",
        "Top cross-encoder score = sigmoid(top logit) after Stage 1 (K1="
        f"{settings.retrieval_top_k}) and Stage 2 re-ranking. Raw queries, no rewriting.",
        "### Score separation",
        separation_summary(results),
        "### Error rates by threshold",
        threshold_table(results, THRESHOLDS, chosen),
        f"### Per-query scores (routed at {chosen:.2f})",
        score_table(results, chosen),
    ])
    print(report)
    (ROOT / args.out).write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
