"""CLI entry point.

Examples:
    python main.py --reindex
    python main.py "What does NIM-E4021 mean and what should I do?"
    python main.py --json "How much is Gale overage per vCPU-hour?"
    python main.py            # interactive session (enables pronoun resolution)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import ConfigError, Settings
from src.llm import LLMError
from src.pipeline import PipelineTrace, TwoStageRAGPipeline
from src.schemas import ChatTurn, RAGResponse


def _print(response: RAGResponse, trace: PipelineTrace, *, as_json: bool, verbose: bool) -> None:
    if as_json:
        payload = response.model_dump()
        if verbose:
            payload["trace"] = {
                "stage1_candidates": trace.stage1_candidates,
                "rerank_status": trace.rerank_status.value if trace.rerank_status else None,
                "top_internal_confidence": round(trace.top_internal_confidence, 4),
                "web_results": trace.web_results,
                "timings_ms": trace.timings_ms,
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\n{response.answer}\n")
    if response.citations:
        print("Sources:")
        for c in response.citations:
            print(f"  {c}")
    print(
        f"\n[route={response.route_taken} confidence={response.confidence_score:.3f} "
        f"query={response.transformed_query!r}]"
    )
    if verbose:
        print(f"[trace: candidates={trace.stage1_candidates} status={trace.rerank_status} "
              f"timings_ms={trace.timings_ms}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nimbus Cloud two-stage RAG")
    parser.add_argument("query", nargs="?", help="Question; omit for interactive mode")
    parser.add_argument("--reindex", action="store_true", help="Force a full re-index")
    parser.add_argument("--json", action="store_true", help="Emit RAGResponse as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logs and trace")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env(args.env_file)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    from src.retriever import ChromaRetriever

    retriever = ChromaRetriever(settings)
    count = retriever.index(force=args.reindex)
    if args.reindex and not args.query:
        print(f"Indexed {count} chunks.")
        return 0

    pipeline = TwoStageRAGPipeline.from_settings(settings, retriever=retriever)

    def ask(question: str, history: list[ChatTurn]) -> RAGResponse | None:
        try:
            response, trace = pipeline.run_with_trace(question, history)
        except LLMError as exc:
            print(f"LLM error: {exc}", file=sys.stderr)
            return None
        _print(response, trace, as_json=args.json, verbose=args.verbose)
        return response

    if args.query:
        return 0 if ask(args.query, []) else 1

    history: list[ChatTurn] = []
    print("Nimbus RAG — ask a question (Ctrl-D or 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        response = ask(question, history)
        if response:
            history += [
                ChatTurn(role="user", content=question),
                ChatTurn(role="assistant", content=response.answer),
            ]


if __name__ == "__main__":
    raise SystemExit(main())
