from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalseed import JudgeAuthError, OpenAIJudge, Pipeline
from evalseed.schemas import QAType


def _parse_types(value: str) -> list[QAType]:
    out: list[QAType] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(QAType(raw))
    if not out:
        raise argparse.ArgumentTypeError("at least one QA type required")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalseed",
        description="Generate quality-filtered synthetic Q&A datasets for RAG evaluation.",
    )
    parser.add_argument("corpus", help="Path to a corpus file or directory.")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output JSONL path. Defaults to <corpus>/output/eval.jsonl "
        "(or <corpus_parent>/output/eval.jsonl if corpus is a file).",
    )
    parser.add_argument("-n", "--n-pairs", type=int, default=50, help="Target pair count.")
    parser.add_argument(
        "--types",
        type=_parse_types,
        default=[QAType.SINGLE_HOP, QAType.MULTI_HOP],
        help="Comma-separated QA types: single_hop,multi_hop,distractor",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Save passed AND rejected pairs.")

    args = parser.parse_args(argv)
    out_path = _resolve_out_path(args.corpus, args.out)
    try:
        judge = OpenAIJudge(model=args.model)
        pipeline = Pipeline(
            judge=judge,
            n_pairs=args.n_pairs,
            types=args.types,
            seed=args.seed,
        )
        dataset = pipeline.generate_from_corpus(args.corpus)
    except JudgeAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dataset.save(out_path, only_passed=not args.all)
    print(f"wrote {out_path}")
    return 0


def _resolve_out_path(corpus: str, out: str | None) -> Path:
    if out is not None:
        return Path(out)
    corpus_path = Path(corpus)
    base = corpus_path if corpus_path.is_dir() else corpus_path.parent
    return base / "output" / "eval.jsonl"


if __name__ == "__main__":
    sys.exit(main())
