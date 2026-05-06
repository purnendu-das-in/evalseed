"""Minimal end-to-end example.

Requires OPENAI_API_KEY in the environment. Point ``CORPUS_DIR`` at any
folder of .txt/.md files. The output ``eval.jsonl`` contains only pairs
that passed every filter stage.
"""

from __future__ import annotations

import os
from pathlib import Path

from evalseed import OpenAIJudge, Pipeline


def main() -> None:
    corpus_dir = os.environ.get("CORPUS_DIR", "./docs")
    if not Path(corpus_dir).exists():
        raise SystemExit(
            f"corpus dir {corpus_dir!r} not found. "
            "Set CORPUS_DIR or create ./docs with some .md files."
        )

    pipeline = Pipeline(
        judge=OpenAIJudge(model="gpt-4o-mini"),
        n_pairs=20,
        types=["single_hop", "multi_hop"],
    )
    dataset = pipeline.generate_from_corpus(corpus_dir)
    dataset.save("eval.jsonl")

    rejected_path = Path("rejected.jsonl")
    dataset.rejected.save(rejected_path, only_passed=False)
    print(f"saved {len(dataset.passed)} passed pair(s) to eval.jsonl")
    print(f"saved {len(dataset.rejected)} rejected pair(s) to {rejected_path}")


if __name__ == "__main__":
    main()
