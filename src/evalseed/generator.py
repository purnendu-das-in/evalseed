from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

from evalseed.corpus import Chunk
from evalseed.exceptions import GenerationError, JudgeAuthError, JudgeError
from evalseed.judges import Judge
from evalseed.schemas import Difficulty, QAPair, QAType

_SYSTEM = (
    "You generate high-quality question-answer pairs for evaluating "
    "retrieval-augmented generation systems. Every question must be "
    "answerable from the provided CONTEXT alone, every answer must be "
    "fully supported by the CONTEXT, and questions must not be verbatim "
    "restatements of context spans. Respond ONLY with JSON."
)

_USER_TEMPLATE = """Generate {n} question-answer pairs of type "{qa_type}" from the CONTEXT.

CONTEXT:
\"\"\"
{context}
\"\"\"

Type definitions:
- single_hop: answerable from a single span; tests basic comprehension.
- multi_hop: requires combining 2+ distant facts from the context.
- distractor: question is plausibly answerable from the context but the
  context only contains related-but-different information; the correct
  answer is "the context does not specify" or equivalent.

For each pair, also assign a difficulty (easy, medium, hard).

Return JSON with this exact schema:
{{
  "pairs": [
    {{
      "question": "<string>",
      "answer": "<string>",
      "difficulty": "<easy|medium|hard>"
    }}
  ]
}}

Requirements:
- Do not produce yes/no questions unless they require non-trivial inference.
- Do not produce questions that quote the context verbatim.
- Each pair must be self-contained (no "the above", "the following", etc.).
"""


class QAGenerator:
    """Generates raw QA pairs from chunks via a judge LLM.

    Output is intentionally unfiltered — the Pipeline applies filter stages
    afterward. This separation lets users plug in alternative generators
    (RAGAS, DeepEval, hand-written) and still benefit from filtering.
    """

    def __init__(
        self,
        judge: Judge,
        types: Sequence[QAType | str] = (QAType.SINGLE_HOP, QAType.MULTI_HOP),
        pairs_per_chunk: int = 2,
        seed: int | None = None,
        max_workers: int = 8,
    ) -> None:
        if not types:
            raise ValueError("types must be non-empty")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.judge = judge
        self.types = [QAType(t) if not isinstance(t, QAType) else t for t in types]
        self.pairs_per_chunk = pairs_per_chunk
        self.max_workers = max_workers
        self._rng = random.Random(seed)

    def generate(self, chunks: Iterable[Chunk], target_n: int) -> list[QAPair]:
        chunks_list = list(chunks)
        if not chunks_list:
            raise GenerationError("no chunks to generate from")

        if self.max_workers <= 1:
            return self._generate_sequential(chunks_list, target_n)
        return self._generate_parallel(chunks_list, target_n)

    def _generate_sequential(
        self, chunks_list: list[Chunk], target_n: int
    ) -> list[QAPair]:
        pairs: list[QAPair] = []
        chunks_cycle = self._round_robin_chunks(chunks_list)
        while len(pairs) < target_n:
            chunk = next(chunks_cycle)
            qa_type = self._rng.choice(self.types)
            try:
                new_pairs = self._generate_one(chunk, qa_type)
            except JudgeAuthError:
                raise
            except JudgeError:
                continue
            if not new_pairs:
                continue
            for p in new_pairs:
                if len(pairs) >= target_n:
                    break
                pairs.append(p)
        return pairs

    def _generate_parallel(
        self, chunks_list: list[Chunk], target_n: int
    ) -> list[QAPair]:
        # Plan one task per (chunk, qa_type), enough to hit target_n if every
        # call returns pairs_per_chunk pairs. Pre-sample all rng decisions
        # before fan-out so behavior stays deterministic for a given seed.
        n_tasks = max(1, math.ceil(target_n / max(1, self.pairs_per_chunk)))
        chunks_cycle = self._round_robin_chunks(chunks_list)
        plan: list[tuple[Chunk, QAType]] = []
        for _ in range(n_tasks):
            plan.append((next(chunks_cycle), self._rng.choice(self.types)))

        workers = min(self.max_workers, n_tasks)
        pairs: list[QAPair] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._generate_one, c, t) for c, t in plan]
            for f in futures:
                try:
                    new_pairs = f.result()
                except JudgeAuthError:
                    raise
                except JudgeError:
                    continue
                pairs.extend(new_pairs)

        return pairs[:target_n]

    def _generate_one(self, chunk: Chunk, qa_type: QAType) -> list[QAPair]:
        prompt = _USER_TEMPLATE.format(
            n=self.pairs_per_chunk,
            qa_type=qa_type.value,
            context=chunk.text,
        )
        result = self.judge.judge(_SYSTEM, prompt)
        raw_pairs = result.get("pairs")
        if not isinstance(raw_pairs, list):
            return []

        out: list[QAPair] = []
        for raw in raw_pairs:
            if not isinstance(raw, dict):
                continue
            question = str(raw.get("question", "")).strip()
            answer = str(raw.get("answer", "")).strip()
            if not question or not answer:
                continue
            difficulty: Difficulty | None = None
            diff_raw = str(raw.get("difficulty", "")).strip().lower()
            if diff_raw:
                try:
                    difficulty = Difficulty(diff_raw)
                except ValueError:
                    difficulty = None
            out.append(
                QAPair(
                    question=question,
                    answer=answer,
                    context=chunk.text,
                    qa_type=qa_type,
                    difficulty=difficulty,
                    source=chunk.source,
                    metadata={"chunk_index": chunk.chunk_index},
                )
            )
        return out

    def _round_robin_chunks(self, chunks: list[Chunk]) -> Iterator[Chunk]:
        idx = 0
        while True:
            yield chunks[idx % len(chunks)]
            idx += 1


def parse_pairs_jsonl(path: str) -> list[QAPair]:
    """Load QAPair records from a JSONL file."""
    out: list[QAPair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            out.append(QAPair.model_validate(data))
    return out
