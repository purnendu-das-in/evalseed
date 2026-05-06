from __future__ import annotations

from evalseed.exceptions import JudgeAuthError, JudgeError
from evalseed.filters.base import Filter
from evalseed.judges import Judge
from evalseed.schemas import Difficulty, FilterResult, QAPair

_SYSTEM = (
    "You assess the reasoning effort required by a question, given its "
    "context. You output one of: 'easy', 'medium', 'hard'. Respond ONLY with JSON."
)

_USER_TEMPLATE = """Assess the difficulty of answering QUESTION from CONTEXT.

CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

Difficulty rubric:
- easy: single-span lookup, no reasoning required.
- medium: requires combining 2-3 facts, light synthesis, or simple inference.
- hard: requires multi-step reasoning, comparison across distant spans, or non-trivial inference.

Return JSON:
{{
  "predicted": "<easy|medium|hard>",
  "reason": "<one short sentence>"
}}
"""


_LEVELS = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}


class DifficultyFilter(Filter):
    """Records judge-predicted difficulty alongside the labeled value.

    By default this is a label-enrichment pass — the prediction is stored in
    metadata and the pair always passes. A 3-level rubric like easy/medium/hard
    is too coarse for one-step disagreements ("easy" vs "medium") to be a
    quality signal, so disagreement-based rejection is opt-in.

    Modes:
      strict=False (default): always pass; record prediction + agreement.
      strict=True: reject only on a TWO-step gap (easy↔hard).
    """

    name = "difficulty"

    def __init__(self, judge: Judge, strict: bool = False) -> None:
        super().__init__(judge)
        self.strict = strict

    def evaluate(self, pair: QAPair) -> FilterResult:
        try:
            result = self.judge.judge(
                _SYSTEM,
                _USER_TEMPLATE.format(context=pair.context, question=pair.question),
            )
        except JudgeAuthError:
            raise
        except JudgeError as exc:
            return self._result(passed=False, reason=f"judge error: {exc}")

        predicted_raw = str(result.get("predicted", "")).strip().lower()
        try:
            predicted = Difficulty(predicted_raw)
        except ValueError:
            return self._result(
                passed=True,
                reason=f"unparseable difficulty: {predicted_raw!r}",
                predicted=predicted_raw,
            )

        if pair.difficulty is None:
            return self._result(passed=True, predicted=predicted.value)

        gap = abs(_LEVELS[pair.difficulty] - _LEVELS[predicted])
        if not self.strict or gap < 2:
            return self._result(
                passed=True,
                labeled=pair.difficulty.value,
                predicted=predicted.value,
                gap=gap,
            )

        return self._result(
            passed=False,
            reason=(
                f"labeled difficulty {pair.difficulty.value!r} is two steps "
                f"away from judge prediction {predicted.value!r}"
            ),
            labeled=pair.difficulty.value,
            predicted=predicted.value,
            gap=gap,
        )
