from __future__ import annotations

from evalseed.exceptions import JudgeAuthError, JudgeError
from evalseed.filters.base import Filter
from evalseed.judges import Judge
from evalseed.schemas import FilterResult, QAPair

_SYSTEM = (
    "You are a strict evaluator of factual grounding. You decide whether an "
    "answer is fully supported by the provided context, with no information "
    "added from outside the context. Respond ONLY with JSON."
)

_USER_TEMPLATE = """Evaluate whether the ANSWER is fully entailed by the CONTEXT for the QUESTION.

CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

ANSWER: {answer}

Return JSON with this exact schema:
{{
  "faithful": <true|false>,
  "score": <float between 0 and 1>,
  "reason": "<one short sentence>"
}}

Rules:
- "faithful" must be false if any factual claim in the answer is not supported by the context.
- "faithful" must be false if the answer adds quantitative details, dates, or names not in the context.
- Paraphrase is OK as long as every claim is supported.
"""


class FaithfulnessFilter(Filter):
    """Rejects pairs where the answer cannot be entailed from the context."""

    name = "faithfulness"

    def __init__(self, judge: Judge, threshold: float = 0.7) -> None:
        super().__init__(judge)
        self.threshold = threshold

    def evaluate(self, pair: QAPair) -> FilterResult:
        try:
            result = self.judge.judge(
                _SYSTEM,
                _USER_TEMPLATE.format(
                    context=pair.context,
                    question=pair.question,
                    answer=pair.answer,
                ),
            )
        except JudgeAuthError:
            raise
        except JudgeError as exc:
            return self._result(passed=False, reason=f"judge error: {exc}")

        faithful = bool(result.get("faithful", False))
        score_raw = result.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        reason = str(result.get("reason", "")).strip() or None
        passed = faithful and score >= self.threshold
        return self._result(passed=passed, score=score, reason=reason if not passed else None)
