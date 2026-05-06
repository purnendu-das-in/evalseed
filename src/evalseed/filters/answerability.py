from __future__ import annotations

from evalseed.exceptions import JudgeAuthError, JudgeError
from evalseed.filters.base import Filter
from evalseed.schemas import FilterResult, QAPair

_SYSTEM = (
    "You evaluate whether a question is well-posed and answerable from a "
    "specific context, with a single defensible answer. Respond ONLY with JSON."
)

_USER_TEMPLATE = """Evaluate the QUESTION against the CONTEXT.

CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

Decide:
1. Is the question unambiguous (a careful reader would not produce multiple equally valid answers)?
2. Is the question answerable from the context alone (no external knowledge required)?
3. Is the question self-contained (does not rely on pronouns or "the above" referring outside itself)?

Return JSON:
{{
  "unambiguous": <true|false>,
  "answerable": <true|false>,
  "self_contained": <true|false>,
  "reason": "<one short sentence if any check fails, else empty string>"
}}
"""


class AnswerabilityFilter(Filter):
    """Rejects ambiguous, externally-dependent, or non-self-contained questions."""

    name = "answerability"

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

        unambiguous = bool(result.get("unambiguous", False))
        answerable = bool(result.get("answerable", False))
        self_contained = bool(result.get("self_contained", False))
        passed = unambiguous and answerable and self_contained
        reason = str(result.get("reason", "")).strip() or None
        if passed:
            return self._result(
                passed=True,
                unambiguous=unambiguous,
                answerable=answerable,
                self_contained=self_contained,
            )
        failed = [
            name
            for name, ok in (
                ("unambiguous", unambiguous),
                ("answerable", answerable),
                ("self_contained", self_contained),
            )
            if not ok
        ]
        return self._result(
            passed=False,
            reason=reason or f"failed: {', '.join(failed)}",
            unambiguous=unambiguous,
            answerable=answerable,
            self_contained=self_contained,
        )
