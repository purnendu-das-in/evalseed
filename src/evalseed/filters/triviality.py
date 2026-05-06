from __future__ import annotations

import re

from evalseed.exceptions import JudgeAuthError, JudgeError
from evalseed.filters.base import Filter
from evalseed.judges import Judge
from evalseed.schemas import FilterResult, QAPair

_SYSTEM = (
    "You judge whether a RAG-evaluation question is degenerate — i.e. so "
    "lexically aligned with the context that it tests string matching rather "
    "than retrieval or comprehension. Respond ONLY with JSON."
)

_USER_TEMPLATE = """The QUESTION below was generated from CONTEXT for use in a
RAG-evaluation dataset. Grounded factual lookup is the GOAL of such datasets,
not a defect — a question whose answer can be retrieved from a single span is
fine, as long as the question itself is not a verbatim re-rendering of that
span.

CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION: {question}
ANSWER: {answer}

Mark the question "trivial" ONLY if at least one of these holds:
  (a) the question reuses a long contiguous phrase from the context (5+ words
      in a row), so a string-match retriever wins without any understanding;
  (b) the answer is essentially a single noun or noun phrase that appears
      immediately adjacent to the question's exact wording in the context
      (the question is the context sentence with one word replaced by "what");
  (c) the question gives away the answer (e.g. "What is X, which is Y?").

Otherwise — including normal factual lookups, definitional questions, and
list-extraction questions phrased in the user's own words — mark it
non-trivial. Concise factual questions are the bread and butter of RAG eval
and should pass.

Return JSON:
{{
  "trivial": <true|false>,
  "reason": "<one short sentence>"
}}
"""


def _ngram_overlap(a: str, b: str, n: int = 5) -> float:
    """Fraction of n-grams from `a` that also appear in `b`."""
    tokens_a = re.findall(r"\w+", a.lower())
    tokens_b = re.findall(r"\w+", b.lower())
    if len(tokens_a) < n:
        return 0.0
    grams_a = {tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)}
    grams_b = {tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)}
    if not grams_a:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a)


class TrivialityFilter(Filter):
    """Rejects pairs that are verbatim restatements of the source.

    Uses a cheap n-gram overlap shortcut before falling back to the judge.
    """

    name = "triviality"

    def __init__(self, judge: Judge, ngram_overlap_threshold: float = 0.8) -> None:
        super().__init__(judge)
        self.ngram_overlap_threshold = ngram_overlap_threshold

    def evaluate(self, pair: QAPair) -> FilterResult:
        overlap = _ngram_overlap(pair.question, pair.context)
        if overlap >= self.ngram_overlap_threshold:
            return self._result(
                passed=False,
                score=overlap,
                reason=f"question is a near-verbatim span (5-gram overlap={overlap:.2f})",
                ngram_overlap=overlap,
            )

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

        trivial = bool(result.get("trivial", False))
        reason = str(result.get("reason", "")).strip() or None
        return self._result(
            passed=not trivial,
            score=overlap,
            reason=reason if trivial else None,
            ngram_overlap=overlap,
        )
