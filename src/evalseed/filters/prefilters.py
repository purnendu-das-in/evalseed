from __future__ import annotations

import re

from evalseed.filters.base import PreFilter
from evalseed.schemas import FilterResult, QAPair


class LengthPreFilter(PreFilter):
    """Reject pairs whose question or answer is implausibly short or long."""

    name = "length_prefilter"

    def __init__(
        self,
        min_question_chars: int = 15,
        max_question_chars: int = 600,
        min_answer_chars: int = 1,
        max_answer_chars: int = 2000,
    ) -> None:
        self.min_q = min_question_chars
        self.max_q = max_question_chars
        self.min_a = min_answer_chars
        self.max_a = max_answer_chars

    def evaluate(self, pair: QAPair) -> FilterResult:
        q_len = len(pair.question.strip())
        a_len = len(pair.answer.strip())

        if q_len < self.min_q:
            return FilterResult(
                filter_name=self.name,
                passed=False,
                reason=f"question too short ({q_len} < {self.min_q})",
            )
        if q_len > self.max_q:
            return FilterResult(
                filter_name=self.name,
                passed=False,
                reason=f"question too long ({q_len} > {self.max_q})",
            )
        if a_len < self.min_a:
            return FilterResult(
                filter_name=self.name,
                passed=False,
                reason=f"answer too short ({a_len} < {self.min_a})",
            )
        if a_len > self.max_a:
            return FilterResult(
                filter_name=self.name,
                passed=False,
                reason=f"answer too long ({a_len} > {self.max_a})",
            )
        return FilterResult(filter_name=self.name, passed=True)


class RegexPreFilter(PreFilter):
    """Reject pairs whose question matches an obvious-junk regex.

    Default patterns catch common RAGAS failure modes: meta-questions about
    the prompt itself, refusals, and questions that are just the heading text.
    """

    name = "regex_prefilter"

    DEFAULT_PATTERNS: tuple[str, ...] = (
        r"^\s*(what|which) (is|are) (the|this) (above|following|given|provided|preceding) ",
        r"\bas an? (ai|language model)\b",
        r"\bi (cannot|can't|won't|am unable to)\b",
        r"^\s*(yes|no)\s*[.?!]?\s*$",
    )

    def __init__(self, patterns: tuple[str, ...] | None = None) -> None:
        compiled_patterns = patterns if patterns is not None else self.DEFAULT_PATTERNS
        self.patterns = [re.compile(p, re.IGNORECASE) for p in compiled_patterns]

    def evaluate(self, pair: QAPair) -> FilterResult:
        for p in self.patterns:
            if p.search(pair.question):
                return FilterResult(
                    filter_name=self.name,
                    passed=False,
                    reason=f"question matches junk pattern: {p.pattern!r}",
                )
        return FilterResult(filter_name=self.name, passed=True)
