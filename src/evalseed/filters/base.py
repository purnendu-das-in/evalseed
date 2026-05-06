from __future__ import annotations

from abc import ABC, abstractmethod

from evalseed.judges import Judge
from evalseed.schemas import FilterResult, QAPair


class PreFilter(ABC):
    """Cheap, judge-free filter — runs before any LLM calls."""

    name: str

    @abstractmethod
    def evaluate(self, pair: QAPair) -> FilterResult: ...


class Filter(ABC):
    """LLM-judge-backed filter."""

    name: str

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    @abstractmethod
    def evaluate(self, pair: QAPair) -> FilterResult: ...

    def _result(
        self,
        passed: bool,
        score: float | None = None,
        reason: str | None = None,
        **metadata: object,
    ) -> FilterResult:
        return FilterResult(
            filter_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metadata=dict(metadata),
        )
