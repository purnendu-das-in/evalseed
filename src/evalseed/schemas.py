from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class QAType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    DISTRACTOR = "distractor"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FilterResult(BaseModel):
    """Outcome of a single filter stage applied to a QA pair."""

    filter_name: str
    passed: bool
    score: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAPair(BaseModel):
    """A single question-answer pair anchored to its source context."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    question: str
    answer: str
    context: str
    qa_type: QAType
    difficulty: Difficulty | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    filter_results: list[FilterResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if every filter that has run passed."""
        return all(r.passed for r in self.filter_results)

    @property
    def rejection_reasons(self) -> list[str]:
        return [r.reason or r.filter_name for r in self.filter_results if not r.passed]
