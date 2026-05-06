from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from evalseed.judges import Judge
from evalseed.schemas import QAPair, QAType


class StubJudge:
    """Deterministic judge that returns scripted responses by route key.

    The pipeline routes calls by the first line of the system prompt; the
    stub matches on substrings of the system prompt to pick a handler.
    """

    def __init__(
        self,
        routes: dict[str, Callable[[str, str], dict[str, Any]]] | None = None,
        default: dict[str, Any] | None = None,
        text_default: str = "",
    ) -> None:
        self.routes = routes or {}
        self.default = default if default is not None else {}
        self.text_default = text_default
        self.calls: list[tuple[str, str]] = []

    def judge(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        for marker, handler in self.routes.items():
            if marker in system:
                return handler(system, user)
        return dict(self.default)

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.text_default


assert isinstance(StubJudge(), Judge)


@pytest.fixture
def make_pair() -> Callable[..., QAPair]:
    def _make(
        question: str = "What is the capital of France?",
        answer: str = "Paris.",
        context: str = "France is a country in Western Europe. Its capital is Paris.",
        qa_type: QAType = QAType.SINGLE_HOP,
        **kwargs: Any,
    ) -> QAPair:
        return QAPair(
            question=question,
            answer=answer,
            context=context,
            qa_type=qa_type,
            **kwargs,
        )

    return _make


@pytest.fixture
def stub_judge() -> StubJudge:
    return StubJudge()
