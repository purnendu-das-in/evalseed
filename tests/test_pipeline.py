from pathlib import Path
from typing import Any

import pytest

from evalseed.exceptions import JudgeAuthError
from evalseed.pipeline import Pipeline
from tests.conftest import StubJudge


def _judge_routes() -> dict[str, Any]:
    return {
        "high-quality question-answer pairs": lambda s, u: {
            "pairs": [
                {
                    "question": "What is the capital of France?",
                    "answer": "Paris.",
                    "difficulty": "easy",
                },
                {
                    "question": "Which European country has Paris as its capital?",
                    "answer": "France.",
                    "difficulty": "easy",
                },
            ]
        },
        "factual grounding": lambda s, u: {"faithful": True, "score": 0.95, "reason": ""},
        "well-posed and answerable": lambda s, u: {
            "unambiguous": True,
            "answerable": True,
            "self_contained": True,
            "reason": "",
        },
        "verbatim restatement": lambda s, u: {"trivial": False, "reason": ""},
        "reasoning effort required": lambda s, u: {"predicted": "easy", "reason": ""},
    }


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(
        "France is a country in Western Europe.\n\n"
        "Its capital is Paris, which sits on the river Seine.\n\n"
        + "\n\n".join(f"More content paragraph {i}." for i in range(5)),
        encoding="utf-8",
    )

    judge = StubJudge(routes=_judge_routes())
    pipeline = Pipeline(judge=judge, n_pairs=2, verbose=False, seed=0)
    dataset = pipeline.generate_from_corpus(corpus)

    assert len(dataset) == 2
    assert all(p.passed for p in dataset)
    stats = dataset.stats()
    assert stats["passed"] == 2


def test_pipeline_short_circuits_on_first_failure(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "d.md").write_text("alpha beta gamma.\n\ndelta epsilon zeta.", encoding="utf-8")

    routes = _judge_routes()
    routes["factual grounding"] = lambda s, u: {
        "faithful": False,
        "score": 0.1,
        "reason": "hallucinated",
    }
    judge = StubJudge(routes=routes)

    pipeline = Pipeline(
        judge=judge, n_pairs=1, verbose=False, seed=0, max_attempts=1
    )
    dataset = pipeline.generate_from_corpus(corpus)
    assert len(dataset) >= 1
    assert not dataset[0].passed
    filter_names_run = [r.filter_name for r in dataset[0].filter_results]
    assert "faithfulness" in filter_names_run
    assert "answerability" not in filter_names_run
    assert "triviality" not in filter_names_run


def test_pipeline_hits_target_with_partial_pass_rate(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "d.md").write_text(
        "\n\n".join(f"Paragraph {i} about France and Paris." for i in range(8)),
        encoding="utf-8",
    )

    routes = _judge_routes()
    gen_count = {"n": 0}

    def fresh_pairs(s: str, u: str) -> dict[str, Any]:
        gen_count["n"] += 1
        i = gen_count["n"]
        return {
            "pairs": [
                {
                    "question": f"What unique fact #{i}-a is in the context?",
                    "answer": "Some fact.",
                    "difficulty": "easy",
                },
                {
                    "question": f"Which detail #{i}-b appears in the context?",
                    "answer": "Some detail.",
                    "difficulty": "easy",
                },
            ]
        }

    routes["high-quality question-answer pairs"] = fresh_pairs

    faith_count = {"n": 0}

    def flaky_faithfulness(s: str, u: str) -> dict[str, Any]:
        faith_count["n"] += 1
        if faith_count["n"] % 2 == 0:
            return {"faithful": False, "score": 0.1, "reason": "rejected"}
        return {"faithful": True, "score": 0.95, "reason": ""}

    routes["factual grounding"] = flaky_faithfulness
    judge = StubJudge(routes=routes)

    pipeline = Pipeline(judge=judge, n_pairs=4, verbose=False, seed=0)
    dataset = pipeline.generate_from_corpus(corpus)
    assert sum(1 for p in dataset if p.passed) == 4
    questions = [p.question for p in dataset]
    assert len(questions) == len(set(questions))


def test_pipeline_dedupes_questions(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "d.md").write_text(
        "\n\n".join(f"Paragraph {i} about France and Paris." for i in range(8)),
        encoding="utf-8",
    )

    routes = _judge_routes()
    routes["high-quality question-answer pairs"] = lambda s, u: {
        "pairs": [
            {
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "difficulty": "easy",
            },
            {
                "question": "  what  IS the CAPITAL of france??  ",
                "answer": "Paris.",
                "difficulty": "easy",
            },
        ]
    }
    judge = StubJudge(routes=routes)
    pipeline = Pipeline(
        judge=judge, n_pairs=1, verbose=False, seed=0, max_attempts=1
    )
    dataset = pipeline.generate_from_corpus(corpus)
    questions = [p.question for p in dataset]
    assert len(questions) == 1


def test_pipeline_propagates_auth_error(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "d.md").write_text("alpha beta gamma.\n\ndelta epsilon zeta.", encoding="utf-8")

    class AuthFailingJudge:
        def judge(self, system: str, user: str) -> dict[str, Any]:
            raise JudgeAuthError("invalid api key")

        def generate(self, system: str, user: str) -> str:
            raise JudgeAuthError("invalid api key")

    pipeline = Pipeline(judge=AuthFailingJudge(), n_pairs=1, verbose=False, seed=0)
    with pytest.raises(JudgeAuthError):
        pipeline.generate_from_corpus(corpus)
