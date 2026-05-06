from evalseed.filters.difficulty import DifficultyFilter
from evalseed.schemas import Difficulty
from tests.conftest import StubJudge


def test_difficulty_passes_when_label_matches(make_pair) -> None:
    judge = StubJudge(default={"predicted": "easy", "reason": ""})
    res = DifficultyFilter(judge).evaluate(make_pair(difficulty=Difficulty.EASY))
    assert res.passed


def test_difficulty_default_records_one_step_disagreement_without_rejecting(
    make_pair,
) -> None:
    judge = StubJudge(default={"predicted": "medium", "reason": ""})
    res = DifficultyFilter(judge).evaluate(make_pair(difficulty=Difficulty.EASY))
    assert res.passed
    assert res.metadata.get("labeled") == "easy"
    assert res.metadata.get("predicted") == "medium"
    assert res.metadata.get("gap") == 1


def test_difficulty_strict_passes_one_step_gap(make_pair) -> None:
    judge = StubJudge(default={"predicted": "medium", "reason": ""})
    res = DifficultyFilter(judge, strict=True).evaluate(
        make_pair(difficulty=Difficulty.EASY)
    )
    assert res.passed


def test_difficulty_strict_rejects_two_step_gap(make_pair) -> None:
    judge = StubJudge(default={"predicted": "hard", "reason": ""})
    res = DifficultyFilter(judge, strict=True).evaluate(
        make_pair(difficulty=Difficulty.EASY)
    )
    assert not res.passed
    assert "easy" in (res.reason or "")
    assert "hard" in (res.reason or "")


def test_difficulty_passes_unlabeled_and_records_prediction(make_pair) -> None:
    judge = StubJudge(default={"predicted": "medium", "reason": ""})
    res = DifficultyFilter(judge).evaluate(make_pair())
    assert res.passed
    assert res.metadata.get("predicted") == "medium"
