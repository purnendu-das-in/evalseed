from evalseed.filters.faithfulness import FaithfulnessFilter
from tests.conftest import StubJudge


def test_faithfulness_passes_when_judge_says_faithful(make_pair) -> None:
    judge = StubJudge(default={"faithful": True, "score": 0.95, "reason": ""})
    f = FaithfulnessFilter(judge)
    res = f.evaluate(make_pair())
    assert res.passed
    assert res.score == 0.95


def test_faithfulness_rejects_when_judge_says_unfaithful(make_pair) -> None:
    judge = StubJudge(
        default={"faithful": False, "score": 0.2, "reason": "answer adds dates"}
    )
    f = FaithfulnessFilter(judge)
    res = f.evaluate(make_pair())
    assert not res.passed
    assert "adds dates" in (res.reason or "")


def test_faithfulness_rejects_below_threshold(make_pair) -> None:
    judge = StubJudge(default={"faithful": True, "score": 0.5, "reason": "weak"})
    f = FaithfulnessFilter(judge, threshold=0.7)
    res = f.evaluate(make_pair())
    assert not res.passed
