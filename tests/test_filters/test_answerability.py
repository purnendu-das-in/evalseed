from evalseed.filters.answerability import AnswerabilityFilter
from tests.conftest import StubJudge


def test_answerability_passes_all_checks(make_pair) -> None:
    judge = StubJudge(
        default={
            "unambiguous": True,
            "answerable": True,
            "self_contained": True,
            "reason": "",
        }
    )
    res = AnswerabilityFilter(judge).evaluate(make_pair())
    assert res.passed


def test_answerability_rejects_ambiguous(make_pair) -> None:
    judge = StubJudge(
        default={
            "unambiguous": False,
            "answerable": True,
            "self_contained": True,
            "reason": "two valid readings",
        }
    )
    res = AnswerabilityFilter(judge).evaluate(make_pair())
    assert not res.passed
    assert res.reason == "two valid readings"


def test_answerability_lists_all_failed_checks(make_pair) -> None:
    judge = StubJudge(
        default={
            "unambiguous": False,
            "answerable": False,
            "self_contained": True,
            "reason": "",
        }
    )
    res = AnswerabilityFilter(judge).evaluate(make_pair())
    assert not res.passed
    assert "unambiguous" in (res.reason or "")
    assert "answerable" in (res.reason or "")
