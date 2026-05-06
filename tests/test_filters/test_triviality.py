from evalseed.filters.triviality import TrivialityFilter, _ngram_overlap
from tests.conftest import StubJudge


def test_ngram_overlap_full_match() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert _ngram_overlap(text, text) == 1.0


def test_ngram_overlap_no_match() -> None:
    assert _ngram_overlap("alpha beta gamma delta epsilon", "one two three four five") == 0.0


def test_triviality_short_circuits_on_high_overlap(make_pair) -> None:
    judge = StubJudge(default={"trivial": False, "reason": ""})
    context = "Section 4.2 describes the capital adequacy ratio for insurers."
    pair = make_pair(
        question="Section 4.2 describes the capital adequacy ratio for insurers.",
        context=context,
    )
    res = TrivialityFilter(judge, ngram_overlap_threshold=0.8).evaluate(pair)
    assert not res.passed
    assert "verbatim" in (res.reason or "")
    assert judge.calls == []


def test_triviality_falls_back_to_judge(make_pair) -> None:
    judge = StubJudge(default={"trivial": True, "reason": "single span lookup"})
    res = TrivialityFilter(judge).evaluate(
        make_pair(question="What is the capital adequacy ratio mentioned for insurers?")
    )
    assert not res.passed
    assert res.reason == "single span lookup"


def test_triviality_passes_substantive_question(make_pair) -> None:
    judge = StubJudge(default={"trivial": False, "reason": ""})
    res = TrivialityFilter(judge).evaluate(make_pair())
    assert res.passed
