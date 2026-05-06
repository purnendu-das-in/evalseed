from evalseed.filters.prefilters import LengthPreFilter, RegexPreFilter


def test_length_prefilter_passes_normal_pair(make_pair) -> None:
    pf = LengthPreFilter()
    res = pf.evaluate(make_pair())
    assert res.passed


def test_length_prefilter_rejects_short_question(make_pair) -> None:
    pf = LengthPreFilter(min_question_chars=15)
    res = pf.evaluate(make_pair(question="too short?"))
    assert not res.passed
    assert "too short" in (res.reason or "")


def test_length_prefilter_rejects_long_answer(make_pair) -> None:
    pf = LengthPreFilter(max_answer_chars=10)
    res = pf.evaluate(make_pair(answer="x" * 50))
    assert not res.passed


def test_regex_prefilter_passes_clean(make_pair) -> None:
    pf = RegexPreFilter()
    res = pf.evaluate(make_pair())
    assert res.passed


def test_regex_prefilter_rejects_meta_question(make_pair) -> None:
    pf = RegexPreFilter()
    res = pf.evaluate(
        make_pair(question="What is the above text describing in detail?")
    )
    assert not res.passed


def test_regex_prefilter_rejects_refusal(make_pair) -> None:
    pf = RegexPreFilter()
    res = pf.evaluate(
        make_pair(question="As an AI language model, what should I answer here?")
    )
    assert not res.passed
