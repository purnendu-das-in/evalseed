from evalseed.schemas import FilterResult, QAPair, QAType


def test_qapair_passed_with_no_filters(make_pair) -> None:
    p = make_pair()
    assert p.passed is True
    assert p.rejection_reasons == []


def test_qapair_passed_only_when_all_filters_pass(make_pair) -> None:
    p = make_pair()
    p.filter_results.append(FilterResult(filter_name="x", passed=True))
    p.filter_results.append(FilterResult(filter_name="y", passed=False, reason="bad"))
    assert p.passed is False
    assert p.rejection_reasons == ["bad"]


def test_qatype_values() -> None:
    assert QAType("single_hop") is QAType.SINGLE_HOP
    assert QAType("multi_hop") is QAType.MULTI_HOP
    assert QAType("distractor") is QAType.DISTRACTOR


def test_qapair_roundtrip(make_pair) -> None:
    p = make_pair()
    raw = p.model_dump_json()
    p2 = QAPair.model_validate_json(raw)
    assert p2.question == p.question
    assert p2.qa_type == p.qa_type
