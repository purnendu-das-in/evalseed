from pathlib import Path

from evalseed.dataset import Dataset
from evalseed.schemas import FilterResult


def test_stats_counts_rejections(make_pair) -> None:
    p1 = make_pair()
    p1.filter_results.append(FilterResult(filter_name="a", passed=True))
    p2 = make_pair()
    p2.filter_results.append(FilterResult(filter_name="a", passed=False, reason="x"))
    p3 = make_pair()
    p3.filter_results.append(FilterResult(filter_name="b", passed=False, reason="y"))

    ds = Dataset([p1, p2, p3])
    stats = ds.stats()
    assert stats["total"] == 3
    assert stats["passed"] == 1
    assert stats["rejected"] == 2
    assert stats["rejections_by_filter"] == {"a": 1, "b": 1}


def test_passed_and_rejected_views(make_pair) -> None:
    p_good = make_pair()
    p_bad = make_pair()
    p_bad.filter_results.append(FilterResult(filter_name="a", passed=False))
    ds = Dataset([p_good, p_bad])
    assert len(ds.passed) == 1
    assert len(ds.rejected) == 1


def test_save_and_load_roundtrip(make_pair, tmp_path: Path) -> None:
    p = make_pair()
    ds = Dataset([p])
    out = tmp_path / "eval.jsonl"
    ds.save(out)
    loaded = Dataset.load(out)
    assert len(loaded) == 1
    assert loaded[0].question == p.question


def test_save_only_passed_by_default(make_pair, tmp_path: Path) -> None:
    p_good = make_pair()
    p_bad = make_pair(question="Different question entirely?")
    p_bad.filter_results.append(FilterResult(filter_name="a", passed=False))
    out = tmp_path / "eval.jsonl"
    Dataset([p_good, p_bad]).save(out)
    loaded = Dataset.load(out)
    assert len(loaded) == 1
