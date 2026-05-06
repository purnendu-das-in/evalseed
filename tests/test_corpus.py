from pathlib import Path

import pytest

from evalseed.corpus import chunk_text, load_corpus
from evalseed.exceptions import GenerationError


def test_chunk_text_packs_paragraphs() -> None:
    text = "\n\n".join(["para " + str(i) + " " + "x" * 200 for i in range(10)])
    chunks = chunk_text(text, source="t.md", target_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert all(c.source == "t.md" for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_chunk_text_empty_input() -> None:
    assert chunk_text("", source="t.md") == []
    assert chunk_text("   \n\n  \n", source="t.md") == []


def test_chunk_text_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", source="t.md", target_chars=100, overlap_chars=100)


def test_load_corpus_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "c.bin").write_text("ignored", encoding="utf-8")
    docs = load_corpus(tmp_path)
    sources = {s for s, _ in docs}
    assert sources == {"a.md", "b.txt"}


def test_load_corpus_missing_path(tmp_path: Path) -> None:
    with pytest.raises(GenerationError):
        load_corpus(tmp_path / "nope")


def test_load_corpus_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(GenerationError):
        load_corpus(tmp_path)
