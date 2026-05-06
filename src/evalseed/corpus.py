from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from evalseed.exceptions import GenerationError


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a source document."""

    text: str
    source: str
    chunk_index: int


_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_corpus(path: str | Path) -> list[tuple[str, str]]:
    """Load (source_name, text) pairs from a file or directory.

    Supports plain-text formats only in v0.1 (.txt, .md, .markdown, .rst).
    PDF/HTML loaders are intentionally deferred — users can preprocess.
    """
    p = Path(path)
    if not p.exists():
        raise GenerationError(f"corpus path does not exist: {p}")

    if p.is_file():
        return [(p.name, _read_text(p))]

    docs: list[tuple[str, str]] = []
    for sub in sorted(p.rglob("*")):
        if sub.is_file() and sub.suffix.lower() in _SUPPORTED_SUFFIXES:
            docs.append((str(sub.relative_to(p)), _read_text(sub)))
    if not docs:
        raise GenerationError(
            f"no supported files (.txt/.md/.rst) found under {p}"
        )
    return docs


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(
    text: str,
    source: str,
    target_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Paragraph-aware char-window chunker.

    Greedily packs paragraphs into windows of roughly ``target_chars``,
    with character overlap between consecutive windows. Good enough for
    v0.1; more sophisticated chunking is a v0.2 concern.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be in [0, target_chars)")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, buf_len, idx
        if not buf:
            return
        chunk_text_value = "\n\n".join(buf)
        chunks.append(Chunk(text=chunk_text_value, source=source, chunk_index=idx))
        idx += 1
        if overlap_chars and len(chunk_text_value) > overlap_chars:
            tail = chunk_text_value[-overlap_chars:]
            buf = [tail]
            buf_len = len(tail)
        else:
            buf = []
            buf_len = 0

    for para in paragraphs:
        if buf_len + len(para) + 2 > target_chars and buf:
            flush()
        buf.append(para)
        buf_len += len(para) + 2

    flush()
    return chunks
