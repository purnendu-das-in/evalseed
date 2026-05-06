from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evalseed.corpus import Chunk, chunk_text, load_corpus
from evalseed.dataset import Dataset
from evalseed.filters import (
    AnswerabilityFilter,
    DifficultyFilter,
    FaithfulnessFilter,
    LengthPreFilter,
    RegexPreFilter,
    TrivialityFilter,
)
from evalseed.filters.base import Filter, PreFilter
from evalseed.generator import QAGenerator
from evalseed.judges import Judge
from evalseed.schemas import QAPair, QAType

_TRAILING_PUNCT = re.compile(r"[\s\.\?!,;:]+$")
_WHITESPACE = re.compile(r"\s+")


def _normalize_question(q: str) -> str:
    """Normalize for dedup: lowercase, collapse whitespace, strip trailing punctuation."""
    text = _WHITESPACE.sub(" ", q.strip().lower())
    return _TRAILING_PUNCT.sub("", text)


class Pipeline:
    """Orchestrates: load corpus → chunk → generate → filter → dataset.

    The five-line API:

        from evalseed import Pipeline, OpenAIJudge
        pipeline = Pipeline(judge=OpenAIJudge(), n_pairs=100)
        dataset = pipeline.generate_from_corpus("./docs/")
        dataset.save("eval.jsonl")

    Filters run in order; the first failing filter short-circuits the rest
    for that pair, since the pair will be rejected anyway and judge calls
    cost money.
    """

    def __init__(
        self,
        judge: Judge,
        n_pairs: int = 50,
        types: Sequence[QAType | str] = (QAType.SINGLE_HOP, QAType.MULTI_HOP),
        pairs_per_chunk: int = 2,
        chunk_chars: int = 1500,
        chunk_overlap: int = 150,
        prefilters: list[PreFilter] | None = None,
        filters: list[Filter] | None = None,
        generator: QAGenerator | None = None,
        seed: int | None = None,
        verbose: bool = True,
        oversample_factor: float = 1.6,
        max_attempts: int = 6,
        max_workers: int = 8,
    ) -> None:
        if oversample_factor < 1.0:
            raise ValueError("oversample_factor must be >= 1.0")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.judge = judge
        self.n_pairs = n_pairs
        self.types = types
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.oversample_factor = oversample_factor
        self.max_attempts = max_attempts
        self.max_workers = max_workers
        self.generator = generator or QAGenerator(
            judge=judge,
            types=types,
            pairs_per_chunk=pairs_per_chunk,
            seed=seed,
            max_workers=max_workers,
        )
        self.prefilters: list[PreFilter] = (
            prefilters
            if prefilters is not None
            else [LengthPreFilter(), RegexPreFilter()]
        )
        self.filters: list[Filter] = (
            filters
            if filters is not None
            else [
                FaithfulnessFilter(judge),
                AnswerabilityFilter(judge),
                TrivialityFilter(judge),
                DifficultyFilter(judge),
            ]
        )

    def generate_from_corpus(self, path: str | Path) -> Dataset:
        docs = load_corpus(path)
        chunks: list[Chunk] = []
        for source, text in docs:
            chunks.extend(
                chunk_text(
                    text,
                    source,
                    target_chars=self.chunk_chars,
                    overlap_chars=self.chunk_overlap,
                )
            )
        if self.verbose:
            self._log(f"loaded {len(docs)} document(s), {len(chunks)} chunk(s)")
        return self.generate_from_chunks(chunks)

    def generate_from_chunks(self, chunks: Sequence[Chunk]) -> Dataset:
        if self.verbose:
            self._log(f"target: {self.n_pairs} passed pair(s)")

        all_pairs: list[QAPair] = []
        passed_count = 0
        seen_questions: set[str] = set()
        duplicates_dropped = 0
        for attempt in range(1, self.max_attempts + 1):
            missing = self.n_pairs - passed_count
            if missing <= 0:
                break
            batch_size = max(1, int(round(missing * self.oversample_factor)))
            if self.verbose:
                self._log(
                    f"attempt {attempt}/{self.max_attempts}: "
                    f"generating {batch_size} (need {missing} more)"
                )
            raw = self.generator.generate(chunks, target_n=batch_size)
            fresh: list[QAPair] = []
            for pair in raw:
                key = _normalize_question(pair.question)
                if key in seen_questions:
                    duplicates_dropped += 1
                    continue
                seen_questions.add(key)
                fresh.append(pair)

            self._apply_filters_batch(fresh)

            for pair in fresh:
                if pair.passed:
                    passed_count += 1
                all_pairs.append(pair)

        if passed_count > self.n_pairs:
            trimmed: list[QAPair] = []
            kept_passed = 0
            for pair in all_pairs:
                if pair.passed:
                    if kept_passed >= self.n_pairs:
                        continue
                    kept_passed += 1
                trimmed.append(pair)
            all_pairs = trimmed
            passed_count = kept_passed

        dataset = Dataset(all_pairs)
        if duplicates_dropped and self.verbose:
            self._log(f"dropped {duplicates_dropped} duplicate question(s)")
        if passed_count < self.n_pairs and self.verbose:
            self._log_shortfall(passed_count, len(all_pairs))
        if self.verbose:
            self._log_stats(dataset)
            self._log_token_usage()
        return dataset

    def filter_pairs(self, pairs: list[QAPair]) -> Dataset:
        self._apply_filters_batch(pairs)
        dataset = Dataset(pairs)
        if self.verbose:
            self._log_stats(dataset)
        return dataset

    def _apply_filters_batch(self, pairs: list[QAPair]) -> None:
        if not pairs:
            return
        if self.max_workers <= 1 or len(pairs) == 1:
            for pair in pairs:
                self._apply_filters(pair)
            return
        workers = min(self.max_workers, len(pairs))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(self._apply_filters, pairs))

    def _apply_filters(self, pair: QAPair) -> None:
        for pf in self.prefilters:
            res = pf.evaluate(pair)
            pair.filter_results.append(res)
            if not res.passed:
                return
        for f in self.filters:
            res = f.evaluate(pair)
            pair.filter_results.append(res)
            if not res.passed:
                return

    def _log_token_usage(self) -> None:
        usage_fn = getattr(self.judge, "usage", None)
        if not callable(usage_fn):
            return
        try:
            usage = usage_fn()
        except Exception:
            return
        if not isinstance(usage, dict) or not usage:
            return
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Token usage", show_header=True)
            table.add_column("metric")
            table.add_column("value", justify="right")
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                if key in usage:
                    table.add_row(key, f"{usage[key]:,}")
            console.print(table)
        except ImportError:
            print(f"[evalseed] token usage: {usage}")

    def _log_shortfall(self, passed: int, total: int) -> None:
        msg = (
            f"only {passed}/{self.n_pairs} pair(s) passed after "
            f"{self.max_attempts} attempt(s) ({total} generated). "
            f"Pass rate too low — check corpus quality or relax filter thresholds."
        )
        try:
            from rich import print as rprint

            rprint(f"[bold yellow][evalseed][/] WARNING: {msg}")
        except ImportError:
            print(f"[evalseed] WARNING: {msg}")

    def _log(self, msg: str) -> None:
        try:
            from rich import print as rprint

            rprint(f"[bold cyan][evalseed][/] {msg}")
        except ImportError:
            print(f"[evalseed] {msg}")

    def _log_stats(self, dataset: Dataset) -> None:
        stats = dataset.stats()
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            console.print(
                f"[bold green][evalseed][/] kept {stats['passed']}/"
                f"{stats['total']} ({stats['pass_rate']:.0%})"
            )
            rejections = stats["rejections_by_filter"]
            if isinstance(rejections, dict) and rejections:
                table = Table(title="Rejections by filter", show_header=True)
                table.add_column("filter")
                table.add_column("count", justify="right")
                for name, count in sorted(rejections.items(), key=lambda x: -x[1]):
                    table.add_row(name, str(count))
                console.print(table)
        except ImportError:
            print(f"[evalseed] stats: {stats}")
