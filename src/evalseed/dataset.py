from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import overload

from evalseed.schemas import QAPair


class Dataset:
    """A collection of QA pairs with serialization and stats helpers.

    Iteration yields all pairs (passed and rejected). Use ``passed`` /
    ``rejected`` to filter, or ``save`` to write only passed pairs to disk.
    """

    def __init__(self, pairs: list[QAPair]) -> None:
        self._pairs: list[QAPair] = list(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[QAPair]:
        return iter(self._pairs)

    @overload
    def __getitem__(self, key: int) -> QAPair: ...
    @overload
    def __getitem__(self, key: slice) -> Dataset: ...
    def __getitem__(self, key: int | slice) -> QAPair | Dataset:
        if isinstance(key, slice):
            return Dataset(self._pairs[key])
        return self._pairs[key]

    @property
    def passed(self) -> Dataset:
        return Dataset([p for p in self._pairs if p.passed])

    @property
    def rejected(self) -> Dataset:
        return Dataset([p for p in self._pairs if not p.passed])

    def stats(self) -> dict[str, object]:
        total = len(self._pairs)
        passed = sum(1 for p in self._pairs if p.passed)
        rejection_counts: Counter[str] = Counter()
        for pair in self._pairs:
            for r in pair.filter_results:
                if not r.passed:
                    rejection_counts[r.filter_name] += 1
        return {
            "total": total,
            "passed": passed,
            "rejected": total - passed,
            "pass_rate": passed / total if total else 0.0,
            "rejections_by_filter": dict(rejection_counts),
        }

    def save(self, path: str | Path, only_passed: bool = True) -> None:
        """Write pairs to a JSONL file. By default writes only passed pairs."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pairs = self.passed if only_passed else self
        with target.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(pair.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        pairs: list[QAPair] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs.append(QAPair.model_validate(json.loads(line)))
        return cls(pairs)
