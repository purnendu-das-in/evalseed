"""evalseed — quality-filtered synthetic Q&A datasets for RAG evaluation."""

from evalseed.dataset import Dataset
from evalseed.exceptions import (
    EvalseedError,
    FilterError,
    GenerationError,
    JudgeAuthError,
    JudgeError,
)
from evalseed.judges import Judge, OpenAIJudge
from evalseed.pipeline import Pipeline
from evalseed.schemas import FilterResult, QAPair, QAType

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "EvalseedError",
    "FilterError",
    "FilterResult",
    "GenerationError",
    "Judge",
    "JudgeAuthError",
    "JudgeError",
    "OpenAIJudge",
    "Pipeline",
    "QAPair",
    "QAType",
    "__version__",
]
