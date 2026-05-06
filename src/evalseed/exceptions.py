class EvalseedError(Exception):
    """Base class for all evalseed errors."""


class GenerationError(EvalseedError):
    """Raised when QA pair generation fails."""


class FilterError(EvalseedError):
    """Raised when a filter cannot evaluate a pair."""


class JudgeError(EvalseedError):
    """Raised when the underlying judge LLM call fails."""


class JudgeAuthError(JudgeError):
    """Raised when the judge cannot authenticate (missing or invalid API key).

    Distinct from JudgeError so callers that normally swallow transient
    judge failures can let auth errors surface — retrying a bad key is
    pointless and would silently produce empty results.
    """
