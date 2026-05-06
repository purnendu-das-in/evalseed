from __future__ import annotations

import json
import threading
from typing import Any, Protocol, runtime_checkable

from evalseed.exceptions import JudgeAuthError, JudgeError


@runtime_checkable
class Judge(Protocol):
    """A judge evaluates a prompt and returns structured JSON.

    Implementations should be deterministic-ish (low temperature) and return
    a dict parsed from the model's JSON output. Raise ``JudgeError`` on
    transport or parse failures so the pipeline can record the failure
    against the pair instead of crashing the run.
    """

    def judge(self, system: str, user: str) -> dict[str, Any]: ...

    def generate(self, system: str, user: str) -> str: ...


class OpenAIJudge:
    """OpenAI-backed judge using JSON-mode chat completions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise JudgeError(
                "openai package not installed — re-install evalseed: pip install -U evalseed"
            ) from exc

        try:
            self._client = OpenAI(
                api_key=api_key, max_retries=max_retries, timeout=timeout
            )
        except Exception as exc:
            raise JudgeAuthError(
                "OpenAI client could not be initialized — set OPENAI_API_KEY "
                f"or pass api_key=... ({exc})"
            ) from exc
        self.model = model
        self.temperature = temperature
        self._usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._usage_lock = threading.Lock()

    def usage(self) -> dict[str, int]:
        """Return cumulative token usage across all judge/generate calls."""
        with self._usage_lock:
            return dict(self._usage)

    def reset_usage(self) -> None:
        with self._usage_lock:
            for k in self._usage:
                self._usage[k] = 0

    def _record_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        with self._usage_lock:
            self._usage["calls"] += 1
            if usage is None:
                return
            for src, dst in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                val = getattr(usage, src, None)
                if isinstance(val, int):
                    self._usage[dst] += val

    @staticmethod
    def _wrap_request_exc(exc: Exception) -> JudgeError:
        try:
            from openai import AuthenticationError, PermissionDeniedError
        except ImportError:
            return JudgeError(f"OpenAI request failed: {exc}")
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return JudgeAuthError(f"OpenAI authentication failed: {exc}")
        return JudgeError(f"OpenAI request failed: {exc}")

    def judge(self, system: str, user: str) -> dict[str, Any]:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise self._wrap_request_exc(exc) from exc

        self._record_usage(resp)
        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"Judge returned non-JSON: {content[:200]}") from exc

        if not isinstance(parsed, dict):
            raise JudgeError(f"Judge returned non-object JSON: {type(parsed).__name__}")
        return parsed

    def generate(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise self._wrap_request_exc(exc) from exc

        self._record_usage(resp)
        return resp.choices[0].message.content or ""
