from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from evalseed.exceptions import JudgeAuthError
from evalseed.judges import OpenAIJudge


def _fake_resp(content: str, prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_: Any) -> Any:
        if not self._responses:
            raise RuntimeError("no more fake responses")
        return self._responses.pop(0)


def _build_judge(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> OpenAIJudge:
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    judge = OpenAIJudge()
    judge._client = client  # type: ignore[assignment]
    judge.reset_usage()
    return judge


def test_openai_judge_tracks_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(
        [
            _fake_resp('{"ok": true}', prompt=120, completion=30),
            _fake_resp('{"ok": false}', prompt=80, completion=20),
        ]
    )
    judge = _build_judge(monkeypatch, client)

    judge.judge("sys", "u1")
    judge.judge("sys", "u2")

    usage = judge.usage()
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 250


def test_openai_judge_missing_key_raises_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeAuthError):
        OpenAIJudge()
