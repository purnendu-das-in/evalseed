# Contributing to evalseed

Thanks for considering a contribution. evalseed is small and focused on purpose — please read this short guide before opening a PR.

## Scope

evalseed v0.1 deliberately stays minimal. The following are out of scope until v0.2 and shouldn't land in PRs without prior discussion:

- Multi-provider judges (Anthropic, Gemini, Bedrock, local models)
- Async / concurrent generation
- PDF, HTML, or DOCX corpus loaders
- Web UIs or dashboards
- Integrations with RAGAS, DeepEval, Langfuse, etc.

If you want to work on any of these, please open an issue first to discuss.

## Setup

```bash
git clone https://github.com/<your-fork>/evalseed
cd evalseed
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you push

```bash
ruff check src tests
mypy src
pytest -q
```

CI runs the same three commands on Python 3.10, 3.11, and 3.12.

## Tests

- Tests use a `StubJudge` (see [tests/conftest.py](tests/conftest.py)) — never call a real LLM in unit tests.
- New filters need at least three tests: pass case, reject case, and a "judge errored" case.
- Pipeline-level tests use the `_judge_routes` pattern in [tests/test_pipeline.py](tests/test_pipeline.py).

## Code style

- Type hints on every public function. `mypy --strict` must pass.
- No comments that restate the code. Comment only non-obvious *why*.
- New filters subclass `Filter` (LLM-backed) or `PreFilter` (cheap). Keep prompts inline as module-level constants.

## Filing issues

- **Bug reports** should include: corpus snippet, generation config, expected vs actual output. Don't paste full corpora — a 500-char excerpt is enough.
- **Feature requests** should explain the failure mode in current synthetic eval pipelines that motivates the request. evalseed exists to fix specific failure modes, not to be feature-complete.

## Releases

Releases are cut by tagging `v<X.Y.Z>` on `main`. PyPI publish happens automatically via Trusted Publishing on tag push. Maintainers only.
