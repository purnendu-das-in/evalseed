# evalseed

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/purnendu-das/evalseed/actions/workflows/test.yml/badge.svg)](https://github.com/purnendu-das/evalseed/actions/workflows/test.yml)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

**Generate RAG evaluation datasets you'd actually trust.**

evalseed produces synthetic question-answer pairs from your documents and **filters out the bad ones** — the unfaithful, ambiguous, or trivial pairs that quietly poison most RAG benchmarks.

> New to RAG? A **RAG app** is a chatbot or assistant that answers questions by looking things up in your documents. To know if it works, you need a list of test questions with correct answers — that's an "eval dataset", and that's what evalseed builds for you. See [What problem does this solve?](#what-problem-does-this-solve) for the longer version.

```python
from evalseed import Pipeline, OpenAIJudge

pipeline = Pipeline(
    judge=OpenAIJudge(model="gpt-4o-mini"),
    n_pairs=100,
    types=["single_hop", "multi_hop", "distractor"],
)
dataset = pipeline.generate_from_corpus("./docs/")
dataset.save("eval.jsonl")
```

---

## Table of contents

- [What problem does this solve?](#what-problem-does-this-solve)
- [Core concepts (read this first)](#core-concepts-read-this-first)
- [Install](#install)
- [Your first run, step by step](#your-first-run-step-by-step)
- [Question types explained](#question-types-explained)
- [How filtering works](#how-filtering-works)
- [The CLI](#the-cli)
- [The Python library](#the-python-library)
- [Inspecting and auditing results](#inspecting-and-auditing-results)
- [Bring your own pieces](#bring-your-own-pieces)
- [Project layout](#project-layout)
- [Status](#status)
- [Contributing](#contributing)
- [License](#license)

---

## What problem does this solve?

If you're building a **RAG app** (retrieval-augmented generation — a chatbot that answers questions from your documents), you eventually need to answer: *is it actually working?*

To measure that, you need an **evaluation dataset**: a list of questions, the correct answers, and the document chunks the answers came from. Then you run your RAG app against the questions and check whether it returned the right answer.

You have three ways to build that dataset, and each one is bad in its own way:

1. **Hand-write 200 Q&A pairs.** High quality, but takes days. Nobody actually does this.
2. **Auto-generate them with an LLM.** Fast, but ~30–50% of generated pairs are garbage:
   - The "answer" isn't actually supported by the cited context (**unfaithful**).
   - The question is ambiguous or refers to "the above text" (**not self-contained**).
   - The question is just a copy of a sentence from the source (**trivial**).
   - Your benchmark looks great until you realize you're measuring noise.
3. **Auto-generate, then filter.** Fast *and* trustworthy. ← this is evalseed.

evalseed runs each generated pair through cheap pattern-based pre-filters, then four LLM-judge filters, and rejects anything that fails. Every rejection comes with a structured `reason` so you can audit it.

---

## Core concepts (read this first)

If you're new to RAG evaluation, four terms will appear constantly in the docs and the code. Learn them once here.

### Corpus
The pile of documents you want to evaluate against. Today evalseed reads `.txt` and `.md` files from a folder. PDFs / HTML / DOCX are deliberately out of scope for v0.1.

### Chunk
A corpus document is split into smaller, paragraph-aware **chunks** (default ~1500 characters with 150-char overlap) before generation. The LLM generates Q&A pairs from one chunk at a time, so each pair has a clear, narrow source of truth — which is what makes "is this answer supported by the context?" a question you can actually check.

### QA pair
The unit of an evaluation dataset. A `QAPair` is a `question`, an `answer`, the `context` (the chunk it was generated from), a `qa_type` (single-hop / multi-hop / distractor), an optional `difficulty`, and a list of `filter_results` (one per filter that ran). See [src/evalseed/schemas.py:32](src/evalseed/schemas.py#L32).

### Judge
An LLM that is asked structured yes/no questions about a pair: "is this answer supported by this context?", "is this question self-contained?", etc. evalseed's filters call the judge; you call the filters. Today the only built-in judge is `OpenAIJudge`. The `Judge` interface is one method (`judge(system, user) -> dict`), so swapping in another provider later is intentionally easy.

---

## Install

### Prerequisites

- **Python 3.10 or newer.** Check with `python --version`. If it's older, grab the latest from [python.org](https://www.python.org/downloads/).
- **An OpenAI API key.** Sign up at [platform.openai.com](https://platform.openai.com/), add a few dollars of credit, and create a key. A run on `gpt-4o-mini` typically costs cents, not dollars.
- **Git.** To clone the repo. (Skip this once we're on PyPI.)

### Steps

```bash
git clone https://github.com/purnendu-das/evalseed
cd evalseed
pip install -e .
```

> **Tip:** install into a virtual environment so it doesn't clash with your other Python projects: `python -m venv .venv && source .venv/bin/activate` (macOS/Linux) or `python -m venv .venv; .venv\Scripts\Activate.ps1` (Windows PowerShell).

Then set your OpenAI key:

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

Verify the install:

```bash
evalseed --help
```

PyPI release is gated on the Phase 0 validation spike — see [Status](#status).

---

## Your first run, step by step

Total time: ~3 minutes. Cost on `gpt-4o-mini`: a few cents.

**1. Make a tiny corpus.** Create a folder with one or two `.txt` or `.md` files in it. A few paragraphs each is plenty for a smoke test.

```
sample_corpus/
├── about.md
└── faq.txt
```

**2. Run the CLI.** Start small so you don't burn tokens before you know the wiring is right.

```bash
evalseed ./sample_corpus/ -o eval.jsonl -n 10 --types single_hop,multi_hop
```

You should see live progress: how many docs and chunks were loaded, how many pairs were generated, and a summary table of how many were rejected by each filter.

**3. Look at what came out.**

- `eval.jsonl` — one line per **passed** Q&A pair. Each line is a self-contained JSON object. (JSONL = "JSON Lines", a streaming-friendly text format that most eval tools — RAGAS, DeepEval, LangChain — read directly.)
- If you also pass `--all`, rejected pairs are saved with the failing filter's `reason`. Read these to judge whether the filters are doing the right thing on *your* documents.

**4. Now scale up.** Once a small run looks sane, bump `-n` to 100 or 200 for a real eval set.

---

## Question types explained

The `--types` flag (or `types=...` in Python) controls **what kind of reasoning each generated question requires**. The "hop" is one retrieval / reasoning step. This is the single most important knob.

### `single_hop`
Answerable from **one chunk, one fact**. No combining, no chaining.

> *Context:* "France is a country in Western Europe. Its capital is Paris."
> *Q:* "What is the capital of France?"
> *A:* "Paris."

This tests whether your retriever finds the right chunk and your model reads it correctly. It's the easy baseline — most RAG systems pass single-hop and fail the rest.

### `multi_hop`
Requires **combining facts from two or more chunks** to answer. The retriever has to find all of them; the model has to join them.

> *Chunk A:* "Marie Curie was born in Warsaw in 1867."
> *Chunk B:* "Warsaw is the capital of Poland."
> *Q:* "In which country was Marie Curie born?"
> *A:* "Poland."

This tests retrieval recall (do you fetch *both* chunks?) and reasoning (does the model actually join A+B instead of guessing?). **This is where most RAG systems quietly fail** — and where having a good eval set actually matters.

### `distractor`
Bundles **relevant chunks together with irrelevant-but-similar-looking ones**. The model has to ignore the distractors.

> Retrieve a chunk about *Paris, France* AND a chunk about *Paris, Texas*.
> *Q:* "What is the population of the capital of France?"
> The model must not be fooled by the Texas chunk.

This tests robustness — whether a noisy retriever (the realistic case) breaks the answer. Off by default because generating good distractors is more expensive; opt in with `--types single_hop,multi_hop,distractor`.

Defined in code at [src/evalseed/schemas.py:10-13](src/evalseed/schemas.py#L10-L13).

---

## How filtering works

Every generated pair runs through this gauntlet **in order**. The first failing filter short-circuits the rest for that pair (because the pair is already dead, and judge calls cost money).

| # | Stage | What it catches | Why it matters |
|---|---|---|---|
| 1 | `LengthPreFilter` | Pairs with implausibly short or long Q or A | Cheap regex/len check before any LLM cost |
| 2 | `RegexPreFilter` | Meta-questions ("what does the above text say…"), refusals, yes/no fragments | Same — cheap, kills obvious junk |
| 3 | `FaithfulnessFilter` | Answer **not entailed** by the cited context | Catches hallucinated answers — the #1 garbage source |
| 4 | `AnswerabilityFilter` | Ambiguous, externally-dependent, or non-self-contained questions | A question like "what year was it released?" is unusable without context |
| 5 | `TrivialityFilter` | Verbatim restatements of a source sentence (n-gram check + judge) | If Q is just "the source sentence with a question mark", you're testing string match, not RAG |
| 6 | `DifficultyFilter` | Labeled difficulty disagrees with the judge's assessment | Keeps the easy/medium/hard split honest |

Stages 1–2 are **PreFilters** (regex/length, no LLM call). Stages 3–6 are **Filters** (each makes one LLM call to the judge).

Every filter result lands on the pair as a `FilterResult` with a structured `reason`, so you can audit a rejection rather than trust the count blindly.

---

## The CLI

```
evalseed CORPUS [-o OUT] [-n N_PAIRS] [--types T1,T2,...] [--model MODEL] [--seed SEED] [--all]
```

| Flag | Default | Purpose |
|---|---|---|
| `CORPUS` (positional) | required | File or directory of `.txt`/`.md` to generate from |
| `-o, --out` | `eval.jsonl` | Where to write the dataset |
| `-n, --n-pairs` | `50` | Target number of pairs to generate |
| `--types` | `single_hop,multi_hop` | Comma-separated QA types: `single_hop`, `multi_hop`, `distractor` |
| `--model` | `gpt-4o-mini` | OpenAI model id used by both generator and judge |
| `--seed` | none | Set for deterministic generation (useful for tests / reproducible runs) |
| `--all` | off | Save rejected pairs too, with their `reason` field — use this to audit |

Example — generate 200 pairs of all three types from a docs folder, with a fixed seed, and keep rejected pairs for inspection:

```bash
evalseed ./docs/ \
  -o eval.jsonl \
  -n 200 \
  --types single_hop,multi_hop,distractor \
  --seed 42 \
  --all
```

CLI source: [src/evalseed/cli.py](src/evalseed/cli.py).

---

## The Python library

The CLI is a thin wrapper. Driving evalseed from Python gives you more control: stats, custom filter sets, and the ability to filter pre-generated pairs.

### Minimal example

```python
from evalseed import Pipeline, OpenAIJudge

pipeline = Pipeline(
    judge=OpenAIJudge(model="gpt-4o-mini"),
    n_pairs=50,
    types=["single_hop", "multi_hop"],
    seed=42,
)
dataset = pipeline.generate_from_corpus("./docs/")

print(dataset.stats())
#   {'total': 60, 'passed': 41, 'rejected': 19, 'pass_rate': 0.68,
#    'rejections_by_filter': {'faithfulness': 8, 'answerability': 6, ...}}

dataset.save("eval.jsonl")                                     # passed pairs only
dataset.rejected.save("rejected.jsonl", only_passed=False)     # for inspection
```

### What `Pipeline` accepts

The constructor (full signature at [src/evalseed/pipeline.py:37](src/evalseed/pipeline.py#L37)):

| Arg | Default | What it does |
|---|---|---|
| `judge` | required | The `Judge` instance the filters will call |
| `n_pairs` | `50` | Target pair count |
| `types` | `(SINGLE_HOP, MULTI_HOP)` | Which QA types to generate |
| `pairs_per_chunk` | `2` | How many pairs the generator tries to make per chunk |
| `chunk_chars` | `1500` | Target chunk size when splitting documents |
| `chunk_overlap` | `150` | Char overlap between adjacent chunks (preserves context across boundaries) |
| `prefilters` | `[LengthPreFilter, RegexPreFilter]` | Override the cheap pre-filter list |
| `filters` | the four LLM filters | Override the LLM filter list (see [Bring your own pieces](#bring-your-own-pieces)) |
| `generator` | a default `QAGenerator` | Plug in a custom generator |
| `seed` | `None` | Deterministic generation |
| `verbose` | `True` | Pretty-print progress and a stats table at the end |

### Three ways to feed it data

```python
# 1. From a folder/file of .txt/.md
dataset = pipeline.generate_from_corpus("./docs/")

# 2. From pre-chunked text (you already have a chunker you like)
from evalseed.corpus import Chunk
chunks = [Chunk(text="...", source="my_doc.md"), ...]
dataset = pipeline.generate_from_chunks(chunks)

# 3. Skip generation entirely — just filter pairs you already have
from evalseed.generator import parse_pairs_jsonl
pairs = parse_pairs_jsonl("ragas_output.jsonl")
dataset = pipeline.filter_pairs(pairs)
```

Option 3 is useful if you already generated pairs with another tool (RAGAS, DeepEval, hand-written) and just want evalseed's filters as a quality gate.

---

## Inspecting and auditing results

The whole point of evalseed is that you can *trust* what comes out. That requires being able to look at it.

### Stats

```python
dataset.stats()
# {
#   'total': 60,
#   'passed': 41,
#   'rejected': 19,
#   'pass_rate': 0.68,
#   'rejections_by_filter': {
#     'faithfulness': 8,
#     'answerability': 6,
#     'triviality': 3,
#     'difficulty': 2,
#   }
# }
```

If `faithfulness` is rejecting half your pairs, your generator is hallucinating — try a stronger model or smaller chunks. If `triviality` is high, your corpus may be very fact-dense (encyclopedia-style) — that's a real signal.

### Read the rejections

```python
for pair in dataset.rejected:
    print(pair.question)
    for r in pair.filter_results:
        if not r.passed:
            print(f"  rejected by {r.filter_name}: {r.reason}")
```

Or just save them and grep:

```python
dataset.rejected.save("rejected.jsonl", only_passed=False)
```

Spend 10 minutes reading 20 rejected pairs from your first real run. You'll quickly see whether the filters are too strict, too lenient, or correct on your domain.

### One QA pair, on disk

```jsonc
{
  "id": "9f2c…",
  "question": "What does Section 12 of the IT Act 2000 require providers to do?",
  "answer": "Acknowledge electronic records they receive…",
  "context": "Section 12. Acknowledgement of receipt. — (1) Where the …",
  "qa_type": "single_hop",
  "difficulty": "medium",
  "source": "it_act_2000.md",
  "filter_results": [
    {"filter_name": "length_prefilter", "passed": true, "score": null, "reason": null},
    {"filter_name": "regex_prefilter",  "passed": true, "score": null, "reason": null},
    {"filter_name": "faithfulness", "passed": true, "score": 0.95, "reason": null}
    // ...
  ]
}
```

---

## Bring your own pieces

evalseed is a thin orchestration layer. You can swap any of these.

### Run only some filters

```python
from evalseed import Pipeline, OpenAIJudge
from evalseed.filters import FaithfulnessFilter, AnswerabilityFilter

judge = OpenAIJudge(model="gpt-4o-mini")
pipeline = Pipeline(
    judge=judge,
    n_pairs=50,
    filters=[                                # only run two LLM stages
        FaithfulnessFilter(judge, threshold=0.8),
        AnswerabilityFilter(judge),
    ],
)
```

### Tighten or loosen a threshold

Most filters take a `threshold` argument. `FaithfulnessFilter(judge, threshold=0.9)` will reject more aggressively; `0.6` will be more permissive.

### Use evalseed only as a filter pass

If another tool (RAGAS, DeepEval, hand-written) generated the pairs, you can skip generation entirely — see option 3 in [Three ways to feed it data](#three-ways-to-feed-it-data).

### Plug in a different LLM provider

Implement the one-method `Judge` protocol ([src/evalseed/judges.py](src/evalseed/judges.py)) and pass your instance to `Pipeline(judge=...)`. A first-party multi-provider judge (Anthropic / Gemini / Bedrock / local) is deferred to v0.2 — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Project layout

```
src/evalseed/
├── __init__.py        # public API: Pipeline, OpenAIJudge, Dataset, QAPair
├── pipeline.py        # orchestrates load → chunk → generate → filter
├── generator.py       # QAGenerator + JSONL loader for pre-generated pairs
├── corpus.py          # file loading + paragraph-aware chunking
├── dataset.py         # Dataset (iterable, sliceable, .stats(), .save())
├── judges.py          # Judge protocol + OpenAIJudge implementation
├── schemas.py         # pydantic models: QAPair, FilterResult, QAType, Difficulty
├── exceptions.py
├── cli.py             # `evalseed` console script
└── filters/
    ├── prefilters.py  # LengthPreFilter, RegexPreFilter (no LLM call)
    ├── faithfulness.py
    ├── answerability.py
    ├── triviality.py
    └── difficulty.py
```

---

## Status

Pre-release alpha. The Phase 0 validation spike (see [docs/phase0/](docs/phase0/)) has not been run yet — no `v0.1.0` PyPI release until the thesis is validated against real data.

<!-- TODO: fill in real numbers from Phase 0 spike before publishing -->
<!-- In benchmarks on <corpus type> documents, evalseed rejected X% of generated pairs that human reviewers also flagged as unusable. -->

### Comparison

|                          | evalseed | RAGAS | DeepEval Synthesizer |
|--------------------------|----------|-------|----------------------|
| Multi-stage filtering    | Yes — 6 ordered stages (2 pre-filters + 4 LLM judges), each rejection carries a structured `reason` | No explicit post-gen filter pipeline — quality is folded into the generation step | Partial — single LLM quality check / threshold |
| Type labeling            | `single_hop` / `multi_hop` / `distractor`, plus a difficulty label | Evolution types (`simple`, `reasoning`, `multi_context`, `conditional`) | Evolution types (reasoning, multi-context, etc.) |
| Pluggable judge          | One-method `Judge` protocol — drop-in for any provider | Via LangChain LLM wrapper | Via `DeepEvalBaseLLM` subclass |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The scope of v0.1 is deliberately tight — please open an issue before working on multi-provider judges, async, PDF loaders, or framework integrations.

## License

MIT — see [LICENSE](LICENSE).
