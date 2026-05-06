# Phase 0: Validation Spike

This directory will hold artifacts from the Phase 0 validation spike that gates the rest of the project.

**Status: not started.**

## Goal

Prove (or disprove) the thesis that RAGAS-default output on real domain corpora has a non-trivial bad-pair rate, before writing any library that depends on that thesis.

## Procedure

1. **Pick a corpus** — ~50 pages of text the maintainer can judge accurately. Default candidate: IRDAI public regulations or a batch of RBI circulars.
2. **Generate baseline** — install RAGAS, generate 100 QA pairs with default settings, save as `baseline.jsonl`.
3. **Human label** — review all 100 pairs, tag each as one of: `good`, `unfaithful`, `ambiguous`, `trivial`, `wrong-difficulty`. Save as `labeled.jsonl`.
4. **Compute bad-pair rate** — anything not `good` counts as bad.

## Gate

| bad-pair rate | decision |
|---------------|----------|
| < 15%         | kill the project; thesis is wrong |
| 15–25%        | marginal — proceed only with strong filter design |
| ≥ 25%         | strong launch story; proceed |

## Artifacts (TODO)

- [ ] `corpus/` — source text used for the spike
- [ ] `baseline.jsonl` — raw RAGAS output
- [ ] `labeled.jsonl` — human-labeled gold set
- [ ] `results.md` — bad-pair breakdown by failure mode + decision
