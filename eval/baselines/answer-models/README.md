# Answer-model comparison (frozen)

Which model writes the final, cited answer. Retrieval was held constant — the
same `Retriever` instance, dense + cross-encoder rerank — so every difference
between rows is attributable to the answer model alone.

Corroborated by the token counts: **input tokens are identical (141,584) across
all four models**, which is the check that retrieval really was constant rather
than assumed to be.

Dataset: `eval/datasets/main.jsonl`, 112 queries (100 answerable, 12
unanswerable). One report per model in this directory.

## Results

| model | refused OK | wrongly refused | cited | unverified quotes | p50 latency | in tok | out tok | $/1k queries |
|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 12/12 | 0 | 100/100 | 14 | 2,254 ms | 141,584 | 16,389 | $0.54 |
| **gemini-3.5-flash-lite** | 12/12 | 0 | 100/100 | 3 | 2,507 ms | 141,584 | 15,038 | **$0.71** |
| gemini-3.7-flash | 12/12 | 0 | 100/100 | 2 | 3,016 ms | 141,584 | 44,553 | $2.44 |
| gemini-3.8-flash | 12/12 | 0 | 100/100 | 1 | 3,968 ms | 141,584 | 56,347 | $2.83 |

`p50` is measured on a **serial** 6-query sample, not from the concurrent run, so
it reflects one user rather than eight competing for CPU.

## The groundedness guarantees hold

Every model: 12/12 unanswerable queries refused, 0 answerable queries wrongly
refused, 100/100 answers carrying a resolvable citation.

This is the first end-to-end measurement of the refusal and citation-validation
machinery built in Phase 1. Before this it was three phases of untested claims.

## Are the differences real?

The only metric that separates the models is unverified quotes — citations whose
`quote` was not found verbatim in the chunk they cite.

**A single run cannot rank them.** `gemini-3.5-flash-lite` produced **5, then 8,
then 3** unverified quotes across three runs of identical configuration. That
range is wider than most of the gaps between models, so reading a one-run table
as a ranking would be reading noise.

Per-query counts compared with the paired bootstrap from
[ADR-0021](../../../Decision.md), against the selected default:

| model | delta | paired 95% CI | verdict |
|---|---|---|---|
| gemini-3.1-flash-lite | +0.098 | [+0.045, +0.161] | **significantly worse** |
| gemini-3.7-flash | −0.009 | [−0.027, +0.000] | no measured difference |
| gemini-3.8-flash | −0.018 | [−0.045, +0.000] | no measured difference |

So: `3.1-flash-lite` is genuinely worse at verbatim quoting. `3.7-flash` and
`3.8-flash` are **not measurably better** than `3.5-flash-lite`, while costing
3.4–4× more and running 20–58% slower.

## Selected: `gemini-3.5-flash-lite`

Chosen because the alternatives' advantage is not measurable, not because it is
cheaper. See [ADR-0024](../../../Decision.md).

## Why the cost column can be trusted

An earlier version of this table priced cost from a total token count split
85/15 input:output. That was wrong by 15–35%, and wrong in a direction that
mattered: it understated `gemini-3.8-flash` by 35%.

The cause is visible in the table. `3.7` and `3.8` emit **3–4× more output
tokens** than the lite models on identical input, because they produce thinking
tokens — which are billed as output and are invisible in the response text.
Cost is now computed from measured input and output totals, priced separately,
with thinking folded into output.

## Regenerating

```bash
python scripts/compare_answer_model.py      # runs the eval per model
python scripts/freeze_answer_baselines.py   # freezes reports here + paired test
```

Both cost real money now that billing is active — roughly $0.65 for all four
models over 112 queries.
