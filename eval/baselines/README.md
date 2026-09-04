# Frozen evaluation baselines

Committed reference evidence. Unlike `eval/results/`, which is regenerable
scratch output and is gitignored, every retrieval number quoted in `README.md`
and `docs/evaluation.md` traces back to a file here.

Each report embeds the full configuration that produced it — embedding model,
chunk parameters, `k`, retrieval mode, rerank settings. A later run is only
comparable if that block matches.

## `phase2/` — current

33 documents, 149 chunks, `eval/datasets/main.jsonl` (112 queries: 100
answerable, 12 unanswerable), `BAAI/bge-small-en-v1.5`, chunking 320/64.

| file | Recall@1 | nDCG@8 | outcome |
|---|---|---|---|
| `dense-k1` / `dense-k8` | 0.780 | 0.895 | baseline |
| `lexical-k1` / `lexical-k8` | 0.620 | 0.805 | rejected — significantly worse |
| `hybrid-k1` / `hybrid-k8` | 0.720 | 0.881 | rejected — no measured difference |
| `dense-rerank-k1` / `dense-rerank-k8` | **0.850** | **0.939** | **adopted** |
| `hybrid-rerank-k1` / `hybrid-rerank-k8` | 0.850 | 0.935 | rejected — hybrid adds nothing once reranking runs |

The rejected configurations are kept deliberately. A claim that hybrid retrieval
did not help is only checkable if the run behind it still exists, and the same
files are what a future change to fusion or reranking must be compared against.

**Compare at k=1 and k=8.** k=1 is where this corpus discriminates; k=8 is the
operating depth, since the model is given 8 chunks. Intermediate depths are in
`eval/results/` after a run and are not frozen.

## `phase1/` — historical

The original 5-document, 17-chunk corpus and the 19-query smoke set. Retained
because it is the baseline Phase 2 was originally specified against, and because
it documents *why* the corpus had to grow: Recall@k pinned at 1.000 from k=2
with zero variance, so no retrieval change could have been measured on it.

Not comparable to `phase2/` — different corpus and different dataset.

## Regenerating

```bash
python scripts/compare_retrieval.py eval/datasets/main.jsonl
```

Writes k=1 and k=8 reports for every configuration into `eval/results/`. Replace
files here only when a baseline legitimately changes — a different embedding
model, chunking strategy, or corpus — and say so in `Decision.md`.
