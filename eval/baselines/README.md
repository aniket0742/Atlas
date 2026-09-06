# Frozen evaluation baselines

Committed reference evidence. Unlike `eval/results/`, which is regenerable
scratch output and is gitignored, every number quoted in `README.md` and
`docs/evaluation.md` traces back to a file here.

Each report embeds the full configuration that produced it — embedding model,
chunk parameters, `k`, retrieval mode, rerank settings. A later run is only
comparable if that block matches.

**The rejected configurations are kept deliberately.** A claim that hybrid
retrieval did not help, or that agent mode did not beat plain RAG, is only
checkable if the run behind it still exists.

## What is here

| directory | question it answers | outcome |
|---|---|---|
| [`phase1/`](phase1) | What is the dense baseline? | historical — corpus too small to discriminate |
| [`phase2/`](phase2) | Dense vs lexical vs hybrid, and does reranking help? | **dense + rerank adopted**; lexical and hybrid rejected |
| [`answer-models/`](answer-models) | Which model should write the cited answer? | **`gemini-3.5-flash-lite`** |
| [`agent-routing/`](agent-routing) | Which model should decide when to call a tool? | **`gemini-3.1-flash-lite`** |
| [`agent-rerank/`](agent-rerank) | Does reranking the agent's evidence union fix its ordering? | adopted — never worse, one reproducible win |
| [`step8-agent-vs-plain/`](step8-agent-vs-plain) | Is agent mode better than plain RAG? | **no measured difference** — agent stays opt-in |

Each directory except `phase1/` and `phase2/` carries its own README with the
method, the numbers and the limits of what they support.

## `phase2/` — the shipped retrieval configuration

33 documents, 149 chunks, `eval/datasets/main.jsonl` (112 queries: 100
answerable, 12 unanswerable), `BAAI/bge-small-en-v1.5`, chunking 320/64.
Retrieval-only, so these runs cost nothing to regenerate.

| file | Recall@1 | nDCG@8 | outcome |
|---|---|---|---|
| `dense-k1` / `dense-k8` | 0.780 | 0.895 | baseline |
| `lexical-k1` / `lexical-k8` | 0.620 | 0.805 | rejected — significantly worse |
| `hybrid-k1` / `hybrid-k8` | 0.720 | 0.881 | rejected — no measured difference |
| `dense-rerank-k1` / `dense-rerank-k8` | **0.850** | **0.939** | **adopted** |
| `hybrid-rerank-k1` / `hybrid-rerank-k8` | 0.850 | 0.935 | rejected — hybrid adds nothing once reranking runs |

**Compare at k=1 and k=8.** k=1 is where this corpus discriminates; k=8 is the
operating depth, since the model is given 8 chunks. Intermediate depths land in
`eval/results/` after a run and are not frozen.

## `phase1/` — historical

The original 5-document, 17-chunk corpus and the 19-query smoke set. Retained
because it is the baseline Phase 2 was originally specified against, and because
it documents *why* the corpus had to grow: Recall@k pinned at 1.000 from k=2
with zero variance, so no retrieval change could have been measured on it.

Not comparable to `phase2/` — different corpus, different dataset.

## Regenerating

Retrieval baselines are free — no model calls:

```bash
python scripts/compare_retrieval.py eval/datasets/main.jsonl
```

The rest call real models and **cost real money**, which is why they are frozen
here rather than regenerated on every change and why CI does not run them:

```bash
python scripts/compare_answer_model.py      # ~$0.73 — four answer models, 112 queries
python scripts/validate_agent_model.py      # ~$0.04 — three routing models, 16 cases
python scripts/agent_rerank_diagnostic.py   # ~$0.05 — 7 questions, three arms
python scripts/evaluate_agent.py            # ~$0.30 — plain vs agent, 112 queries paired
```

Replace files here only when a baseline legitimately changes — a different
embedding model, chunking strategy, corpus, or system configuration — and say so
in `Decision.md`.
