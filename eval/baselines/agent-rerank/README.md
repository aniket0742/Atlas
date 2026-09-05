# Agent evidence-ordering diagnostic

Does reranking the union of the agent's searches, once, against the original
question change the answer? Produced by `scripts/agent_rerank_diagnostic.py`.

Not the Phase 4 evaluation. This is a targeted experiment on one hypothesis,
run because the first live agent comparison (ADR-0030) went *against* the
agent and the leading explanation was evidence ordering.

## Design

Three arms on the same questions with the same answer model:

| arm | evidence |
|---|---|
| `plain` | one retrieval, `top_k` passages, globally reranked. The existing baseline. |
| `A-interleave` | agent, evidence interleaved by rank across searches (behaviour before the fix) |
| `B-union-rerank` | agent, deduplicated union reranked once against the original question |

**A and B share one gathered plan per question.** The agent searches once, and
both arms are built from that same union. Running two independent agent runs
would confound the ordering change with the model happening to choose different
searches; holding the searches fixed isolates the ordering.

Questions are the five `multi-doc` cases from `eval/datasets/main.jsonl`, which
carry ground-truth documents, plus the two questions from the ADR-0030
observation. Seven in total.

Metric is **document coverage**: of the documents the question is labelled as
needing, how many appear in the answer's citations. It measures whether the
answer actually used the material, not merely whether retrieval found it.

## Results, two runs

| question | plain | A | B |
|---|---|---|---|
| downgrade-and-closure-retention | 2/2, 2/2 | 2/2, 2/2 | 2/2, 2/2 |
| revocation-and-release | 2/2, 2/2 | **1/2**, 2/2 | 2/2, 2/2 |
| refund-window-by-plan | 2/2, 2/2 | 2/2, 2/2 | 2/2, 2/2 |
| retention-privacy-vs-eng | 2/2, 2/2 | 2/2, 2/2 | 2/2, 2/2 |
| ratelimit-error-and-header | 2/2, 2/2 | 2/2, 2/2 | 2/2, 2/2 |
| step6-refunds | 2/2, 2/2 | **1/2**, **1/2** | 2/2, 2/2 |
| step6-apikey | 1/1, 1/1 | 1/1 refused | 1/1 refused |

Totals across both runs: **plain 26/26, A 23/26, B 26/26.**

Prompt tokens were **identical** between A and B on every question — same
union, same cap, only the order differs. Both arms produced the same evidence
order on both runs, so the ordering itself is deterministic.

Union rerank latency: 28–556 ms over unions of 5–10 passages.

## What this does and does not show

**Reproduced:** `step6-refunds` — A cited one document, B cited both, twice.
This is the case ADR-0030 was written about.

**Not reproduced:** `revocation-and-release` — A scored 1/2 then 2/2 on
identical evidence in identical order. That difference was answer-model
nondeterminism, not ordering. It is left in the table rather than dropped,
because discarding the inconvenient half of a small sample is how a
seven-question experiment gets talked into meaning more than it does.

So: B never lost to A, matched plain everywhere, and beat A on one
reproducible case. That is consistent with the ordering hypothesis and is not
a measurement of it. Seven questions, a metric with demonstrated run-to-run
variance, and no confidence intervals.

**The agent still does not beat plain RAG here.** It matches it, at the cost of
an extra model and several searches. Whether the agent earns that is the Phase 4
step 8 question, not this one.

`step6-apikey` refuses in both agent arms and answers under plain. Both arms
retrieved the right document; the corpus does not appear to cover the "what if
it leaks" half, and plain answered only the rotation half without flagging the
gap. Arguably the agent is behaving *better* there. Not scored either way.
