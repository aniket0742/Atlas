# Step 8: plain RAG vs agent mode

The full Phase 4 evaluation. Produced by `scripts/evaluate_agent.py`, scoring
logic in `src/atlas/eval/agent_compare.py`. Raw per-query results in
`agent-vs-plain.json`.

## Configuration (the shipped default, not a search)

| | |
|---|---|
| answer model | `gemini-3.5-flash-lite` |
| agent model | `gemini-3.1-flash-lite` |
| retrieval | dense + rerank (`ATLAS_RERANK_ENABLED=true`) |
| agent evidence ordering | union rerank against the original question (ADR-0031) |
| agent bounds | 4 iterations, 8 tool calls, 60s budget, 12-passage evidence cap |
| dataset | `eval/datasets/main.jsonl`, 112 questions (100 answerable, 12 unanswerable) |

## Method

Every question runs through both systems, in the same process, against the
same corpus and the same retriever/reranker instances — so any difference is
the system, not the environment. Quality is scored by **citation recall**:
of a question's gold labels, how many are satisfied by a chunk the answer
*cited* (not merely retrieved) — see `agent_compare.citation_recall`. That is
a stricter bar than "produced a citation," which is all the Phase 2 harness
checks.

Paired bootstrap (`metrics.paired_bootstrap_delta`, the same function ADR-0021
established) compares the two systems on identical questions rather than
reading two independent confidence intervals side by side.

## Headline result

Of 100 answerable questions, **96 scored identically** on both systems. Of the
4 that differed: **2 wins for the agent, 2 losses.** Exactly even.

| | plain | agent |
|---|---|---|
| citation recall (mean, 100 Qs) | 0.975 [0.94, 1.0] | 0.970 [0.93, 1.0] |
| paired delta (agent − plain) | — | **−0.005**, CI [−0.04, 0.03] — crosses zero |
| unanswerable correctly refused | 12/12 | 12/12 |
| answerable wrongly refused | 0 | 0 |
| unverified quotes | 6 | 6 |
| errors / degraded runs | 0 | 0 / 0 |
| cost per 1000 questions | $0.79 | $1.40 (**1.8×**) |
| latency p50 / p95 | 3286 / 4286 ms | 5132 / 7050 ms (**+56% / +64%**) |
| mean tool calls | — | 1.15 (2.0 on multi-doc, ~1.0 elsewhere) |

**No measurable overall quality difference.** The paired delta's confidence
interval straddles zero, and every per-kind breakdown does too (multi-doc
[0.0, 0.3], identifier [0.0, 0.23], paraphrase [−0.16, 0.0] — each just
touching zero at one edge). The agent costs 1.8× and takes ~60% longer to
produce answers that are, on this evidence, indistinguishable from plain RAG.

## The four discordant questions, by name

**Agent won:**
- `refund-window-by-plan` (multi-doc) — plain 0.5, agent 1.0. Plain cited only
  one of the two required documents; the agent's second search reached both.
  This is the case the multi-doc hypothesis (Recall@1 = 0.400, Phase 2) was
  about.
- `hybrid-flag-name` (identifier) — plain 0.0, agent 1.0. A clean win, not
  investigated further; one query is not worth building a theory on.

**Agent lost**, both `paraphrase`, both investigated:
- `revocation-delay` — the union rerank correctly ranked the right document
  (`engineering/authentication.md`) **first**. The answer model then cited a
  *different* document (`product/release-notes-2026-q1.md`) whose passage
  ("access tokens remain valid until expiry by design") answers the question
  as directly as the labelled one. This is not a retrieval failure — it is the
  answer model choosing an equally-plausible source over the labelled one, and
  arguably not wrong so much as unlucky against a single-answer label.
- `queue-first-checks` — the right document (`runbooks/runbook-queue-backlog.md`)
  filled 3 of 7 evidence slots, but none of the three retrieved chunks happened
  to contain the labelled line ("Are workers alive?") — a different section of
  the same document was needed and was not among what the searches surfaced.
  This is a genuine retrieval-depth miss, on a document the agent searched for
  correctly.

Two losses out of 100 questions is not a pattern; it is two examples. Recorded
in that spirit — as material for anyone deciding whether to invest further,
not as a proven failure mode.

## What is and is not shown

**Shown:** the union rerank (ADR-0031) works — 0 errors, 0 degraded runs, 0
bound hits across 100 live agent runs, and answer quality that does not
regress against plain RAG in aggregate. The multi-document case the agent was
built to help shows one real win in five instances, consistent with (not
proof of) the original hypothesis.

**Not shown:** that agent mode is worth its cost. 96% of questions get an
identical answer either way, at a system that costs 1.8× and answers
noticeably slower. A well-engineered feature is not the same claim as a
valuable one, and this evaluation answers the first, not the second.

**Limits.** One corpus, one labeller, 112 questions, one run (not repeated —
ADR-0024 and ADR-0031 both observed run-to-run variance on smaller samples;
this run is larger but still a single draw). Tool-selection accuracy and
unnecessary-call rate are not separately benchmarked here because this
dataset has no question where the correct action is *not* to search — that
axis is already covered by the 16-case routing benchmark in
`eval/baselines/agent-routing/`, which measured 16/16 accuracy and 0
unnecessary searches for `gemini-3.1-flash-lite` on cases that do include
that choice.
