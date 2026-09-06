# Evaluation methodology

This describes how retrieval and answering quality are measured, what the numbers
mean, and — as importantly — what they do not mean.

## Why this exists in Phase 1

The original plan put evaluation in Phase 8, after hybrid retrieval and
reranking. That ordering cannot work. Phase 2 exists to answer "does hybrid
retrieval improve results?", and that question is only answerable against a
recorded Phase 1 baseline. By Phase 8 the Phase 1 system no longer exists, and
the comparison would have to be asserted from memory or from other people's blog
posts.

So the harness ships with Phase 1, and its first job is to record the dense-only
baseline that Phase 2 must beat.

## What a label points at

A label names a **document** by its stable external id and, optionally, a
**snippet** that must appear in the retrieved chunk:

```json
{"id": "refund-window",
 "question": "How long do customers have to request a refund?",
 "answerable": true,
 "labels": [{"document": "policies/billing.md", "contains": "within 30 days"}]}
```

It deliberately does **not** name chunk ids. Chunk ids derive from
`(document, version, ordinal)`, so changing the chunk size, the overlap, or the
splitting strategy renumbers every chunk in the corpus. A dataset labelled with
chunk ids would be silently invalidated by the first chunking experiment it was
built to evaluate — every label would point at a chunk that no longer exists, and
the metrics would collapse to zero for reasons having nothing to do with
retrieval quality. Document-plus-snippet labels survive re-chunking,
re-embedding and re-ingestion.

The cost is coarser granularity: any chunk from the right document containing the
snippet satisfies the label. That is an acceptable trade for labels that remain
valid across the experiments this harness exists to run.

A test asserts that every shipped label's snippet still exists in its corpus
file, so editing the corpus without updating the dataset fails CI rather than
quietly degrading the numbers.

## Unanswerable queries

The dataset contains questions the corpus genuinely cannot answer. They carry no
labels and are scored on a separate axis: **did the system refuse?**

This is not a secondary metric. A system that scores well on Recall@5 and then
confidently answers "How many vacation days do employees get?" from a corpus
containing no HR content is broken, and no retrieval metric will reveal it.

The three unanswerable queries are chosen to be progressively harder:

| Query | Why it is hard |
|---|---|
| vacation days | no topically related content at all — the easy case |
| Enterprise per-seat price | `billing.md` is topically adjacent but never mentions prices |
| SAML SSO support | `authentication.md` is *very* close in embedding space and says nothing about SAML |

The third is the one that matters. Dense retrieval will return the
authentication document with a high similarity score, so refusal depends on the
model and the validation layer, not on retrieval failing to find anything.

## Metrics

Definitions are implemented explicitly in `src/atlas/eval/metrics.py` rather than
imported, because each of these has more than one convention in circulation and a
benchmark with implicit definitions is not a benchmark.

| Metric | Definition as implemented |
|---|---|
| **Recall@k** | fraction of the query's *labels* satisfied within the top k. Counted over labels, not chunks — with overlapping chunks, several chunks can legitimately carry the same fact, and counting chunks would inflate it. |
| **Precision@k** | fraction of the top k retrieved chunks that satisfy some label. |
| **MRR** | 1 / (1-based rank of the first relevant chunk), 0 if none. |
| **nDCG@k** | binary gain, `log2(rank+1)` discount. IDCG uses `min(labels, k)`, so retrieving one of two relevant items at rank 1 does *not* score 1.0. |

All relevance is binary. Graded relevance would need multiple annotators to mean
anything, and there is one.

### Precision@k needs a caveat

With overlapping chunks and a single-label query, the theoretical maximum
Precision@8 may be well below 1.0 — there simply are not eight relevant chunks to
find. The absolute value is therefore not interpretable on its own. It is
comparable *between retrieval configurations on the same dataset*, which is the
only way it is used here.

### Confidence intervals

Every aggregate is reported with a 95% percentile bootstrap confidence interval
over queries. This is not decoration. With 16 answerable queries, a difference of
a few points in mean nDCG is well inside sampling noise, and reporting it as an
improvement would be dishonest.

**Reading rule: if two configurations' intervals overlap, the harness has not
measured a difference between them.** The CLI prints this reminder after every
run.

## Answer-level metrics

`--with-answers` additionally reports:

- **Refusal correctness, both directions.** How many unanswerable queries were
  correctly refused, and how many answerable queries were *incorrectly* refused.
  Both matter: a system that refuses everything scores perfectly on the first.
- **Citation coverage.** How many non-refused answers carried at least one
  resolvable citation.
- **Unverified quotes.** Citations that resolved to a real chunk but whose quote
  was not found verbatim in it. Usually paraphrase; sometimes fabrication. Either
  way it is tracked rather than hidden.
- **Latency percentiles and total tokens.** p50/p95/max end-to-end, and token
  usage — the cost side of any quality claim.

This mode costs one model call per query, so it is run deliberately. Retrieval-
only mode uses no LLM quota at all and can be run on every change.

## Retrieval metrics ignore the similarity floor

The runner retrieves with `min_similarity=0.0` when scoring retrieval. The floor
is an *answering policy* — it decides when to refuse — not a property of
retrieval. Folding it into retrieval scoring would make a threshold change look
like a retrieval regression and hide the actual effect.

The floor's effect is measured where it belongs: in the refusal metrics.

## Reproducibility

Every report embeds the configuration that produced it:

```json
"config": {
  "k": 8,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "chunk_target_tokens": 320,
  "chunk_overlap_tokens": 64,
  "min_similarity": 0.3,
  "retrieval": "dense"
}
```

Reports without this are not comparable to anything, and a directory of such
reports is worse than none because it invites false comparisons.

Note that fastembed downloads a **quantised** ONNX build of the embedding model.
Any published number should name that, since quantisation can shift retrieval
quality slightly relative to the original FP32 weights.

## Phase 1 results (historical)

Run on the 19-query smoke set over the original 5-document corpus. Retained as
the historical baseline; superseded for measurement by the Phase 2 results
below, which use a 33-document corpus and 112 queries.

### E1 — dense retrieval baseline

| k | Recall@k | MRR | nDCG@k | P@k |
|---|---|---|---|---|
| 1 | 0.906 [0.78–1.00] | 0.938 [0.81–1.00] | 0.938 [0.81–1.00] | 0.938 |
| 2 | 1.000 [1.00–1.00] | 0.969 [0.91–1.00] | 0.977 [0.93–1.00] | 0.562 |
| 3 | 1.000 [1.00–1.00] | 0.969 [0.91–1.00] | 0.977 [0.93–1.00] | 0.375 |
| 5 | 1.000 [1.00–1.00] | 0.969 [0.91–1.00] | 0.977 [0.93–1.00] | 0.225 |
| 8 | 1.000 [1.00–1.00] | 0.969 [0.91–1.00] | 0.977 [0.93–1.00] | 0.141 |

Brackets are 95% bootstrap confidence intervals over 16 answerable queries.

**Reading this honestly.** Dense retrieval puts a relevant chunk at rank 1 for
about 91% of queries, and finds everything by rank 2. That sounds excellent and
mostly is not informative: the corpus is 5 documents and 17 chunks, so k=8
retrieves nearly half the index. **Recall saturates at k=2 and stays pinned at
1.000 with zero variance**, which means Recall@5 and Recall@8 cannot distinguish
between any two retrieval strategies on this corpus.

Consequence for Phase 2: **hybrid retrieval and reranking must be compared at
k=1, and on a larger corpus.** Reporting "hybrid keeps Recall@8 at 1.0" would be
meaningless — dense already achieves it and so would picking chunks at random
often enough. Precision@k falling as k grows is likewise an artefact of a
single-label dataset over a tiny index, not a quality signal.

The one query whose first relevant chunk is not at rank 1 is what drags MRR to
0.969; that single query is the only headroom this dataset currently offers.

### E2 — similarity floor calibration

`scripts/calibrate_floor.py`. No LLM calls: the floor acts on retrieval scores,
so its effect is pure arithmetic over a single retrieval pass.

| | min | max |
|---|---|---|
| Answerable — best **relevant** chunk | 0.669 | 0.879 |
| Unanswerable — best chunk overall | 0.569 | 0.639 |

The distributions are separable, with a gap from 0.639 to 0.669.

| floor | unanswerable auto-refused (of 3) | answerable wrongly refused (of 16) | answerable losing relevant evidence |
|---|---|---|---|
| 0.30 (old default) | 0 | 0 | 0 |
| 0.58 | 1 | 0 | 0 |
| 0.60 (**chosen**) | 1 | 0 | 0 |
| 0.62 | 2 | 0 | 0 |
| 0.64–0.66 | 3 | 0 | 0 |
| 0.68+ | 3 | 0 | 1 |

**0.60 was chosen over the apparently-optimal 0.64–0.66.** The separating gap is
0.03 wide and rests on three unanswerable queries; putting the threshold inside
it fits a parameter to three data points and should not be expected to survive a
real corpus. 0.60 keeps 0.069 of margin below the weakest genuine answer while
still eliminating one unanswerable query before any token is spent. The remaining
two are caught by the model's `sufficient_evidence` judgement, verified live.

This also confirms the old default of 0.30 filtered nothing whatsoever, exactly
as ADR-0013 predicted it would.

### A bug this found

The first E1 run reported **nDCG@8 = 1.0164** — impossible for a normalised
metric. DCG was summing gain at every rank holding a relevant chunk while IDCG
counted labels, and overlapping chunks let several chunks satisfy one label. Now
fixed (ADR-0014) with a regression test.

Worth stating plainly: this bug was in the measurement code, and a smaller
version of it would have silently inflated every number without ever exceeding
1.0. Phase 2 would then have compared hybrid retrieval against a corrupted
baseline and drawn a confident wrong conclusion. This is the argument for
building the harness early and *running* it, rather than trusting it because the
code reads correctly.

## Phase 2 results

Corpus: 33 documents, 149 chunks. Dataset: `eval/datasets/main.jsonl`, 112
queries (100 answerable, 12 unanswerable). Embedding `BAAI/bge-small-en-v1.5`,
chunking 320/64. All retrieval-only: no LLM calls, so every run here is free.

Every number below is backed by a committed report in
[`eval/baselines/phase2/`](../eval/baselines/phase2), **including the runs for
the configurations that were rejected**. A claim that hybrid retrieval did not
help is only checkable if the run behind it still exists.

### The measurement that made Phase 2 possible

The Phase 1 corpus (5 documents, 17 chunks) could not discriminate: Recall@k
pinned at 1.000 from k=2 with zero variance. The expanded corpus restored
headroom — dense Recall@1 fell from 0.906 to 0.780 — and tightened intervals from
±0.11 to ±0.07 by taking n from 16 to 100. Without that step every number below
would have been noise.

### All configurations

| configuration | Recall@1 | nDCG@1 | Recall@8 | nDCG@8 | retrieval p50 |
|---|---|---|---|---|---|
| dense (baseline) | 0.780 | 0.800 | 0.980 | 0.895 | 77 ms |
| lexical | 0.620 | 0.640 | 0.955 | 0.805 | 2 ms |
| hybrid (RRF) | 0.720 | 0.740 | 0.990 | 0.881 | 70 ms |
| dense + rerank | **0.850** | **0.870** | **1.000** | **0.939** | 750 ms |
| hybrid + rerank | 0.850 | 0.870 | 0.990 | 0.935 | 758 ms |

### E5 — dense vs lexical vs hybrid

Paired bootstrap against dense (see [ADR-0021](../Decision.md) on why paired):

| configuration | depth | metric | delta | paired 95% CI | verdict |
|---|---|---|---|---|---|
| lexical | k=1 | Recall@1 | −0.160 | [−0.250, −0.070] | **worse** |
| lexical | k=8 | nDCG@8 | −0.089 | [−0.139, −0.039] | **worse** |
| hybrid | k=1 | Recall@1 | −0.060 | [−0.130, +0.010] | no difference |
| hybrid | k=8 | nDCG@8 | −0.014 | [−0.046, +0.018] | no difference |

**Hybrid retrieval was not adopted.** It is implemented, selectable, and measured
to make no difference on this dataset. That is a real result, not a failure to
deliver: the specification asked whether each technique actually improves things,
and the answer here is no.

The aggregate hides a genuine effect, which is why the per-kind breakdown exists:

| query kind | n | dense | lexical | hybrid | dense+rerank |
|---|---|---|---|---|---|
| identifier | 13 | 0.615 | 0.538 | 0.769 | **0.923** |
| conceptual | 13 | 0.615 | **0.846** | 0.692 | 0.846 |
| lookup | 36 | 0.861 | 0.667 | 0.750 | **0.917** |
| paraphrase | 31 | **0.871** | 0.548 | 0.742 | 0.806 |
| multi-doc | 5 | 0.400 | 0.400 | 0.400 | 0.400 |
| distractor | 2 | **1.000** | 0.500 | 0.500 | **1.000** |

Hybrid helps `identifier` queries (0.615 → 0.769) exactly as predicted before the
experiment, and hurts `paraphrase` and `lookup`. Since two thirds of this dataset
is paraphrase-or-lookup, the aggregate comes out flat. On an identifier-heavy
corpus the conclusion could invert — which is the argument for keeping the mode
rather than deleting the code.

Two other things worth reading off this table. `lexical` is the *best*
configuration for `conceptual` queries (0.846 vs dense 0.615), which was not
predicted and is not currently explained. And `multi-doc` is stuck at 0.400 for
every configuration — no retrieval strategy tested here helps a query that needs
evidence from two documents at once, which is a genuine open problem rather than
a tuning issue.

### E6 — reranking

| configuration | metric | delta vs dense | paired 95% CI | verdict |
|---|---|---|---|---|
| dense + rerank | nDCG@8 | +0.044 | [+0.009, +0.081] | **better** |
| dense + rerank | Recall@1 | +0.070 | [+0.000, +0.150] | not significant |

**Reranking was adopted**, on by default. nDCG@8 is the metric that matters
operationally — the model receives 8 chunks — and the improvement there is
significant. Recall@1 improves by more in absolute terms but its interval touches
zero.

Cost: retrieval p50 goes from 77ms to 750ms. Framed against the whole request,
generation already takes ~2.8s, so this is roughly +23% end to end rather than
10x. `ATLAS_RERANK_ENABLED=false` reverts it.

### E7 — does hybrid still contribute once reranking runs?

No.

| comparison | depth | metric | delta | paired 95% CI | verdict |
|---|---|---|---|---|---|
| hybrid+rerank vs dense+rerank | k=1 | Recall@1 | +0.0000 | [+0.0000, +0.0000] | identical |
| hybrid+rerank vs dense+rerank | k=8 | nDCG@8 | −0.003 | [−0.010, +0.000] | no difference |

A cross-encoder reads the query and passage together, which subsumes what lexical
matching contributed. Running both pays twice for one effect. This is why the
shipped configuration is **dense + rerank**, not hybrid + rerank.

### Measurement bugs found by running these experiments

Two, both of which would have corrupted conclusions rather than crashed:

1. **nDCG exceeded 1.0** (1.0164) in the first Phase 1 baseline. DCG summed gain
   at every rank holding a relevant chunk while IDCG counted labels, and
   overlapping chunks let several chunks satisfy one label. A smaller version of
   this bug would have inflated every number without ever crossing 1.0 and left
   Phase 2 comparing against a corrupted baseline ([ADR-0014](../Decision.md)).
2. **The comparison test was the wrong test.** Independent-interval overlap is
   badly conservative for paired data. Correcting it to a paired bootstrap first
   made a *negative* result significant ([ADR-0021](../Decision.md)).

### What these numbers still do not establish

Unchanged from Phase 1, and worth repeating because the numbers now look
respectable:

- One synthetic corpus, written by the same person who wrote the queries and the
  system. Real corpora contain contradictions, near-duplicates and stale versions.
- Single annotator, so "relevant" means one person's judgement.
- Several configurations are compared against one baseline with **no correction
  for multiple comparisons**. With four candidate configurations and two metrics,
  some chance of a spurious "significant" result remains.
- Answer quality is still not scored for correctness. Faithfulness is enforced
  structurally (citation resolution, quote verification); usefulness is not
  measured.

## Answer-model selection (Phase 4)

The first time `--with-answers` was ever run. Until this point every claim about
refusal correctness and citation validity described code that had never been
executed end to end.

Retrieval held constant (same retriever, dense + rerank), 112 queries, only the
answer model varied. Frozen in
[`eval/baselines/answer-models/`](../eval/baselines/answer-models).

| model | refused OK | wrongly refused | cited | unverified | p50 | $/1k Q |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 12/12 | 0 | 100/100 | 14 | 2,254 ms | $0.54 |
| **gemini-3.5-flash-lite** | 12/12 | 0 | 100/100 | 3 | 2,507 ms | **$0.71** |
| gemini-3.7-flash | 12/12 | 0 | 100/100 | 2 | 3,016 ms | $2.44 |
| gemini-3.8-flash | 12/12 | 0 | 100/100 | 1 | 3,968 ms | $2.83 |

**The guarantees hold.** Every model refused all 12 unanswerable questions,
wrongly refused none of the 100 answerable ones, and produced a resolvable
citation on every answer. The groundedness machinery from
[ADR-0010](../Decision.md) works, and that is now measured rather than asserted.

### One run cannot rank these

`gemini-3.5-flash-lite` produced **5, then 8, then 3** unverified quotes across
three runs of the same configuration. That spread is wider than most differences
between models, so a single-run table is noise dressed as a ranking.

Paired bootstrap on per-query counts, against the selected default:

| model | delta | paired 95% CI | verdict |
|---|---|---|---|
| gemini-3.1-flash-lite | +0.098 | [+0.045, +0.161] | **significantly worse** |
| gemini-3.7-flash | −0.009 | [−0.027, +0.000] | no measured difference |
| gemini-3.8-flash | −0.018 | [−0.045, +0.000] | no measured difference |

`gemini-3.5-flash-lite` is selected because nothing measurably beats it, not
because it is cheapest. Reading the first single-run table literally would have
produced the opposite conclusion from noise — the same error
[ADR-0021](../Decision.md) exists to prevent.

### Cost accounting was wrong, and the fix mattered

Cost was previously derived from a total token count split 85/15
input:output — the summary carried no breakdown. That was wrong by 15–35%,
understating `gemini-3.8-flash` most.

The reason is in the data: `3.7` and `3.8` emit **3–4× more output tokens** than
the lite models on *identical* input (141,584 tokens for all four), because they
generate thinking tokens. Those bill as output and never appear in the response
text, so a blended ratio cannot see them. Input and output are now measured and
priced separately, with thinking folded into output.

That identical input figure is also the check that retrieval really was held
constant, rather than assumed to be.

## Agent mode vs plain RAG (Phase 4, step 8)

The question Phase 4 exists to answer: does letting a model plan the retrieval
beat retrieving once? Frozen in
[`eval/baselines/step8-agent-vs-plain/`](../eval/baselines/step8-agent-vs-plain),
produced by `scripts/evaluate_agent.py`.

Both systems run on the same 112 questions, in the same process, against the
same retriever and reranker instances, so anything that differs is the system
rather than the environment.

### A stricter quality metric than the harness had

`EvalRunner` measures citation *presence* — did a non-refused answer cite
anything. That says nothing about whether it cited the **right** thing, which is
the whole question here.

So `atlas.eval.agent_compare` reuses the dataset's own label definition —
`Label.matches(document, chunk_text)`, unchanged — and applies it to the chunks
the answer actually **cited**, not the chunks that were merely retrieved. A
label counts as satisfied only if some cited chunk carries it. Citation
resolution already guarantees every cited id names a chunk in `Answer.retrieved`,
so the lookup is exact rather than approximate.

### Results

| | plain RAG (default) | agent mode (opt-in) |
|---|---|---|
| citation recall, 100 answerable | 0.975 [0.94, 1.0] | 0.970 [0.93, 1.0] |
| paired delta (agent − plain) | — | **−0.005**, CI [−0.04, 0.03] |
| unanswerable correctly refused | 12/12 | 12/12 |
| answerable wrongly refused | 0 | 0 |
| unverified quotes | 6 | 6 |
| errors / degraded runs | 0 | 0 / 0 |
| cost per 1000 questions | $0.79 | $1.40 (**1.8x**) |
| latency p50 / p95 | 3,286 / 4,286 ms | 5,132 / 7,050 ms |

**96 of 100 answerable questions scored identically.** Four differed: two
favoured the agent, two favoured plain RAG. The paired delta's interval crosses
zero, and so does every per-kind interval — multi-doc [0.0, 0.3], identifier
[0.0, 0.23], paraphrase [−0.16, 0.0], each touching zero at one edge.

Refusal correctness and unverified-quote counts being *identical* is the
expected result, not a coincidence: both systems share one answering path
([ADR-0030](../Decision.md)), so the groundedness machinery cannot behave
differently depending on how the evidence was gathered.

### The four discordant questions

Named, because four differences out of a hundred is a small enough number that
aggregate statistics say less than the cases themselves.

**Agent won:** `refund-window-by-plan` (multi-doc — plain cited one of two
required documents, the agent's second search reached both) and
`hybrid-flag-name` (identifier).

**Agent lost:** both `paraphrase`, and both investigated rather than left as a
number. In `revocation-delay` the union rerank ranked the correct document
*first* and the answer model cited a different, genuinely relevant document
instead — a citation choice, not a retrieval miss. In `queue-first-checks` the
right document filled 3 of 7 evidence slots but the labelled line sat in a
section none of the searches surfaced — a real retrieval-depth gap.

### What this does and does not support

**Supports:** the agent is bounded and robust — 0 errors, 0 degraded runs and 0
bound-hits across 100 live executions — and it does not regress answer quality.

**Does not support:** that it is worth its cost. It ships opt-in and off by
default ([ADR-0032](../Decision.md)). The one place it shows promise,
multi-document questions, rests on five instances, which is why E10 stays open
rather than being marked solved.

**Limits.** One run, not repeated. Earlier work established that the
unverified-quote metric varies run to run (5, then 8, then 3 on identical
configuration), so a single draw on a 112-question set is evidence, not proof.

### An earlier diagnostic, kept

Before this run, a 7-question experiment tested one hypothesis: that the agent's
evidence ordering, not its searching, was what hurt answer quality
([`eval/baselines/agent-rerank/`](../eval/baselines/agent-rerank)). It compared
interleaved evidence against a single rerank of the deduplicated union, holding
the agent's searches **fixed** by building both arms from one gathered plan —
otherwise the ordering change would have been confounded with the model choosing
different searches.

Coverage across two runs: plain 26/26, interleave 23/26, union rerank 26/26.
One difference reproduced and one did not; both are recorded, because discarding
the inconvenient half of a small sample is how seven questions get talked into
meaning more than they do. The union rerank was adopted
([ADR-0031](../Decision.md)) because it never lost, cost no extra tokens, and
stops relying on an ordering the score scale does not support.

## Planned experiments

Recorded here in advance so results are not selected after the fact.

| # | Question | Status |
|---|---|---|
| E1 | What is the dense baseline? | **done** — rerun on the expanded corpus |
| E2 | Where should the similarity floor sit? | **done**, then invalidated by the larger corpus ([ADR-0019](../Decision.md)) |
| E5 | Does lexical retrieval add anything over dense? | **done** — no; hybrid not adopted ([ADR-0018](../Decision.md)) |
| E6 | Does cross-encoder reranking pay for its latency? | **done** — yes; adopted ([ADR-0020](../Decision.md)) |
| E7 | Does hybrid still contribute once reranking runs? | **done** — no; reranking subsumes it |
| E3 | Does structure-aware chunking beat fixed-size? | deferred — would move [ADR-0009](../Decision.md) off `provisional` |
| E4 | What chunk size and overlap? | deferred |
| E8 | Is a larger embedding model worth it? | deferred — `bge-small` vs alternatives |
| E9 | Why is lexical retrieval best on conceptual queries? | open — unexplained result from E5 |
| E10 | Can anything help multi-document queries? | open — stuck at 0.400 for every configuration tested |
| E11 | Which model should write the answer? | **done** — no measurable gain above `3.5-flash-lite` ([ADR-0024](../Decision.md)) |
| E12 | Which model should route agent tool calls? | **done** — the rebuilt 16-case set scored 16/16 for every candidate; chose on latency and cost ([ADR-0025](../Decision.md)) |
| E13 | Does reranking the agent's evidence union fix its ordering? | **done** — adopted; never worse, one reproducible win ([ADR-0031](../Decision.md)) |
| E14 | Does agent mode beat plain RAG? | **done** — no significant difference at 1.8x cost; kept opt-in ([ADR-0032](../Decision.md)) |

E9 and E10 were raised *by* the Phase 2 results rather than planned, and are
recorded so they are not quietly forgotten. E10 is still open: agent mode was
the most plausible attack on it and produced one win out of five multi-document
questions, which is not enough to close it.

## What this evaluation cannot tell you

Stated so the numbers are not over-read:

- **19 queries over 5 documents is a smoke set, not a benchmark.** It catches
  regressions. It does not establish that this system would work on a real
  corpus of thousands of documents.
- **The labels are single-annotator.** There is no inter-annotator agreement
  figure, so "relevant" means "one person judged it relevant".
- **The corpus is synthetic**, written alongside the queries. Real corpora have
  contradictions, near-duplicates, outdated versions and inconsistent
  terminology, all of which are exactly what makes retrieval hard.
- **Answer quality is not scored for correctness.** Faithfulness (is the answer
  supported by its citations?) is checked structurally. Whether the answer is
  actually *useful* is not measured, and an LLM-as-judge would need its own
  validation before its scores meant anything.

The last point is the honest ceiling of the current setup. Widening the dataset
and adding a validated faithfulness judge would be the next real step.

One qualification, added after the agent evaluation: answer quality is now
scored *slightly* more strictly than "faithfulness is structural" implies. The
plain-vs-agent comparison checks whether the answer's citations satisfy the
question's gold labels, which is closer to correctness than citation presence
is. It is still not a usefulness judgement, and it inherits every limit above —
single annotator, synthetic corpus, 112 questions.
