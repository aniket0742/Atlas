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

## Results so far

Run on the 19-query smoke set, `bge-small-en-v1.5`, structure-aware chunking at
320/64 tokens. Reports in `eval/results/`.

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

## Planned experiments

Recorded here in advance so results are not selected after the fact.

| # | Question | Compares | Phase |
|---|---|---|---|
| E1 | What is the dense baseline? | dense, default parameters | 1 |
| E2 | Where should the similarity floor sit? | sweep floor; refusal correctness both directions | 1 |
| E3 | Does structure-aware chunking beat fixed-size? | ADR-0009 vs fixed windows, same model | 1–2 |
| E4 | What chunk size and overlap? | sweep target/overlap tokens | 2 |
| E5 | Does BM25 add anything over dense? | dense vs hybrid fusion | 2 |
| E6 | Does cross-encoder reranking pay for its latency? | hybrid vs hybrid + rerank, with latency | 2 |
| E7 | Is a larger embedding model worth it? | bge-small vs alternatives | 2 |

E1 and E2 are the immediate next steps once a database is running. E3 is what
moves [ADR-0009](../Decision.md) from `provisional` to `accepted` or replaces it.

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
and adding a validated faithfulness judge is Phase 8 work.
