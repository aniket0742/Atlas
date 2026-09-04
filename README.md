# Atlas

A retrieval platform that answers questions about an organisation's own
documents, with citations, and refuses when the answer is not in the corpus.

**Status: Phase 2 complete.** Lexical retrieval, hybrid fusion and cross-encoder
reranking are implemented and measured against a recorded baseline. Two of the
three were **not adopted**, because measurement said they did not help — the
reasoning and the numbers are in [`docs/evaluation.md`](docs/evaluation.md) and
[`Decision.md`](Decision.md). Every number below was measured on this machine.

---

## The problem

An organisation's knowledge is spread across policy documents, engineering docs,
wikis and repositories. People ask questions whose answers exist somewhere in
that material, and finding them is a search problem that keyword search handles
badly and that a general-purpose chatbot handles worse — because a chatbot with
no access to the corpus will answer anyway.

The failure mode that matters is not "the system could not find the answer". It
is "the system produced a confident, plausible, wrong answer, and nothing in the
response indicated which parts were grounded." Atlas is built around preventing
that specific outcome.

## What makes this different from a "chat with your PDFs" demo

Three things are enforced in code rather than requested in a prompt:

- **A citation cannot name a source that was not retrieved.** Evidence ids are
  server-generated per request; the model's cited ids are validated against that
  exact set, and anything else is discarded.
- **A quote is checked against the chunk it cites.** Verbatim match required;
  mismatches are surfaced and counted, not silently accepted.
- **An answer with no resolvable citation is converted into a refusal.** An
  uncited answer from a retrieval system is indistinguishable from a guess.

And one thing is enforced in the schema: **`tenant_id` is on every table and in
every query from the first migration**, with document ids derived from the tenant
so two tenants uploading identical files get different ids by construction — even
though Phase 1 has a single tenant and no authentication. Cross-tenant leakage is
the worst bug this system could have, and retrofitting isolation is how it
happens.

Reasoning for these and every other significant choice is in
[`Decision.md`](Decision.md).

## Architecture

```text
             ┌──────────────┐
  documents  │ Ingestion    │   parse → normalise → chunk → embed → index
  ──────────▶│ pipeline     │   (synchronous in Phase 1; queued in Phase 3)
             └──────┬───────┘
                    │
             ┌──────▼─────────────────────────────────┐
             │ PostgreSQL 16 + pgvector               │
             │                                        │
             │  tenants / sources                     │
             │  documents  (content, hash, version)   │
             │  chunks     (text, char offsets)       │
             │  chunk_embeddings (per model, HNSW)    │
             └──────┬─────────────────────────────────┘
                    │
             ┌──────▼───────┐
  question   │ Retrieval    │   dense  (default)
  ──────────▶│              │   lexical / hybrid RRF  (selectable, measured)
             │              │   → cross-encoder rerank  (default on)
             └──────┬───────┘
                    │ evidence, if the query passed the similarity gate
             ┌──────▼───────┐
             │ Answering    │   structured output, then:
             │              │     resolve citations
             │              │     verify quotes
             │              │     downgrade uncited answers
             └──────┬───────┘
                    │
              answer + citations (or an explicit refusal)
```

The API also serves a static inspection console at `/` — see
[Inspection console](#inspection-console).

One process, one database. Redis is in `docker-compose.yml` for Phase 3 and 6
but is not used yet. There is no message broker — see
[ADR-0002](Decision.md#adr-0002-no-kafka-a-queue-is-deferred-and-will-probably-be-postgres)
for why Kafka was considered and rejected.

## Technology

| Component | Choice | Why (short form) |
|---|---|---|
| API | FastAPI, Python 3.11 | async, typed request/response models |
| Database | PostgreSQL 16 + pgvector | one transaction covers metadata *and* vectors ([ADR-0001](Decision.md)) |
| DB driver | psycopg 3, hand-written SQL | the interesting queries are pgvector operators ([ADR-0005](Decision.md)) |
| Migrations | numbered `.sql` + checksum | index DDL that autogenerate cannot model ([ADR-0006](Decision.md)) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (local, CPU, ONNX) | free re-indexing makes retrieval experiments affordable ([ADR-0007](Decision.md)) |
| Lexical search | PostgreSQL FTS (`tsvector` + GIN, `ts_rank_cd`) | no new infrastructure; **not BM25**, and not called that ([ADR-0017](Decision.md)) |
| Reranking | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder, local | the one change measured to beat the baseline ([ADR-0020](Decision.md)) |
| Generation | Google Gemini free tier, behind a `Protocol` | no paid dependency; swappable ([ADR-0008](Decision.md)) |
| UI | plain HTML/CSS/JS served by FastAPI | same-origin, no build step, no npm ([ADR-0015](Decision.md)) |
| Local env | Docker Compose (api + postgres + redis) | one command starts everything ([ADR-0016](Decision.md)) |
| Tests | pytest | 76 unit tests run with no database and no API key |

## Running it locally

### Prerequisites

- Python 3.11+
- Docker Desktop (for Postgres with pgvector)
- A Gemini API key from <https://aistudio.google.com/apikey> — free tier is
  sufficient. Not needed for the test suite.

### Setup

```bash
cp .env.example .env            # then put your GEMINI_API_KEY in it
docker compose up -d --build    # api + postgres + redis; migrations run on start
```

Then open **<http://localhost:8000>**.

That is the whole setup. The API is containerised ([ADR-0016](Decision.md)), so
no host Python environment is needed to run Atlas. The first query downloads
~67MB of ONNX embedding weights into the `atlas_models` volume, where they
survive image rebuilds.

For the CLI, the test suite, or the eval harness, you also want a local
environment:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

### Confirm your model

Free-tier model availability changes and is not reliably documented per model, so
Atlas does not hard-code a claim about it:

```bash
atlas models                    # lists what your key can actually reach
```

Set `ATLAS_LLM_MODEL` in `.env` to one of those.

### Index and ask

```bash
atlas ingest eval/corpus --source handbook
atlas stats

atlas query "How long do customers have to request a refund?"
atlas query "How many vacation days do employees get?"   # should refuse
```

Or run the API on the host instead of in a container:

```bash
atlas serve                     # http://127.0.0.1:8000
```

## Inspection console

`http://localhost:8000` serves a single-page console for driving and inspecting
the system. It is a technical instrument rather than a chat product: everything
it shows is something the API already returns.

| Panel | Shows |
|---|---|
| Ask | question, with per-request overrides for retrieval mode, `top_k`, `min_similarity` and reranking |
| Answer | the answer, or the refusal and its `refusal_reason` |
| Citations | quote, document, character span, page, and a **verbatim / not verbatim** badge |
| Retrieved chunks | every chunk sent to the model, in rank order, with **per-component scores**: dense similarity and rank, lexical rank, fused RRF score, reranker score |
| Request | retrieval mode, whether reranking ran, the gate's best dense score, per-stage latency and token usage |
| Corpus | document and chunk counts, per-document indexing status |
| Add a document | upload and index a file, with an explicit processing state |

Switching the retrieval mode and re-asking the same question is the quickest way
to see hybrid fusion working: the per-component chips show a chunk's dense rank,
its lexical rank, and the fused score that put it where it is.

The **verbatim badge** is the panel worth looking at: it makes the otherwise
invisible groundedness guarantee visible. A citation is only listed at all if its
id matched a chunk actually sent to the model, and the badge says whether the
quote was found character-for-character in that chunk.

Responses are **not streamed** — the request blocks until the model returns, and
the console says so rather than animating text that has already arrived
([ADR-0015](Decision.md)). Uploads are synchronous for the same reason Phase 1
ingestion is, so the upload panel states what it is doing while it waits.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness, database reachability, active models |
| `GET` | `/v1/stats` | document / chunk / embedding counts for the tenant |
| `POST` | `/v1/documents` | upload and index a document (multipart) |
| `GET` | `/v1/documents` | list indexed documents |
| `GET` | `/v1/documents/{id}` | document detail, including failure reason |
| `GET` | `/v1/sources` | configured sources and their document counts |
| `POST` | `/v1/query` | ask a question; returns answer, citations, usage, timings |

`POST /v1/query` returns per-stage timings and token usage on every response, so
the numbers Phase 6 (observability) and Phase 8 (evaluation) need are available
from the start rather than requiring an API change later.

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question":"What happens after a chargeback?","include_evidence":true}'
```

Error codes are meaningful: `415` unsupported document type, `422` document could
not be parsed, `502` the model provider failed, `504` the model provider timed
out. Atlas being healthy while its upstream is not is a distinct condition.

## Evaluation

Retrieval quality is measured, not asserted. The methodology, the label format
and its rationale are in [`docs/evaluation.md`](docs/evaluation.md).

```bash
atlas eval eval/datasets/smoke.jsonl               # retrieval only; free, no LLM
atlas eval eval/datasets/smoke.jsonl --with-answers  # adds refusal + citation metrics
```

Reports are JSON, written to `eval/results/`, and each one records the full
configuration that produced it — embedding model, chunk parameters, `k`,
similarity floor. A retrieval number without those is not comparable to anything.

Metrics reported: Recall@k, Precision@k, MRR, nDCG@k, each with a 95% bootstrap
confidence interval, plus (with `--with-answers`) refusal correctness in both
directions, citation coverage, unverified-quote count, latency percentiles and
token totals.

The shipped dataset has 19 queries (16 answerable, 3 unanswerable). That is a
smoke set for catching regressions, not a benchmark, and the confidence intervals
are wide enough to say so. Overlapping intervals mean no measured difference.

**Measured** on 112 queries over 33 documents (149 chunks), retrieval only:

| configuration | Recall@1 | nDCG@8 | retrieval p50 | adopted |
|---|---|---|---|---|
| dense (baseline) | 0.780 | 0.895 | 77 ms | — |
| lexical | 0.620 | 0.805 | 2 ms | no — significantly worse |
| hybrid (RRF) | 0.720 | 0.881 | 70 ms | no — no measured difference |
| **dense + rerank** | **0.850** | **0.939** | 750 ms | **yes** |
| hybrid + rerank | 0.850 | 0.935 | 758 ms | no — hybrid adds nothing here |

Comparisons use a **paired** bootstrap over per-query differences, not overlap of
independent intervals. The rule registered before the experiments used the
latter, which is the wrong test for paired data; the correction and the reasons
it is not post-hoc rationalisation are in [ADR-0021](Decision.md).

Two of three Phase 2 techniques were implemented, measured, and **not adopted**.
That is the intended outcome of measuring rather than assuming.

## Testing

```bash
pytest                          # 72 unit tests, no database or API key needed
docker compose up -d && pytest -m integration   # 12 more against real Postgres
```

The integration tests cover the properties that are hard to reason about without
a database: idempotent re-ingestion, version replacement without orphaned chunks,
concurrent ingestion of the same document, tenant isolation, and refusal on an
empty corpus.

## Known limitations

Current, and honest:

- **Ingestion is synchronous.** A large PDF blocks its own request. No retry —
  a failed document is marked `failed` with its error and must be resubmitted.
  ([ADR-0012](Decision.md))
- **Chunking is unvalidated.** Structure-aware chunking is implemented and its
  invariants are tested, but whether it beats fixed-size chunking on this corpus
  has not been measured. Marked `provisional`. ([ADR-0009](Decision.md))
- **No authentication.** Every request is attributed to one configured tenant.
  The tenant plumbing is complete; the identity layer is Phase 5.
- **Multi-document queries are unsolved.** Questions needing evidence from two
  documents score Recall@1 of 0.400 under *every* configuration tested. No
  retrieval strategy here helps; it is an open problem, not a tuning gap.
- **Reranking costs ~680ms per query.** Adopted because the quality gain is
  measured and significant, but it is the dominant retrieval cost. Disable with
  `ATLAS_RERANK_ENABLED=false`.
- **The similarity floor no longer separates.** At 33 documents, 10 of 12
  unanswerable queries score above it. It is an interim crash barrier at 0.55,
  not a validated threshold ([ADR-0019](Decision.md)).
- **Lexical search is English-only** (`to_tsvector('english', ...)`) and is
  PostgreSQL FTS rather than BM25 ([ADR-0017](Decision.md)).
- **Embedding throughput is a concern.** *Measured* at roughly 350 ms per
  ~250-token chunk on one laptop CPU with the quantised ONNX build. Fine
  interactively, slow for bulk ingestion. Phase 6 target.
- **No OCR.** Scanned PDFs fail ingestion with an explicit message rather than
  indexing as empty.
- **No streaming.** `/v1/query` returns one complete response, so the console
  shows a spinner for the duration of the model call rather than incremental
  text. Deferred deliberately ([ADR-0015](Decision.md)).
- **The container image is a development image.** Editable install, root user,
  bind-mounted source. Phase 7 produces a deployment image.
- **Prompt injection is contained, not solved.** Retrieved text cannot mint a
  citation and, in Phase 1, cannot trigger an action because there are no tools.
  A document that states something false will be faithfully reported as stating
  it. ([ADR-0010](Decision.md))

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 1 | End-to-end RAG, citations, tenancy in schema, **eval harness** | complete, E1+E2 run |
| 2 | Lexical + hybrid + reranking, measured against the Phase 1 baseline | complete — reranking adopted, hybrid rejected |
| 3 | Postgres job queue, workers, retries, DLQ, incremental re-crawl | planned |
| 4 | Tool use: knowledge base, GitHub diffs, fixed metadata queries | planned |
| 5 | AuthN/AuthZ, RBAC, rate limiting, filtered-recall fix for HNSW | planned |
| 6 | OpenTelemetry, Prometheus, caching, embedding throughput | planned |
| 7 | Deployment, load testing | planned |
| 8 | Expanded eval, failure injection, write-up | planned |

Two deliberate departures from the original plan, both argued in `Decision.md`:
the eval harness ships in Phase 1 rather than Phase 8 (otherwise Phase 2's
"hybrid retrieval improved recall" claim has no baseline to compare against), and
tenancy lands in the schema in Phase 1 rather than Phase 5.

## Project layout

```text
src/atlas/
  api/          FastAPI app, request/response schemas
  answer/       prompt construction, citation validation, refusal policy
  core/         domain models, deterministic id derivation
  db/           connection pool, migrations, all SQL
  eval/         dataset format, metrics, runner
  ingest/       parsers, normalisation, chunking, pipeline
  providers/    embedding + LLM protocols, Gemini, fastembed, offline fakes
  api/static/   the inspection console (3 files, no build step)
  cli.py        migrate / ingest / query / eval / models / serve
migrations/     numbered SQL
scripts/        calibrate_floor.py (experiment E2)
eval/corpus/    sample knowledge base
eval/datasets/  labelled evaluation queries
eval/baselines/ frozen Phase 1 retrieval numbers Phase 2 must beat
tests/          unit tests (no infra) + integration tests (marked)
Dockerfile      API image (development)
```
