# Atlas

[![CI](https://github.com/aniket0742/Atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/aniket0742/Atlas/actions/workflows/ci.yml)

A retrieval platform that answers questions about an organisation's own
documents, with citations, and refuses when the answer is not in the corpus.

**Status: Phase 4 complete.** Retrieval, asynchronous ingestion, and a bounded
agent/tool framework are implemented and evaluated.

The evaluation is the point. Four significant pieces of work — lexical
retrieval, hybrid fusion, hybrid-plus-reranking, and agent mode — are
implemented, tested, and **not enabled by default**, because the measurements
did not support adopting them. The runs behind every one of those decisions are
committed under [`eval/baselines/`](eval/baselines), including the runs for the
rejected configurations.

The default path is **dense retrieval + cross-encoder reranking + a grounded,
citation-validated answer**. Reasoning in [`Decision.md`](Decision.md), numbers
in [`docs/evaluation.md`](docs/evaluation.md).

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

Enforced in code, not requested in a prompt:

- **A citation cannot name a source that was not retrieved.** Evidence ids are
  server-generated per request and validated against that exact set.
- **A quote is checked verbatim against the chunk it cites.** Mismatches are
  surfaced and counted, not silently accepted.
- **An uncited answer is converted into a refusal**, not served as a guess.
- **`tenant_id` is on every table and every query from the first migration**,
  with ids derived from the tenant — even though there is a single tenant and no
  authentication today. Retrofitting isolation is how cross-tenant leaks happen.
- **A tool may not declare `tenant_id` as an argument at all.** A poisoned
  document instructing the model to "search tenant acme-corp" has nowhere to
  land — tested with real injection payloads ([ADR-0027](Decision.md)).
- **Techniques that did not help were not shipped as defaults.** Lexical
  retrieval, hybrid fusion, hybrid+reranking and agent mode are all
  implemented, tested, and off — the rejected runs are committed next to the
  adopted ones so the claims are checkable, not asserted.

Reasoning for every significant choice is in [`Decision.md`](Decision.md),
which opens with an index of all 33 decisions and their status.

## Architecture

```mermaid
flowchart TB
    subgraph store["PostgreSQL 16 + pgvector"]
        DB[("tenants · sources · documents<br/>chunks · char offsets<br/>chunk_embeddings · HNSW")]
    end

    subgraph ingest["Ingestion — asynchronous"]
        direction LR
        UP["POST /v1/documents<br/>202 + job id"] --> Q[("ingest_jobs<br/>SKIP LOCKED queue")]
        Q --> W["Worker process<br/>leases · retries · DLQ"]
        W --> CH["parse → normalise →<br/>structure-aware chunk"]
        CH --> EM["BGE-small-en-v1.5<br/>local ONNX, CPU"]
    end

    EM --> DB
    UP -. "enqueue and document write<br/>share one transaction" .-> DB

    QU["question"] --> ROUTE{"agent=true?"}

    ROUTE -- "no — DEFAULT" --> R1["Dense retrieval<br/>pgvector cosine, HNSW"]
    R1 --> RR["Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2"]

    ROUTE -- "yes — OPT-IN" --> AG["Agent loop<br/>gemini-3.1-flash-lite<br/>bounded: 4 iterations · 8 calls · 60s"]
    AG <--> TOOL["search_knowledge_base<br/>tenant from server context only"]
    TOOL --> R1
    AG --> UNION["dedupe union →<br/>rerank once vs original question"]

    DB -.-> R1
    RR --> EV["evidence"]
    UNION --> EV
    EV --> GEN["gemini-3.5-flash-lite<br/>structured output"]

    subgraph guard["Groundedness — enforced in code, not prompted"]
        V1["resolve citations against<br/>server-generated ids"] --> V2["verify quotes verbatim"]
        V2 --> V3["downgrade uncited<br/>answers to refusals"]
    end

    GEN --> V1
    V3 --> OUT["answer + citations,<br/>or an explicit refusal"]

    classDef default fill:#f6f7f9,stroke:#8892a4,color:#16181d
    classDef agent fill:#fdf3e0,stroke:#8a5a00,color:#3b2a08,stroke-dasharray:4 3
    classDef guardian fill:#e6f4ef,stroke:#17795e,color:#0d3a2c
    classDef storage fill:#e8eefb,stroke:#2c5cc5,color:#12224a
    class AG,TOOL,UNION agent
    class V1,V2,V3 guardian
    class DB,Q storage
```

**Solid is the default path.** Dashed amber is agent mode — opt-in, off by
default, and measured to give no significant quality gain at ~1.8x the cost
([ADR-0032](Decision.md#adr-0032-step-8-agent-mode-matches-plain-rag-recommendation-is-opt-in-not-default)).
Both paths converge on **one** answering step — same prompt, same
server-generated ids, same citation resolution and refusal downgrade either way
([ADR-0030](Decision.md#adr-0030-one-answering-path-two-ways-of-choosing-evidence)).
The agent only ever chooses *which* evidence arrives.

One process, one database. The API also serves the inspection console at `/`
(below). Redis is in `docker-compose.yml`, unused; there is no message broker
([ADR-0002](Decision.md#adr-0002-no-kafka-the-queue-is-postgres)).

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
| Generation | `gemini-3.5-flash-lite` behind a `Protocol` | selected because nothing measurably beat it ([ADR-0024](Decision.md)) |
| Agent (opt-in) | `gemini-3.1-flash-lite` + in-house tool registry | bounded loop; no LangChain ([ADR-0026](Decision.md), [ADR-0029](Decision.md)) |
| UI | plain HTML/CSS/JS served by FastAPI | same-origin, no build step, no npm ([ADR-0015](Decision.md)) |
| Queue | Postgres `SELECT ... FOR UPDATE SKIP LOCKED` | enqueue shares a transaction with the write; no broker ([ADR-0002](Decision.md), [ADR-0022](Decision.md)) |
| Local env | Docker Compose (api + worker + postgres + redis) | one command starts everything ([ADR-0016](Decision.md)) |
| Tests | pytest | 276 unit tests run with no database and no API key; 41 more need only Postgres |
| CI | GitHub Actions | lint + both suites on every push, no secrets required ([ADR-0033](Decision.md)) |

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

Then open **<http://localhost:8000>**. That's the whole setup — no host Python
needed ([ADR-0016](Decision.md)). The first query downloads ~67MB of ONNX
weights into the `atlas_models` volume; they survive rebuilds.

For the CLI, tests, or the eval harness, add a local environment:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

### Confirm your model

Free-tier availability changes and isn't reliably documented per model:

```bash
atlas models                    # lists what your key can actually reach
```

Set `ATLAS_LLM_MODEL` in `.env` to one of those, then **recreate**, don't
restart:

```bash
docker compose up -d --force-recreate api worker
```

`env_file` is read at container *creation*. `restart` reuses the existing
container's environment, so code changes take effect (source is bind-mounted)
but config changes silently don't — the console keeps reporting the old model.

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
| Agent mode | a toggle that runs the same question through the opt-in agent path, showing the searches it chose, the stop reason, and whether it fell back |
| Answer | the answer, or the refusal and its `refusal_reason` |
| Citations | quote, document, character span, page, and a **verbatim / not verbatim** badge |
| Retrieved chunks | every chunk sent to the model, in rank order, with **per-component scores**: dense similarity and rank, lexical rank, fused RRF score, reranker score |
| Request | retrieval mode, whether reranking ran, the gate's best dense score, per-stage latency and token usage |
| Corpus | document and chunk counts, per-document indexing status |
| Add a document | upload a file; returns a job id immediately and polls it to completion |
| Ingestion queue | pending / running / succeeded / dead counts, oldest pending age, and per-job errors with a requeue button |

Switching the retrieval mode and re-asking the same question is the quickest way
to see hybrid fusion working: the per-component chips show a chunk's dense rank,
its lexical rank, and the fused score that put it where it is.

The **verbatim badge** is the panel worth looking at: it makes the otherwise
invisible groundedness guarantee visible. A citation is only listed at all if its
id matched a chunk actually sent to the model, and the badge says whether the
quote was found character-for-character in that chunk.

Turning **agent mode** on disables the single-search controls, because the API
rejects them rather than silently ignoring them: they describe one search, and
the agent runs several with parameters it chooses. Reporting a `top_k` that
never applied would misdescribe what ran.

Responses are **not streamed** — the request blocks until the model returns, and
the console says so rather than animating text that has already arrived
([ADR-0015](Decision.md)). Uploads are asynchronous: the upload returns a job id
immediately and the console polls that job, because claiming success on the 202
would report a document as searchable before any worker had touched it.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness, database reachability, active models |
| `GET` | `/v1/stats` | document / chunk / embedding counts for the tenant |
| `POST` | `/v1/documents` | queue a document for indexing (multipart) — returns **202** |
| `GET` | `/v1/documents` | list indexed documents |
| `GET` | `/v1/documents/{id}` | document detail, including failure reason |
| `GET` | `/v1/sources` | configured sources and their document counts |
| `POST` | `/v1/query` | ask a question; returns answer, citations, usage, timings |
| `GET` | `/v1/jobs` | queue depth and recent ingestion jobs |
| `GET` | `/v1/jobs/{id}` | one job's status, attempts and error |
| `POST` | `/v1/jobs/{id}/requeue` | return a dead-letter job to the queue |

`POST /v1/query` returns per-stage timings and token usage on every response, so
the numbers Phase 6 (observability) and Phase 8 (evaluation) need are available
from the start rather than requiring an API change later.

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question":"What happens after a chargeback?","include_evidence":true}'

# agent=true is opt-in and off by default; the response then carries an
# agent_trace (searches issued, stop reason, evidence union before/after rerank).
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question":"What is the refund window, and who approves exceptions?","agent":true}'
```

Combining `agent` with the single-search knobs (`top_k`, `mode`, `rerank`,
`min_similarity`, `source_ids`) is a **400**, not a silent ignore — they
describe one search, and the agent runs several with parameters it chooses.

**`POST /v1/documents` returns 202, not 201.** The document is queued, not
indexed — it is *not* searchable when the response arrives. Poll
`GET /v1/jobs/{job_id}`. This is a deliberate unversioned break; the reasoning is
in [ADR-0023](Decision.md). One consequence: document-type validation moved into
the worker, so a file this endpoint accepts can still fail, surfacing as a
dead-lettered job rather than a 415.

Error codes are meaningful: `413` too large, `422` empty upload, `502` the model
provider failed, `504` the model provider timed out. Atlas being healthy while
its upstream is not is a distinct condition.

## Ingestion

```bash
docker compose up -d          # starts a worker alongside the API
atlas jobs                    # queue depth and recent jobs
atlas worker --once           # drain the queue in the foreground, then exit
```

Uploads enqueue a job **in the same transaction that creates the source**, which
is the property a message broker cannot provide without an outbox table — and an
outbox table is a Postgres queue with an extra hop ([ADR-0002](Decision.md)).

Reliability behaviour, all covered by tests:

- **Concurrent workers never claim the same job** (`SKIP LOCKED`).
- **A crashed worker loses no work.** Jobs carry a lease; a reaper returns
  expired ones to the queue. Attempts are counted at *claim* time, so a document
  that reliably kills workers exhausts its budget instead of cycling forever.
- **Retries back off exponentially with jitter**, then land in a dead-letter
  state that **keeps its payload** so it can be requeued once the cause is fixed.
- **Unparseable documents are not retried** — the same bytes fail identically, so
  retrying only burns the budget and delays the real error.
- **Duplicate delivery is safe.** Deterministic ids plus the content-hash
  short-circuit make reprocessing converge rather than duplicate.

## Evaluation

Retrieval quality is measured, not asserted. Full methodology, per-kind
breakdowns and every frozen run are in
[`docs/evaluation.md`](docs/evaluation.md) and [`eval/baselines/`](eval/baselines);
this section is the headline numbers only.

```bash
atlas eval eval/datasets/smoke.jsonl                 # retrieval only; free, no LLM
atlas eval eval/datasets/smoke.jsonl --with-answers  # adds refusal + citation metrics
```

**Retrieval**, 112 queries over 33 documents (149 chunks):

| configuration | Recall@1 | nDCG@8 | p50 | adopted |
|---|---|---|---|---|
| dense (baseline) | 0.780 | 0.895 | 77 ms | — |
| lexical | 0.620 | 0.805 | 2 ms | no — worse |
| hybrid (RRF) | 0.720 | 0.881 | 70 ms | no — no difference |
| **dense + rerank** | **0.850** | **0.939** | 750 ms | **yes** |
| hybrid + rerank | 0.850 | 0.935 | 758 ms | no — adds nothing over rerank |

**Answer quality**, same 112 queries, retrieval held constant: every candidate
model refused 12/12 unanswerable and wrongly refused 0/100 answerable.
`gemini-3.5-flash-lite` was selected because `gemini-3.8-flash` is **not
measurably better** at 4x the cost ([ADR-0024](Decision.md)). Tool-routing
accuracy saturated at 16/16 for every candidate tested; `gemini-3.1-flash-lite`
won on latency and cost ([ADR-0025](Decision.md),
[`eval/baselines/agent-routing/`](eval/baselines/agent-routing)).

**Agent mode vs. plain RAG**, the same 112 questions through both systems,
paired, scored by whether the answer's *citations* (not just its retrieval)
satisfy the gold labels
([`eval/baselines/step8-agent-vs-plain/`](eval/baselines/step8-agent-vs-plain)):

| | plain RAG (default) | agent mode (opt-in) |
|---|---|---|
| citation recall, 100 answerable | 0.975 | 0.970 |
| paired delta (agent − plain) | — | **−0.005**, CI crosses zero |
| cost / latency vs. plain | — | **1.8x cost, +56–64% latency** |

96 of 100 questions scored identically; of the four that differed, two favoured
each system. **No measured quality gain justifies the cost**, so agent mode
ships opt-in and off by default
([ADR-0032](Decision.md#adr-0032-step-8-agent-mode-matches-plain-rag-recommendation-is-opt-in-not-default)).

All comparisons use a **paired** bootstrap over per-query differences, not
independent-interval overlap — the correction and why it matters are in
[ADR-0021](Decision.md). Four pieces of work (lexical, hybrid, hybrid+rerank,
agent mode) were implemented, measured, and **not adopted as default**; every
rejected run is committed, not deleted.

## Testing

```bash
pytest -m "not integration"     # 276 tests, no database or API key needed
# Integration tests need a database no live worker is polling: jobs.claim() is
# global by design, so a running worker competes with the tests for their jobs.
docker compose up -d && docker compose stop worker && pytest -m integration
```

All 317 tests run on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
against a real `pgvector` Postgres, with **no API key**: every test uses the
deterministic offline providers, so a fork runs the whole suite with access to
nothing. The evaluation harness is deliberately not in CI — it calls real models
and costs real money; its results are frozen under `eval/baselines/` instead.

The integration job fails if its tests *skip*. They skip when no database is
reachable, which is right locally and would otherwise mean a broken service
container produced a green build that tested nothing.

The integration tests cover the properties that are hard to reason about without
a database: idempotent re-ingestion, version replacement without orphaned chunks,
concurrent ingestion of the same document, tenant isolation, and refusal on an
empty corpus.

## Known limitations

Current, and honest — see [Deliberately out of scope](#deliberately-out-of-scope)
for what was decided against rather than merely not yet measured:

- **Chunking is unvalidated.** Implemented and tested, but not measured against
  fixed-size chunking. Marked `provisional` ([ADR-0009](Decision.md)).
- **Multi-document queries are unsolved.** Recall@1 of 0.400 under *every*
  retrieval configuration tested — an open problem, not a tuning gap.
- **The similarity floor (0.55) no longer separates.** At 33 documents, 10 of 12
  unanswerable queries score above it; it's a crash barrier, not a validated
  threshold ([ADR-0019](Decision.md)).
- **Lexical search is English-only**, PostgreSQL FTS rather than BM25
  ([ADR-0017](Decision.md)).
- **Embedding throughput**: ~350ms per ~250-token chunk on one laptop CPU. Fine
  interactively, slow for bulk ingestion.
- **No OCR.** Scanned PDFs fail ingestion explicitly rather than indexing empty.
- **Prompt injection is contained, not solved.** A tool cannot be told whose
  data to read — tested with poisoned-document payloads
  ([ADR-0027](Decision.md)) — but retrieved text can still steer what the agent
  searches for, and a false document is faithfully reported as saying so
  ([ADR-0010](Decision.md)).
- **Agent mode is not proven useful.** No significant quality difference vs.
  plain RAG at 1.8x cost; ships opt-in for that reason
  ([ADR-0032](Decision.md#adr-0032-step-8-agent-mode-matches-plain-rag-recommendation-is-opt-in-not-default)).
- **`mypy --strict` reports 75 errors**, mostly `Any` from untyped SDKs. CI runs
  it as informational output, not a gate — a gate that's never been green isn't
  one ([ADR-0033](Decision.md)).

## Deliberately out of scope

Not "not done yet" — decided against for this project, so that what *is* built
could be built properly and measured.

| Area | Status | Why |
|---|---|---|
| **Authentication / RBAC** | out of scope | Every request is attributed to one configured tenant. The tenant *plumbing* is complete and enforced in every query and at the tool boundary ([ADR-0003](Decision.md), [ADR-0027](Decision.md)); what is absent is the identity layer that would populate it. Building isolation first and identity later is the safe order — the reverse is how cross-tenant leaks happen. |
| **Observability stack** | out of scope | No OpenTelemetry, Prometheus or dashboards. Structured logs carry the fields that would feed them (tool name, outcome, duration, tenant, request id), and the evaluation harness measures what a dashboard would only display. |
| **Deployment / infrastructure** | out of scope | The container image is a development image: editable install, root user, bind-mounted source. No TLS, no autoscaling, no load testing. This is a system to be read and run locally, not operated. |
| **Caching** | out of scope | Redis is in `docker-compose.yml` and unused. Caching before there is a measured latency problem optimises a guess. |
| **`query_metadata` tool** | deferred | A second agent tool for Atlas's own operational data (queue depth, failed ingestions, last-indexed time). Deliberately not built: the first tool has not yet justified the agent's cost, and adding a second would add code without changing that conclusion. |
| **Full tool-trace panel** | deferred | The console shows the agent's searches and stop reason, not a per-tool-event timeline. The trace is returned in full by the API for anyone who wants it. |
| **Streaming responses** | deferred | `/v1/query` returns one complete response ([ADR-0015](Decision.md)). |


## Project layout

```text
src/atlas/
  agent/        tool registry, search tool, bounded loop, agent answering
  api/          FastAPI app, request/response schemas
  api/static/   the inspection console (3 files, no build step)
  answer/       prompt construction, citation validation, refusal policy
  core/         domain models, deterministic id derivation
  db/           connection pool, migrations, job queue, all SQL
  eval/         dataset format, metrics, runner, agent comparison harness
  ingest/       parsers, normalisation, chunking, pipeline, worker
  providers/    embedding / LLM / rerank protocols, Gemini, fastembed, offline fakes
  cli.py        migrate / ingest / query / eval / models / serve
migrations/     numbered SQL
scripts/        retrieval, answer-model, routing and agent-vs-plain experiments
eval/corpus/    sample knowledge base
eval/datasets/  labelled evaluation queries
eval/baselines/ frozen evidence for every number quoted in this README
tests/          unit tests (no infra) + integration tests (marked)
.github/        CI: lint, unit tests, integration tests against pgvector
Dockerfile      API image (development)
```
