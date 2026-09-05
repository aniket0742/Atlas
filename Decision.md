# Architecture Decision Record

Decisions are recorded here as they are made, and reflect what is **actually
implemented**, not what was planned. Where a decision has not been validated by
measurement yet, that is stated rather than glossed over.

Each entry names what would make us reconsider. A decision with no such trigger
is usually a preference wearing a decision's clothes.

**Status key:** `accepted` — implemented and in use. `provisional` — implemented
but not yet validated by measurement. `deferred` — deliberately not done yet.

---

## ADR-0001: Postgres with pgvector, not a dedicated vector database

**Status:** accepted (Phase 1)

**Problem.** Vector similarity search needs an index. The obvious options are a
purpose-built vector database (Qdrant, Weaviate, Milvus, Pinecone) or a vector
index inside a general-purpose database.

**Alternatives considered.**

- *Qdrant / Weaviate / Milvus.* Purpose-built ANN, richer index tuning, better
  scaling story past ~10^8 vectors. Cost: a second stateful system to run,
  back up and keep consistent with Postgres — because document metadata,
  tenancy, versions and ingestion state all still live in a relational database.
  Every write then spans two systems with no shared transaction.
- *Pinecone / hosted.* Removes the operational cost, adds a paid dependency,
  which this project has ruled out.
- *pgvector in Postgres.* One system, one transaction boundary.

**Decision.** Postgres 16 with pgvector.

**Why.** The decisive factor is transactional consistency, not raw ANN
performance. Writing a document version's chunks and their embeddings must be
atomic with the metadata update — otherwise a crash mid-ingest leaves chunks
that retrieval can find but whose document row says `pending`, or embeddings
orphaned from deleted chunks. With pgvector that is one `BEGIN`/`COMMIT`. With
an external vector store it is a distributed write requiring an outbox or a
reconciliation job, which is real complexity bought to solve a problem this
system does not have at its size.

Secondary: metadata filtering (tenant, source, version) is a join, and joins are
what a relational database is for.

**Trade-offs.** pgvector's HNSW is slower than a tuned dedicated engine at large
scale, and Postgres applies the ANN index *before* filter predicates, which
causes the under-fill problem in ADR-0003. Index build is single-threaded per
index and holds memory proportional to `m` and `ef_construction`.

**Reconsider if.** The corpus exceeds roughly 10^7 chunks per tenant and p95
retrieval latency stops meeting target after tuning `m`, `ef_construction` and
`hnsw.ef_search`; or filtered-recall under selective tenant predicates cannot be
fixed by partial indexes. Both are measurable, so this gets revisited with
numbers rather than by feel.

---

## ADR-0002: No Kafka. The queue is Postgres

**Status:** accepted (Phase 3) — was `deferred`; implemented as described, see
ADR-0022

**Problem.** The original architecture put Kafka between the ingestion API and
the processing workers.

**Alternatives considered.**

- *Kafka.* Partitioned log, consumer groups, replay, high throughput. Costs a
  cluster to operate (KRaft or ZooKeeper), partition/consumer-group design, and
  rebalancing behaviour to understand. Enqueue cannot participate in the
  Postgres transaction that writes the document row.
- *Redis Streams.* Much lighter, consumer groups, at-least-once. Still a second
  system, and durability depends on the persistence configuration.
- *Postgres job table with `SELECT ... FOR UPDATE SKIP LOCKED`.* Transactional
  enqueue in the same transaction as the document write. Queue depth is a
  `COUNT(*)`. DLQ is a status column. No new infrastructure.

**Decision.** Phase 1 ingested synchronously. Phase 3 introduced a Postgres job
table, not Kafka. This is now built and running; ADR-0022 covers the details of
the implementation and what it cost.

**Why.** Kafka earns its complexity at throughput and fan-out this system does
not have: one developer, a corpus measured in thousands of documents, one
consumer type. `SKIP LOCKED` is a well-established pattern that handles
concurrent workers correctly, and it makes enqueue atomic with the document
write — something Kafka structurally cannot do without an outbox table, at which
point the outbox *is* a Postgres queue.

Adding Kafka here would make the architecture diagram more impressive and the
system worse. That trade is explicitly against the goals of this project.

**Trade-offs.** A Postgres queue competes with query traffic for connections and
generates table churn requiring autovacuum attention. It does not give replay of
historical events, and it will not scale to millions of jobs per hour.

**Reconsider if.** Sustained ingestion exceeds roughly 10^3 jobs/second, or a
second independent consumer needs the same event stream (a genuine fan-out), or
event replay becomes a requirement. Redis Streams is the next step before Kafka.

---

## ADR-0003: `tenant_id` on every table from the first migration

**Status:** accepted (Phase 1)

**Problem.** Multi-tenancy was scheduled for Phase 5. Phase 1 has one tenant and
no authentication.

**Alternatives considered.**

- *Add tenancy in Phase 5.* Least work now.
- *Row-level security in Postgres.* Strongest enforcement; requires per-request
  role or session variables and complicates connection pooling.
- *Schema-per-tenant.* Strong isolation, poor at many tenants, painful migrations.
- *`tenant_id` column on every table and in every index, enforced at the
  repository boundary.*

**Decision.** The last one, from migration `0001`. Every repository function
takes a `tenant_id` and every query filters on it. Document, source and chunk
ids are `uuid5` values derived from a string that **includes the tenant**, so two
tenants uploading byte-identical files get different ids by construction.

**Why.** Tenancy is a schema property, not a feature. Retrofitting it means
touching every table, every composite index, every query and every cache key at
once — and the failure mode of missing one is silent cross-tenant data
disclosure, which is the single worst bug this system could have. The cost of
doing it now is one column and one predicate per query. The cost of doing it in
Phase 5 is a migration plus an audit of every call site, with no way to prove
completeness.

Authentication, RBAC and rate limiting are genuinely Phase 5 work and are *not*
being pulled forward. Only the schema is.

**Trade-offs.** Slightly more verbose queries. A composite index on
`(tenant_id, ...)` is larger than one without.

**Known gap.** The HNSW index is global, not per-tenant. pgvector searches the
ANN index and applies the `tenant_id` predicate afterwards, so with many tenants
a selective filter can return fewer than `k` rows even when more matching rows
exist. With one tenant this cannot occur. Phase 5 addresses it — partial indexes
per tenant, or pgvector's iterative scan — and the choice will be made against
measured filtered-recall, not assumed.

**Reconsider if.** Tenant counts or compliance requirements make row-level
security or physical separation necessary. The column stays either way.

---

## ADR-0004: Embeddings in their own table, keyed by model

**Status:** accepted (Phase 1)

**Problem.** Where does a chunk's vector live — a column on `chunks`, or a
separate table?

**Decision.** A separate `chunk_embeddings(chunk_id, model, embedding)` table,
primary-keyed on `(chunk_id, model)`.

**Why.** Section 7 of the specification requires comparing retrieval approaches.
Comparing two embedding models is the most basic such comparison, and with a
column on `chunks` it requires either re-indexing the corpus destructively
(losing the ability to A/B) or adding a nullable column per model. With a
separate table, two models coexist over *identical chunks*, so a comparison
isolates the model rather than confounding it with a re-chunk.

**Trade-offs.** One extra join per retrieval query. pgvector requires a fixed
dimensionality per indexed column, so the table pins `vector(384)`; adding a
model with different dimensions means a new table, not just a new row. That is
recorded here so it is a known constraint rather than a surprise.

**Reconsider if.** Only one embedding model is ever used and the join shows up
as a measured cost — neither is true today.

---

## ADR-0005: Hand-written SQL with psycopg 3, no ORM

**Status:** accepted (Phase 1)

**Problem.** Data access layer.

**Alternatives considered.** SQLAlchemy ORM; SQLAlchemy Core; raw psycopg.

**Decision.** psycopg 3 (async) with hand-written SQL in a repository module.

**Why.** The queries that matter are vector-distance queries using pgvector
operators, and from Phase 2 a hybrid lexical/vector fusion. Neither is
expressible in an ORM without dropping to raw SQL anyway. Adding an ORM would
put a layer of indirection over the one part of the codebase most worth reading,
in exchange for conveniences (identity map, lazy loading, relationship
navigation) this system does not use.

**Trade-offs.** No automatic schema/model consistency, no query composition
helpers, manual parameter binding. Mitigated by keeping all SQL in one module
and taking `tenant_id` explicitly everywhere.

**Reconsider if.** The schema grows enough CRUD surface that hand-written SQL
becomes the bulk of the code rather than the interesting part of it.

---

## ADR-0006: Plain numbered SQL migrations, not Alembic

**Status:** accepted (Phase 1)

**Decision.** Numbered `.sql` files applied in order, tracked in a
`schema_migrations` table with a checksum per applied file.

**Why.** Following from ADR-0005, the schema's interesting content is pgvector
index DDL with tuning parameters (`m`, `ef_construction`) and, in Phase 2, a
generated `tsvector` column with a GIN index. Alembic's autogenerate does not
model those, so they would be hand-written inside migration files regardless —
leaving Alembic contributing a dependency and boilerplate rather than leverage.
Postgres has transactional DDL, so each migration is applied atomically without
framework help.

The checksum check means editing an already-applied migration is refused rather
than silently letting the database and the repository disagree.

**Trade-offs.** No autogenerate, no downgrade path, no branching support. Forward-
only migrations are a deliberate constraint (see the deployments corpus note:
rollback after a migration requires a compensating migration).

**Reconsider if.** Multiple developers start producing concurrent migrations and
need branch merging, or downgrade becomes an operational requirement.

---

## ADR-0007: Local ONNX embeddings, hosted LLM

**Status:** accepted (Phase 1)

**Problem.** Embeddings and generation are separate model choices. A hard project
constraint is no paid API dependency.

**Decision.** Embeddings run locally on CPU via `fastembed`
(`BAAI/bge-small-en-v1.5`, 384 dimensions, 512-token input limit). Generation
goes to Google Gemini's free tier. Reranking in Phase 2 will use a local
cross-encoder from the same library.

**Why local embeddings.** Cost drives retrieval quality work. Phase 2 compares
chunking strategies and retrieval configurations, and every comparison re-embeds
the corpus. If re-embedding costs money and network round-trips, experiments get
run once and conclusions get assumed — exactly the failure the specification
warns against. Local embeddings make re-indexing free and repeatable, and remove
the network from the ingest path. They are also deterministic, so an eval run
from last month is comparable with one from today; hosted embedding models are
versioned and deprecated underneath you.

**Why fastembed rather than sentence-transformers.** ONNX Runtime instead of
PyTorch, which keeps the eventual worker container small, and it exposes the
tokenizer so chunking can budget in real tokens rather than guessing.

**Trade-offs.** A 33M-parameter model is weaker than a large hosted embedding
model; that gap is a measured quantity, not an accepted one — the provider
interface allows swapping and the eval harness reports the delta. Also note
fastembed downloads a **quantised** ONNX build
(`qdrant/bge-small-en-v1.5-onnx-q`), which may differ slightly from the original
FP32 weights; any published number must name the build.

**Measured on this machine (single laptop CPU, not a benchmark).** Embedding
throughput was roughly 350 ms per ~250-token chunk in a batch of ten. That is
slow enough to matter for bulk ingestion and is a Phase 6 optimisation target
(batching, `threads`, possibly a smaller model). Recorded so the Phase 6 claim
has a baseline; not to be quoted as a benchmark.

**Reconsider if.** Retrieval quality measured against the eval set is materially
worse than a hosted embedding model *and* the project constraint on paid
dependencies changes.

---

## ADR-0008: Gemini behind a two-method provider interface

**Status:** accepted (Phase 1)

**Decision.** `LLMProvider` and `EmbeddingProvider` are `Protocol`s. The Gemini
implementation uses `client.models.generate_content` from `google-genai` 2.21.0.

**Why `models.generate_content` and not `client.interactions`.** The installed
SDK exposes both. `generate_content` is the stable, widely-documented surface and
accepts `response_schema`, which the answering path depends on. Verified against
the installed SDK rather than assumed.

**Why an interface at all.** Two concrete reasons, not speculative flexibility:
tests and CI must run with no API key and no network (there is a deterministic
fake for both providers), and the eval harness must be able to hold retrieval
constant while varying the model.

**Deliberately narrow.** The interface has one generation method, and it returns
**structured output only** — `generate_structured(...) -> (parsed, TokenUsage)`.
There is no free-form text method, because requiring a schema is what makes
citation validation possible (ADR-0010). Provider-specific concerns (safety
settings, thinking budgets) are constructor arguments of the concrete class, not
parameters on the interface.

**Free-tier consequences.** HTTP 429 is a normal operating condition, not an
exception, so retry with exponential backoff and full jitter is implemented in
Phase 1 rather than deferred to the reliability phase. 4xx errors other than 429
fail fast — retrying them only burns quota. Free-tier model availability is not
reliably documented per model, so `atlas models` lists what the key can actually
reach instead of the project asserting which models are free.

**Reconsider if.** Rate limits make development impractical, at which point a
local model via Ollama is the fallback that preserves the no-paid-dependency
constraint.

---

## ADR-0009: Structure-aware chunking with exact character offsets

**Status:** provisional (Phase 1) — not yet validated against alternatives

**Problem.** Where to cut documents.

**Alternatives considered.** Fixed-size character windows; fixed token windows
with overlap; recursive character splitting; structure-aware splitting.

**Decision.** Split on structure (markdown headings, then blank-line-delimited
blocks), then pack blocks into token-budgeted windows measured with the embedding
model's own tokenizer. A chunk never spans a section boundary. Oversized blocks
split on sentence boundaries, and only then hard-split.

Two invariants the rest of the system depends on:

1. `chunk.text == document.content[chunk.char_start:chunk.char_end]` exactly.
   Citations resolve by slicing, so any deviation surfaces as a wrong quote.
2. `chunk.token_count <= embedding model max_tokens`, so nothing is silently
   truncated at embed time.

**Why.** Fixed-size windows cut mid-sentence and mid-table, producing chunks that
cannot be read alone. Most of the target corpus (markdown, docs, READMEs) carries
explicit structure, so respecting it is cheap. The no-spanning rule exists
because a chunk crossing a section boundary gets labelled with only one section's
heading path, and a citation that names the wrong section is worse than one with
no section at all — this was an actual defect found by the chunking tests during
Phase 1, not a hypothetical.

Undersized chunks are **merged backwards within their section, never dropped**.
Dropping is tempting and wrong: a short section ("## Contact" plus an address) is
small but is exactly the kind of thing users ask about, and a dropped chunk is
unretrievable content with no signal that it is missing.

**Trade-offs.** More complex than a character splitter. Structure detection is
markdown-specific; PDFs and plain text degrade to blank-line blocks. Sentence
splitting is regex-based and will mis-split on abbreviations.

**Not yet justified.** The parameters (320 target tokens, 64 overlap) are
starting values, not tuned ones. Whether this beats a naive fixed-size splitter
on this corpus is an open question the eval harness exists to answer, and the
status stays `provisional` until it does.

**Reconsider if.** Measurement shows no advantage over fixed-size chunking, or
shows parent/child retrieval materially better.

---

## ADR-0010: Groundedness is enforced in code, not requested in the prompt

**Status:** accepted (Phase 1)

**Problem.** Preventing fabricated answers and fabricated sources.

**Decision.** The model returns a JSON object matching a schema (`answer`,
`citations[]`, `sufficient_evidence`). Its output is treated as a *proposal* and
validated:

1. **Citation resolution.** A cited id must be one of the server-generated ids
   supplied for that specific question. Ids naming anything else are discarded.
2. **Quote verification.** The quote must appear verbatim (whitespace- and
   case-insensitively) in the cited chunk. A citation that resolves but whose
   quote does not match is **kept and flagged**, not dropped — it is usually
   paraphrase, occasionally fabrication, and both are worth counting.
3. **Refusal downgrade.** An answer claiming sufficient evidence but producing no
   resolvable citation is converted into a refusal.

Retrieval that returns nothing above the similarity floor never reaches the model
at all.

**Why.** A prompt asking the model to cite its sources is a request. These are
guarantees. The distinction matters because the failure being defended against
is the model being confidently wrong, and asking a confidently-wrong model to
self-report is not a control.

**Prompt injection.** Retrieved documents are attacker-influenced: anyone who can
get a document into the corpus can put instructions in it. There is no complete
defence and none is claimed. What is implemented: evidence is delimited in tagged
blocks declared as untrusted data; citation ids are server-generated per request,
so injected text cannot mint a source; and in Phase 1 the model has no tools, so
the worst case is a wrong answer rather than an action. When tools arrive in
Phase 4, authorisation will be derived from the caller's identity and never from
retrieved text.

**Residual risk, stated plainly.** A document asserting "the refund window is 900
days" will be faithfully reported as saying so. The property enforced is
faithfulness to sources. Source trustworthiness is a different problem and is out
of scope.

**Trade-offs.** Structured output constrains phrasing and costs some fluency.
Strict validation can downgrade a correct answer to a refusal when the model
cites sloppily — a deliberate bias toward false refusals over false confidence,
and the eval harness measures the rate.

**Reconsider if.** Measured incorrect-refusal rate on answerable questions gets
high enough to outweigh the groundedness benefit.

---

## ADR-0011: Eval labels anchor to documents and snippets, never to chunk ids

**Status:** accepted (Phase 1)

**Problem.** What does a relevance label point at?

**Decision.** A label names a document by its stable external id plus, optionally,
a text snippet that must appear in the retrieved chunk.

**Why.** Labelling chunk ids is the obvious choice and it is a trap. Chunk ids
derive from `(document, version, ordinal)`, so any change to chunking, overlap
or token budget renumbers every chunk and silently invalidates the whole dataset.
Since the harness exists precisely to compare chunking strategies, chunk-id
labels would be destroyed by the first experiment they were built to evaluate.
Document-plus-snippet labels survive re-chunking, re-embedding and re-ingestion,
and stay readable to a human reviewer.

**Trade-offs.** Coarser than chunk-level labelling: a label is satisfied by any
chunk from the right document containing the snippet. Snippet matching is
substring-based, so a label breaks if the source document is edited — the test
suite asserts every shipped label still matches its corpus file.

**Reconsider if.** Chunking stabilises permanently and finer-grained labels are
needed to discriminate between close configurations.

---

## ADR-0012: Phase 1 ingests synchronously, and says so

**Status:** superseded (Phase 3) by ADR-0022 and ADR-0023. Retained because the
reasoning for starting synchronously, and the two properties that made the move
cheap, are the argument for how Phase 3 went.

**Decision.** `POST /v1/documents` runs the full pipeline in the request and
returns 201 once the document is queryable.

**Why.** The pipeline's behaviour is still being established; running it inline
keeps it debuggable and makes failures immediate rather than buried in a worker
log. The two properties that make the Phase 3 move cheap — deterministic ids for
idempotency, and a content-hash short-circuit — are already implemented, so
moving the call site behind a queue does not change the pipeline itself.

**Trade-offs.** A large PDF blocks its request. The embedding step is pushed to a
thread so it does not block the event loop for *other* requests, but the
uploading client waits. There is no retry on failure; the document is marked
`failed` with its error and must be re-submitted.

**Reconsider.** Phase 3, by design. The endpoint becomes an enqueue returning 202
and a job id.

---

## ADR-0013: The similarity floor, calibrated at 0.60

**Status:** accepted (Phase 1) — was `provisional`; calibrated by experiment E2

**Problem.** Retrieval always returns its top-k. Without a floor, a question with
no answer in the corpus still yields k confident-looking chunks and the model is
asked to answer from irrelevant evidence.

**Decision.** A cosine-similarity floor of **0.60**. Below it, Atlas refuses
without calling the model at all.

**How the number was chosen.** The floor acts on retrieval scores, so its effect
is computable with no LLM calls. `scripts/calibrate_floor.py` retrieves every
eval query once and sweeps candidate thresholds, measuring three things: how many
unanswerable questions get refused before a model call, how many answerable
questions lose *all* their evidence, and how many lose their *relevant* evidence.

Measured on the 19-query smoke set with `bge-small-en-v1.5`:

| | range |
|---|---|
| Answerable — score of best **relevant** chunk | 0.669 – 0.879 |
| Unanswerable — score of best chunk overall | 0.569 – 0.639 |

The distributions do not overlap. A floor anywhere in 0.64–0.66 would refuse all
three unanswerable questions with zero false refusals.

**Why 0.60 and not 0.65.** The separating gap is 0.03 wide and rests on **three**
unanswerable queries. Placing the threshold inside that gap would be fitting a
parameter to three data points, and it would not survive contact with a real
corpus. 0.60 sits 0.069 below the weakest genuine answer — enough margin that
normal variation does not start refusing real questions — and still eliminates
one unanswerable query before any token is spent.

This is deliberately a *conservative* setting, because the floor is not the only
control. It is a cheap pre-filter; the model's own `sufficient_evidence`
judgement plus citation validation (ADR-0010) handle the rest. Verified live: the
SAML question scores 0.639, passes the floor, and is correctly refused by the
model anyway.

**Trade-offs.** At 0.60 only one of three unanswerable questions is caught before
the model call, so the other two cost tokens. Raising the floor would save those
tokens and increase the risk of refusing real questions. That trade is now
measurable in both directions rather than a matter of taste.

**Note.** Retrieval metrics in the eval harness are computed *before* the floor is
applied. The floor is an answering policy; folding it into retrieval scoring
would make a threshold change look like a retrieval regression.

**Reconsider if.** The eval set grows (the current separation is very likely an
artefact of a small synthetic corpus — on real data these distributions should be
expected to overlap), the embedding model changes (absolute cosine values are not
comparable across models, so this number is meaningless for any other model), or
the measured false-refusal rate rises.

---

## ADR-0014: nDCG counts gain once per label

**Status:** accepted (Phase 1)

**Problem.** The first baseline run reported **nDCG@8 = 1.0164**. nDCG is
normalised and cannot exceed 1.0, so the metric was wrong.

**Cause.** DCG summed gain at every rank holding a relevant chunk, while IDCG was
computed from the number of *labels*. Because chunks overlap, several chunks can
carry the same fact and satisfy the same label, so DCG could exceed IDCG.

**Decision.** nDCG takes one position per distinct label — the earliest rank at
which that label was satisfied. Precision still counts every relevant chunk in
the top k, because precision is a statement about the retrieved list rather than
about label coverage. MRR is unaffected.

**Why it matters beyond the arithmetic.** A score above 1.0 is visibly wrong and
was caught immediately. The same bug at a smaller magnitude would have silently
inflated every retrieval number, and Phase 2 would have compared hybrid retrieval
against a corrupted baseline. That is the argument for building the harness in
Phase 1 and *running* it, rather than trusting it because the code looks right.

**Guard.** A regression test asserts nDCG stays within [0, 1] for inputs where
several chunks share a label.

---

## ADR-0015: A static inspection console, not a frontend framework

**Status:** accepted (Phase 1)

**Problem.** The system needed to be demonstrable and inspectable through a
browser rather than only through CLI output, without frontend work displacing
the backend and retrieval work that is the point of the project.

**Alternatives considered.**

- *React or Svelte with Vite.* Component reuse, familiar tooling, and a story
  interviewers recognise. Costs a `package.json`, a lockfile, `node_modules`, a
  build step, a bundler configuration, a second container in compose, and a
  second dependency-update surface — to render a form and three lists.
- *A separate static container behind nginx.* Keeps concerns separate; adds a
  service, a cross-origin boundary and therefore CORS configuration.
- *Plain HTML/CSS/JS served by the existing FastAPI app.* No build step, no
  tooling, same origin.

**Decision.** Three static files (`index.html`, `app.css`, `app.js`) served by
FastAPI: `/` returns the page, `/static` is a mount.

**Why.** Same-origin means no CORS, no second port, and no auth story to invent
twice when Phase 5 arrives. More usefully, it makes the console a plain client of
the documented API — **it cannot display anything the API does not already
return**, which keeps the API honest instead of letting a capable frontend paper
over gaps. Adding `document_external_id` to the evidence response was a gap the
console surfaced immediately.

The whole console is ~350 lines. A framework's advantages start where this ends.

**Trade-offs.** No components, no reactive state, manual DOM construction. It
will get unwieldy somewhere past ~500 lines; that is the signal to introduce a
framework, with a reason to point at rather than a default to inherit.

**Security note.** Document titles, heading paths and chunk text all originate in
uploaded files, so they are untrusted. The console builds every node with
`textContent` and never assigns response data to `innerHTML`, which makes stored
XSS structurally impossible rather than a review item.

**Not included.** Streaming. `/v1/query` returns one complete JSON response, so
the console shows a spinner and then the whole answer. Server-Sent Events are a
different endpoint shape and are deferred to a later phase; a fake typewriter
animation over an already-complete response would be a lie about how the system
works.

**Reconsider if.** The console outgrows one file per concern, or a genuinely
interactive view (a retrieval-comparison workbench in Phase 2) needs real state
management.

---

## ADR-0016: The API runs in Docker Compose

**Status:** accepted (Phase 1)

**Problem.** Postgres and Redis were containerised while the API ran on the host,
so "run Atlas" meant starting containers *and* a host process with a matching
Python environment. The console needed to be servable as part of the local
environment.

**Decision.** A `Dockerfile` for the API and an `api` service in
`docker-compose.yml`. `docker compose up` now starts everything, applies
migrations, and serves the console at `http://localhost:8000`.

**Four things this forced, each worth stating.**

1. **`localhost` is wrong inside a container.** `ATLAS_DATABASE_URL` is overridden
   in the compose `environment:` block to `postgres:5432` — the service name on
   the compose network. `environment` takes precedence over `env_file`, so the
   host-oriented value in `.env` stays correct for CLI use on the host.
2. **The model cache is a named volume.** `ATLAS_MODEL_CACHE_DIR=/models` backed
   by `atlas_models`, so the ~67MB ONNX model downloads once and survives image
   rebuilds. Baking it into the image would add that to every layer.
3. **The secret never enters the image.** `.env` is in `.dockerignore` and the key
   arrives as an environment variable at run time. Verified: `.env` is absent from
   the image and the key does not appear in image history.
4. **Editable install.** The migration runner locates `migrations/` relative to
   the package, which resolves for a source layout but not for a package copied
   into `site-packages`. `pip install -e .` with the source at `/app` keeps that
   working without adding a configuration knob. A deployment image in Phase 7
   will need to address this properly.

`env_file` is marked `required: false` so a fresh clone can `docker compose up`
before creating a `.env`; the app then starts and fails on the first query, which
is a clearer error than compose refusing to start.

`./src` is bind-mounted over the editable install so code and console edits take
effect on restart without a rebuild. That is a local-development convenience and
is explicitly not how a deployment image should behave.

**Trade-offs.** Image build takes a few minutes, mostly `onnxruntime`. Compose now
rebuilds on dependency changes. No Kubernetes, no registry, no orchestration —
Phase 7 addresses deployment.

**Reconsider.** Phase 7, which needs a non-editable install, a pinned base image
digest, and a non-root user.

---

## ADR-0017: PostgreSQL full-text search, not BM25

**Status:** accepted (Phase 2)

**Problem.** Dense retrieval is weak on rare literal tokens — error codes,
environment variable names, header names. Measured on the Phase 2 baseline:
Recall@1 was 0.615 for `identifier` queries against 0.871 for `paraphrase`.

**Alternatives considered.**

- *Okapi BM25 via an extension* (`pg_search` / ParadeDB). Real BM25 with term
  saturation and length normalisation. Requires a non-standard Postgres image
  and pins us to that extension's release cycle.
- *BM25 hand-implemented in SQL* over a term-frequency table. Accurate, and a
  meaningful amount of index maintenance to own.
- *An in-process library* such as `rank_bm25`. Accurate scoring, but it holds the
  corpus in memory and cannot compose with the tenant predicate inside the
  query, which is exactly the property ADR-0003 exists to protect.
- *PostgreSQL full-text search.* Built in, composes with existing filters, one
  generated column and one GIN index.

**Decision.** PostgreSQL FTS: a `text_search` tsvector generated column with a
GIN index, ranked by `ts_rank_cd`.

**Naming, deliberately.** This is **not BM25** and is not described as such
anywhere in the codebase. `ts_rank_cd` is a coverage-density ranking: it rewards
the count and proximity of matching lexemes but implements neither BM25's term
saturation nor its document-length normalisation. The specification asked for
"BM25 or equivalent lexical retrieval", and calling FTS "BM25" is a claim the
code would not support.

**Two implementation details that decide whether it works at all.**

1. **OR, not AND.** `plainto_tsquery` joins lexemes with `&`, so a twelve-word
   question requires all twelve to appear in one chunk and reliably matches
   nothing. Rewriting `&` to `|` makes it "any of these lexemes" and lets
   ranking, rather than the WHERE clause, decide quality.
2. **The generated column must use the two-argument `to_tsvector`.** The
   one-argument form depends on `default_text_search_config` and is only
   STABLE, which a generated column rejects. Pinning the config is required,
   and it also stops the index changing meaning if a database setting changes.

**Trade-offs.** English-only stemming. No BM25-quality ranking. Unlike the HNSW
index, a GIN index composes normally with the tenant predicate via a bitmap
scan, so the ADR-0003 under-fill problem does not apply here.

**Reconsider if.** Measurement shows lexical ranking quality is the limiting
factor. It currently is not — see ADR-0018, where lexical retrieval measured
*worse* than dense.

---

## ADR-0018: Hybrid retrieval implemented, measured, and not adopted

**Status:** implemented but **not default** (Phase 2)

**Problem.** Does combining dense and lexical retrieval improve results?

**Decision on fusion method.** Reciprocal Rank Fusion. Cosine similarity sits in
a narrow band around 0.6-0.9 while `ts_rank_cd` is bounded to [0, 1) and
distributed quite differently; the two are not comparable, and per-query min-max
normalisation would make the top hit of every query score 1.0 regardless of
whether it is any good — destroying the signal that a query has no good match at
all. RRF combines ranks, which are on the same scale by construction. The cost is
that score magnitude is discarded: a dense match at 0.95 and one at 0.62
contribute identically when they hold the same rank.

**Decision on adoption: dense remains the default.**

Measured on 100 answerable queries, paired bootstrap against the dense baseline:

| configuration | Recall@1 | nDCG@1 | Recall@8 | nDCG@8 |
|---|---|---|---|---|
| dense (baseline) | 0.780 | 0.800 | 0.980 | 0.895 |
| lexical | 0.620 | 0.640 | 0.955 | 0.805 |
| hybrid | 0.720 | 0.740 | 0.990 | 0.881 |

- **Lexical alone is significantly worse**: Recall@1 −0.160, CI [−0.250, −0.070].
- **Hybrid shows no measured difference** from dense at either depth: Recall@1
  −0.060 (CI [−0.130, +0.010]), nDCG@8 −0.014 (CI [−0.046, +0.018]).

Hybrid is therefore implemented and selectable but is not the default. Making it
the default would be adding a subsystem for its own sake, which is the specific
failure this project set out to avoid.

**What the aggregate hides, and why the code stays.** Per-kind Recall@1 shows
hybrid doing exactly what was predicted, on a slice too small to move the total:

| query kind | n | dense | lexical | hybrid |
|---|---|---|---|---|
| identifier | 13 | 0.615 | 0.538 | **0.769** |
| conceptual | 13 | 0.615 | **0.846** | 0.692 |
| lookup | 36 | **0.861** | 0.667 | 0.750 |
| paraphrase | 31 | **0.871** | 0.548 | 0.742 |

Hybrid helps identifier queries substantially and hurts paraphrase and lookup
queries, and this corpus is about two thirds paraphrase-or-lookup. On a corpus
weighted towards identifiers the conclusion could invert, which is why the mode
is kept rather than deleted.

**Reconsider if.** A deployment's query mix is identifier-heavy, or a weighted
fusion — tuning the dense and lexical contributions rather than treating them
equally — is measured and beats plain RRF.

---

## ADR-0019: The similarity floor is a query-level gate, not a per-chunk filter

**Status:** accepted (Phase 2), supersedes the mechanism in ADR-0013

**Problem.** In Phase 1 the floor filtered individual chunks by cosine
similarity. Fusion breaks that: an RRF score is a sum of reciprocal ranks and a
reranker score is an unnormalised logit. A threshold calibrated on cosine has no
meaning against either, and applying one anyway would silently change refusal
behaviour while every retrieval metric continued to look correct.

**Decision.** The floor is evaluated on the **dense candidates, before fusion**,
as a single query-level decision: if no dense candidate reaches the floor, the
query is treated as having no evidence and answering refuses. Otherwise the final
ranking is returned untouched.

**Why this is better than a patch.** It matches what the floor was always for —
deciding whether the corpus can answer at all, which is a property of the query
rather than of each chunk. It is also the only formulation that survives a change
to the final ranking method, including ones not yet built.

In `lexical` mode there is no dense score and therefore no gate; that mode is for
measurement, not for serving, and the API documents it as such.

**Value.** 0.55, interim and explicitly **not a validated optimum**. The 0.60
calibrated in Phase 1 stopped separating when the corpus grew from 5 to 33
documents: 10 of 12 unanswerable queries now score above it, and the answerable
and unanswerable distributions overlap outright. With 33 documents there is
nearly always something topically adjacent. The floor is a crash barrier for
pathological queries; the model's `sufficient_evidence` judgement and citation
validation (ADR-0010) are the real controls.

**Reconsider.** When the eval set grows further, or if measured refusal
behaviour degrades.

---

## ADR-0020: Cross-encoder reranking, adopted

**Status:** accepted, on by default (Phase 2)

**Problem.** Both retrieval methods score a query and a passage independently and
then compare the results, so neither ever sees the pair together. A cross-encoder
reads both in one forward pass and can judge whether a passage answers *this*
question rather than whether it is about the same topic.

**Decision.** `Xenova/ms-marco-MiniLM-L-6-v2` (~80MB) via fastembed, reranking
the top 30 first-stage candidates. Local and free, consistent with ADR-0007.

**Measured**, paired bootstrap against the dense baseline:

| configuration | Recall@1 | nDCG@8 | retrieval p50 |
|---|---|---|---|
| dense | 0.780 | 0.895 | 77ms |
| dense + rerank | **0.850** | **0.939** | 750ms |

nDCG@8 **+0.044, CI [+0.009, +0.081] — significant.** Recall@1 +0.070 with
CI [+0.000, +0.150], which does not clear the bar on its own.

Per-kind Recall@1 shows the gain is broad rather than concentrated in one slice:
identifier 0.615 → 0.923, conceptual 0.615 → 0.846, lookup 0.861 → 0.917.

**The cost, stated plainly.** Retrieval goes from 77ms to roughly 750ms, about
10x. In context that is +680ms on a request whose generation step already takes
~2.8s, so around +23% end to end rather than 10x. It is enabled by default
because answer quality is this system's stated priority and the gain is measured;
`ATLAS_RERANK_ENABLED=false` returns to 77ms retrieval at lower quality. Cost is
linear in `rerank_candidates`, which is the dial to turn before disabling it
outright.

**A negative result worth recording (E7).** Reranking makes the lexical half
redundant. `dense+rerank` and `hybrid+rerank` are indistinguishable — Recall@1
differs by exactly 0.0000, and at k=8 hybrid is marginally *worse* (−0.010,
CI [−0.030, +0.000]). A cross-encoder that reads the pair directly subsumes what
lexical matching was contributing, so running both is paying twice for one
effect. This is why the shipped default is dense + rerank rather than
hybrid + rerank.

**Reconsider if.** Latency becomes the binding constraint, or a larger reranker
(`BAAI/bge-reranker-base`, roughly 13x the size) is measured to justify its cost.

---

## ADR-0021: Paired bootstrap, replacing comparison of independent intervals

**Status:** accepted (Phase 2), corrects the rule registered in Step 4

**Problem.** The decision rule registered before running the Phase 2 experiments
was: adopt a configuration only if it beats the incumbent with **non-overlapping
95% confidence intervals**. That rule is statistically wrong, and it was wrong
before any number existed.

Configurations are evaluated on the *same* queries, so the comparison is paired.
Independent intervals ignore the pairing and are dominated by variance *between
queries* — some questions are simply harder — rather than by the difference
between configurations. Two configurations can differ on nearly every query and
still produce comfortably overlapping intervals.

**Decision.** Comparisons use a paired bootstrap over per-query differences. The
interval is on the difference itself, so "the interval excludes zero" is the
statement that the configurations actually differ.

**Changing an analysis method after seeing results is how false positives get
manufactured**, so three things were done to keep this honest:

1. The change was made for a stated methodological reason, not because a result
   was disappointing.
2. It was applied symmetrically to every configuration. Its first effect was to
   make a **negative** result significant: lexical retrieval, which the unpaired
   test scored as "overlapping", is significantly *worse* than dense
   (CI [−0.250, −0.070]).
3. Both tests are still printed side by side, so the change in method is visible
   in the output rather than quietly swapped in.

**What this does not fix.** The sampling limitation is untouched: 100 queries
over one synthetic corpus labelled by one person. This removes a statistical
error, not the generalisation problem.

**Reconsider if.** The eval set grows enough that a permutation test, or a
correction for multiple comparisons, becomes worthwhile — the current sweep tests
several configurations against one baseline without any such correction, which is
a real if minor weakness.

---

## ADR-0022: The ingestion queue is a Postgres table

**Status:** accepted (Phase 3) — realises the decision deferred in ADR-0002

**Problem.** Ingestion ran inside the HTTP request. A large PDF blocked its
caller for the duration, a failure had no retry, and query traffic competed with
embedding for the same process.

**Decision.** An `ingest_jobs` table claimed with
`SELECT ... FOR UPDATE SKIP LOCKED`, drained by a separate worker process. No
Kafka, no Redis Streams — the reasoning is in ADR-0002 and has not changed.

The property that justifies it, concretely: `POST /v1/documents` creates the
source and the job **in one transaction**. A broker cannot participate in that
transaction, so it needs an outbox table to avoid "job queued, source missing"
after a crash — and an outbox table *is* a Postgres queue with an extra hop
behind it.

### Where the uploaded bytes live

In the job row, as `bytea`. Postgres TOASTs and compresses anything past ~2KB
and stores it out of line, so a 20MB upload does not sit in the main heap.

The alternative — write to object storage, reference a key — introduces a second
place the payload can disagree with the job after a crash, and adds
infrastructure this system does not otherwise need.

The payload is cleared **when a job succeeds** and **retained when a job dies**.
That asymmetry is deliberate: a dead-letter queue whose entries cannot be
replayed is not a dead-letter queue. Its whole purpose is to fix the cause and
requeue, which needs the bytes. Storage is bounded by how many jobs are dead, and
if that number is large enough to matter it is itself the signal.

### Attempts are counted at claim time, not at completion

A job that crashes its worker never reaches a completion handler. If attempts
were counted there, a document that reliably kills workers would cycle forever:
claim, crash, lease expires, requeue, claim. Counting at claim means such a job
exhausts its budget and reaches the dead-letter state, which is what an operator
needs to see. There is a test for exactly this.

### Leases, not heartbeats

A claimed job carries `locked_at`; a reaper returns rows whose lease has expired
to `pending`. Heartbeating would detect a dead worker faster, at the cost of a
write per job per interval and a liveness protocol to get wrong. A 300-second
lease means a crash costs up to five minutes of delay on one document, which is
an acceptable trade at this scale — and the lease must exceed the slowest
plausible document, because a worker still legitimately embedding a 200-page PDF
must not have its job stolen.

### Three transactions per job

Claim commits immediately; the work runs holding no transaction; completion is a
third. Holding the claim transaction across the work would keep a row lock for
the tens of seconds a large document takes to embed, block the reaper, and pin a
connection from a small pool. The lease exists precisely so the claim can commit
and the work proceed unlocked.

### Duplicate suppression

A partial unique index over `pending` rows means re-uploading a document while an
earlier upload is still queued **replaces** that job rather than adding a second.
A job already `running` cannot be superseded — a worker holds it — so that case
falls through to the pipeline's content-hash short-circuit, which makes the
duplicate nearly free. Suppression is an optimisation; correctness under
duplicate delivery comes from the deterministic ids built in Phase 1.

**Trade-offs.** The queue competes with query traffic for connections and
generates table churn that autovacuum must keep up with. It does not offer replay
of historical events. Polling costs up to one poll interval of latency before a
job starts; `LISTEN/NOTIFY` would remove that but needs a dedicated connection
per worker and a fallback poll anyway, and one second is not the bottleneck when
ingesting a document takes far longer.

**Reconsider if.** Sustained ingestion exceeds roughly 10^3 jobs/second, a second
independent consumer needs the same stream, or payload sizes make in-row storage
untenable. Redis Streams remains the next step before Kafka.

---

## ADR-0023: `POST /v1/documents` returns 202, and the break is not versioned

**Status:** accepted (Phase 3)

**Problem.** The endpoint returned 201 once a document was queryable. With
ingestion behind a queue it can only return "accepted"; the document is not
searchable when the response is written.

**Alternatives considered.**

- *Keep 201 and block until the job finishes.* Preserves the contract and
  discards the entire benefit of the queue.
- *Add `/v2/documents` and keep `/v1` synchronous.* Two ingestion paths to
  maintain and test, one of which exists for callers that do not exist.
- *A `?wait=true` parameter.* Same problem in a smaller package, and it makes
  request duration unbounded and dependent on queue depth.
- *Change `/v1` and accept the break.*

**Decision.** `POST /v1/documents` now returns **202 Accepted** with a `job_id`
and a `document_id`. No `/v2`.

**Why not version it.** There are no external consumers. A compatibility shim
would be cost paid now for a beneficiary that does not exist, and it would leave
two ingestion paths in the codebase where one of them is dead weight that still
has to be tested. The break is documented here, in the endpoint's docstring, and
in the README.

**The `document_id` is returned despite no work having happened.** It is derived
from `(tenant, source, external_id)` via uuid5 rather than assigned by the
worker, so it is knowable at enqueue time. A caller can hold it, poll
`GET /v1/documents/{id}`, and store the reference before indexing completes.
This is a direct payoff from the Phase 1 decision to derive ids from identity
rather than generate them.

**A behavioural consequence worth stating.** Document-type validation moved into
the worker, because parsing is part of the deferred work. The endpoint therefore
accepts files it previously rejected with 415 or 422; those now surface as a
dead-lettered job carrying the error. The failure is not hidden — it moved from
the response to the job — but a client that treated 202 as success will report
success for a document that never indexes. The console polls the job rather than
trusting the 202, and any other client should too.

**Trade-offs.** Clients must poll. There is no push notification when a job
finishes; webhooks are a plausible later addition and are not built.

**Reconsider if.** The API gains external consumers, at which point versioning
stops being ceremony.

---

## ADR-0024: Two model roles, and `gemini-3.5-flash-lite` for the answer

**Status:** accepted (Phase 4)

**Problem.** One model served every purpose. Phase 4 introduces an agent loop
whose tool-routing turns are high-volume, cheap, and latency-sensitive, while the
final grounded answer is low-volume and is where quality is visible. Those are
different jobs with different requirements.

**Decision.** Two roles behind the existing provider abstraction:

| role | model | why |
|---|---|---|
| final answer | `gemini-3.5-flash-lite` | no measurably better option; see below |
| agent / tool routing | `gemini-3.1-flash-lite` | equal routing quality, fastest, cheapest |

`get_llm()` and `get_agent_llm()` build both from one `_build_llm(settings, model)`.
No new provider type, no second abstraction.

### One API key covers both

Free-tier quota is scoped per project **per model**
(`GenerateRequestsPerMinutePerProjectPerModel`). Verified by exhausting one
model until it returned 429 and finding the other still served on the same key.
Splitting roles therefore adds headroom instead of dividing one budget. No second
key or project is required.

### Why `gemini-3.5-flash` was dropped

It was the default through Phases 1–3 and is **strictly dominated**: $1.50/$9.00
per 1M tokens against $0.75/$3.75 for `gemini-3.8-flash`, a model the vendor
describes as newer and more capable. It also carries a 20 requests/**day**
free-tier cap, which cannot support a single 112-query evaluation run. It had
never been re-examined after being chosen in Phase 1.

### Why the answer model is the cheap one

Measured on 112 queries with retrieval held constant (frozen in
`eval/baselines/answer-models/`):

| model | unverified quotes | p50 | $/1k queries |
|---|---|---|---|
| gemini-3.1-flash-lite | 14 | 2,254 ms | $0.54 |
| **gemini-3.5-flash-lite** | 3 | 2,507 ms | $0.71 |
| gemini-3.7-flash | 2 | 3,016 ms | $2.44 |
| gemini-3.8-flash | 1 | 3,968 ms | $2.83 |

All four scored 12/12 on refusing unanswerable questions, 0 wrongly refused, and
100/100 on citation coverage. The guarantees from ADR-0010 hold regardless of
model; only verbatim-quoting fidelity varies.

**The ranking is not readable from one run.** `gemini-3.5-flash-lite` produced
5, then 8, then 3 unverified quotes across three runs of identical
configuration — a spread wider than most gaps between models. Paired bootstrap
on per-query counts (ADR-0021), against the selected default:

- `gemini-3.1-flash-lite` +0.098, CI [+0.045, +0.161] — **significantly worse**
- `gemini-3.7-flash` −0.009, CI [−0.027, +0.000] — no measured difference
- `gemini-3.8-flash` −0.018, CI [−0.045, +0.000] — no measured difference

So the more expensive models are **not measurably better** at the one thing that
separates them, while costing 3.4–4× more and running 20–58% slower. Choosing the
cheaper model here is the evidence-backed decision, not the budget one — and had
the first single-run table been taken at face value, the opposite conclusion
would have been drawn from noise.

### What this cost to measure honestly

Two corrections were needed before the numbers meant anything.

**Cost was computed from a blended 85/15 input:output split**, because the eval
summary carried only a total token count. That was wrong by 15–35%, understating
`gemini-3.8-flash` most. The cause is visible in the data: `3.7` and `3.8` emit
3–4× more output tokens than the lite models on identical input, because they
produce thinking tokens — billed as output, invisible in the response text. The
runner now records input and output separately, folding thinking into output.

**Input tokens are identical (141,584) across all four models**, which is the
check that retrieval was genuinely held constant rather than assumed to be.

**Reconsider if.** The eval set grows enough to resolve differences the current
n cannot; quoting fidelity degrades in production; or promotional Flash pricing
expires on 2027-01-01, which doubles the cost of `3.6`/`3.7`/`3.8` and widens the
gap further in favour of the lite models.

---

## ADR-0025: `gemini-3.1-flash-lite` routes the agent's tool calls

**Status:** accepted (Phase 4)

**Problem.** The agent role decides when to call a tool and what to search for.
It is high-volume, latency-sensitive and cheap per call — the opposite profile to
writing the final cited answer, which ADR-0024 settled separately.

**Decision.** `gemini-3.1-flash-lite`, measured against
`gemini-3.5-flash-lite` and `gemini-3.7-flash` on 16 routing cases with real
retrieval. Frozen in `eval/baselines/agent-routing/`.

| model | selection accuracy | unnecessary searches | multi-doc coverage | latency | $/1k |
|---|---|---|---|---|---|
| **gemini-3.1-flash-lite** | 16/16 | 0 | **4/4** | **2,568 ms** | **$0.36** |
| gemini-3.5-flash-lite | 16/16 | 0 | 3/4 | 2,724 ms | $0.58 |
| gemini-3.7-flash | 16/16 | 0 | 3/4 | 4,684 ms | $1.83 |

### The first benchmark could not justify any choice

An initial 8-case set scored **8/8 for every candidate**, including models
costing 15x more. That is not evidence of equivalence; it is evidence the
benchmark was too easy to measure anything. Choosing on it would have been
choosing on noise-free ties.

It was rebuilt around the failure modes routing actually has — vocabulary
mismatch between question and corpus, questions that only look like lookups,
answers contained in the question, cross-domain comparisons, and terse input.

**Selection accuracy still saturated at 16/16.** Routing over a single tool is
genuinely easy, and that is recorded as the finding rather than papered over.

### What did separate them

Multi-document coverage. `gemini-3.1-flash-lite` was the only model to issue two
genuinely different queries for a comparison spanning two documents and reach
both; the others collapsed it into one query and reached one. That is a real
capability difference on exactly the query class Phase 2 measured as hardest
(multi-doc Recall@1 stuck at 0.400 for every retrieval configuration).

Persistence cut the other way. On an unanswerable question `gemini-3.7-flash`
searched four times over 10.5 s before giving up, against one or two searches for
the lite models, reaching the same correct conclusion. More persistence bought
nothing but latency.

**Reconsider if.** The Phase 4 agent evaluation — 112 queries against the real
tool set, rather than 16 synthetic cases against one tool — separates these
models on selection quality. That evaluation is the proper instrument; this
benchmark exists to make a defensible interim choice, and it is one environment
variable to change.

---

## ADR-0026: Tool guarantees live in the registry, not in the tools

**Status:** accepted (Phase 4)

**Problem.** Every tool needs argument validation, a timeout, an authorization
check and structured logging. The obvious placement is inside each tool.

**Decision.** A tool implements exactly one method, `execute`. Validation,
timeouts, permission checks and logging all happen in `ToolRegistry.invoke`,
around the call.

**Why.** If validation is the tool's job, the third tool someone adds forgets it,
and the failure is a malformed query against a database rather than a clean
rejection. Placing the guard in the registry means a tool **cannot opt out of it
by being written carelessly**. The tool author writes the interesting part; the
framework enforces the boring parts uniformly.

The same argument applies to timeouts. Per-tool budgets are declared as class
attributes — a local database query and a remote API call do not deserve the
same allowance — but the enforcement is `asyncio.wait_for` in the registry, so a
tool that forgets to bound its own work is still bounded.

### Failures are returned, not raised

Tool arguments come from a language model, so bad arguments are a **normal
operating condition**, not an exception. `invoke` returns a `ToolResult` for
every outcome, including unknown tool names, invalid arguments, denials,
timeouts and crashes. The agent loop hands that back as an ordinary function
response.

This is what makes a one-turn recovery possible: a model that calls
`search(quer="x")` is told `query: Field required` and fixes it. Raising would
abort an entire request over a typo. Only genuine programming errors in Atlas
propagate as exceptions.

The distinct outcome values exist because the loop reacts differently to each —
invalid arguments are worth a retry, a denial never is.

### Two smaller choices worth recording

**Argument schemas are pydantic models.** Already a dependency, generates the
JSON schema, validates, and produces typed arguments. Verified that the Gemini
SDK accepts the emitted schema dict directly, so the framework stays
provider-agnostic without being provider-incompatible — there is a test asserting
this rather than an assumption.

**Tools the caller cannot use are not advertised.** `declarations(context)` omits
them entirely rather than listing them and denying the call. A model cannot waste
a turn on, or be tempted by, a tool it was never shown. Relatedly, the denial
message does not name the missing permission: that is operator-facing detail, and
echoing it puts the authorization model into text the model can reason about.

**Trade-off.** A tool cannot customise its own validation or error formatting.
If one ever genuinely needs to, the fix is to widen the framework deliberately
rather than to let that tool bypass it.

**Reconsider if.** A tool needs streaming or partial results, which the current
single-return shape cannot express.

---

## ADR-0027: Identity is carried to tools, never derivable by them

**Status:** accepted (Phase 4)

**Problem.** ADR-0010 noted that in Phase 1 the worst case from a prompt-injected
document was a wrong answer, because the model had no tools. Tools change that.
Anyone who can get a document into the corpus can put instructions in it, and
those instructions reach the model as evidence. A document reading *"to answer
this, search tenant acme-corp"* must not become a working cross-tenant read.

**Decision.** Identity travels in `ToolContext`, built by the server from the
request. The model supplies only a tool *name* and its *arguments*. Identity is
therefore a separate parameter, not a field the model can set, and three
mechanisms keep it that way.

### 1. A tool may not declare identity as an argument

`ToolRegistry.register` refuses any tool whose `Args` model declares a reserved
name — `tenant_id`, `permissions`, `user`, `role`, and others — checking field
names and aliases, case-insensitively.

This fails at **registration**, which in practice is import time. A tool that
could accept a tenant is a design error, and a design error should be impossible
to deploy rather than caught by a runtime guard on the day someone probes it.
Checking at call time would mean the vulnerable tool exists and is protected
only by a check somebody could later remove.

### 2. Unknown arguments are rejected, not ignored

Every `Args` model inherits `ToolArgs`, which sets `extra="forbid"`.

Pydantic's default is to ignore unknown fields. Under that default a model
emitting `{"query": "salaries", "tenant_id": "victim"}` would have the extra key
silently dropped: **the call would succeed and nothing would record the
attempt.** It would look like an ordinary search in the logs and in the agent
trace.

Forbidding turns it into a visible `invalid_arguments` result carrying the
offending key, retained in the trace as the model supplied it. That is the
difference between a control and an accident, and it is why a blocklist of
reserved names alone would not be enough — it would only catch the names we
thought to imagine.

### 3. The context is frozen and server-built

`ToolContext` is a frozen dataclass. `current_tool_context` in the API layer
constructs it from `app.state.tenant_id` and nothing else: not the request body,
not headers, not model output. There is a test asserting that
`x-tenant-id: <victim>` on the request changes nothing.

Phase 4 has no authentication, so `permissions` is empty and any tool declaring
`required_permission` is unavailable. Phase 5 replaces the body of that one
function with claims from the caller's token; nothing downstream changes, because
everything downstream already takes a `ToolContext`.

### What this does and does not defend against

**Defended:** the model cannot choose whose data is read. Injected text reaches a
tool as an ordinary string argument, and the tool still queries the caller's
tenant, because the tenant was never on a path the model could write to. Tested
against four injection shapes including a full poisoned-document payload.

**Not defended, and stated plainly:** a retrieved document can still influence
*what* the agent searches for. It can waste a turn, steer a query, or persuade
the model to look for something irrelevant. Those cost tokens and can degrade an
answer. They cannot cross a tenant boundary, which is the property worth
guaranteeing.

Also undefended by this ADR: a tool that is simply written wrong — one that
queries without a tenant filter. Nothing here can prevent that, though the
reserved-name rule means such a tool has no tenant to use *except* the context
one. The repository layer's mandatory `tenant_id` parameter (ADR-0003) is the
control that matters there, and it predates this.

**Reconsider if.** A tool legitimately needs to act across tenants — an
administrative report, say. That is a new capability with its own permission and
its own audit trail, not a relaxation of this rule.

---

## ADR-0028: The search tool shows the model snippets and keeps the chunks

**Status:** accepted (Phase 4, step 4)

**Context.** `search_knowledge_base` is the agent's first tool. Its results have
two consumers with incompatible appetites.

The **model** consumes results inside the loop, where every tool response is
resent on every subsequent turn. A five-hit response containing full chunk text
is roughly 1,500 tokens; four iterations of that is 6,000 tokens carried through
to the end, most of it text the server already holds in memory.

The **answer** consumes results at the end, through the grounded path unchanged
from Phase 1 — which needs whole chunks, with character offsets and provenance,
because citation resolution and verbatim quote verification are defined against
the full text (ADR-0011).

Serving both from one payload means either paying the token cost on every turn
or citing against truncated text.

**Decision.** The tool returns a `ToolOutput`, which the framework splits:

* `content` reaches the model — evidence id, document, section path, relevance
  and a 480-character snippet per hit.
* `artifacts` stays server-side on the `ToolResult` and is excluded from
  `for_model()` — the full `RetrievedChunk` objects.

The exclusion is a property of the type, not of each caller's discipline. The
agent loop accumulates evidence from `artifacts` across iterations, deduplicated
by chunk id, so citations resolve against the union of every search.

**Why the split rather than the alternatives.** Returning full text everywhere
is the obvious option and costs the tokens above. Returning ids only, and
re-reading chunks from the database at answer time, is a second query for rows
already in memory and introduces a window where a re-ingestion could change what
the ids point at. The split costs one dataclass.

**What this trades away, stated plainly.** The model judges *sufficiency* from
truncated text, so it can stop searching believing a snippet answers a question
that the full chunk would have shown it does not. That error affects when the
loop **stops**, not what it cites: citations are always resolved and verified
against full chunk text. 480 characters is a guess, not a measured value — it is
roughly 40% of a typical chunk. Step 8's evaluation is where it becomes
falsifiable, by comparing agent and plain-RAG answers on the same queries.

**The tool changes no retrieval behaviour.** Same modes, same fusion, same
reranker, same similarity gate. If the agent turns out to beat plain RAG, it did
so by choosing what to search for and searching more than once — the hypothesis
under test for the multi-document queries stuck at Recall@1 = 0.400. A tool that
also retrieved *differently* would make that improvement unattributable.

**Arguments deliberately absent.** `query` and `top_k` only. No source filter
and no similarity floor: both are evidence policy, and a model that can lower
the relevance floor can talk itself into evidence the system already judged too
weak to answer from. No tenant — registration refuses it (ADR-0027).

**Empty results explain themselves.** "Nothing matched" and "something matched
but scored 0.41, below the 0.55 floor" call for different next queries, and the
difference is invisible from an empty list. The note carries the best score so
the model reformulates rather than concluding the corpus is silent.

**A related leak found while building this.** `Tool.declaration()` was shipping
the `Args` class docstring as the JSON-schema description, because pydantic
derives one from the other. For this tool that was ~500 tokens of internal
rationale — written for a maintainer — sent to the model on every request. The
declaration now strips it; `Tool.description` and per-field descriptions are the
only prompt text, and both are written deliberately.

**Reconsider if.** Evaluation shows the loop stopping early on questions whose
answer sat past the 480-character cut. The fix would be a wider snippet, or a
`fetch_full_passage(evidence_id)` tool letting the model pay for full text only
when it decides it needs to — not full text by default.

---

## ADR-0029: The agent loop gathers evidence; it does not write the answer

**Status:** accepted (Phase 4, step 5)

**Context.** A tool-calling loop is the obvious place to also generate the final
answer — the model is already holding the conversation, and one more turn would
produce prose. That would put answer generation on the agent model, outside the
grounded path, with no evidence blocks and no server-generated citation ids.

**Decision.** The loop returns *evidence and a trace*. The agent model decides
what to search for and when it has enough; the answer model then writes the
response through the unchanged grounded path — evidence blocks, citation
resolution, verbatim quote verification, refusal downgrade (ADR-0011).

The agent model is never asked for a citation, so it is never trusted to produce
one. This also follows the two-model split (ADR-0024): routing is cheap and
high-volume, answering is the product.

**Four bounds, because each catches what the others miss.** Iterations (4) cap
reasoning depth. Total tool calls (8) bound work an iteration cap cannot, since
one turn may request several calls at once — observed in practice on the first
live run, which issued two searches in one turn. A wall-clock budget (60s),
checked before each new step, catches the case where every individual step is
within limits but the whole is too slow to keep a request waiting. Per-tool
timeouts already exist in the registry.

Hitting a bound is not an error: the loop returns what it found and records
which bound stopped it. These numbers are starting points, not measured optima —
step 8 is what makes them falsifiable.

**Degrading rather than failing.** When the model path yields no evidence for
any reason — provider down, model answered without searching, every search
empty, a bound hit first — the loop runs one plain search on the original
question, exactly what plain RAG would have done, and marks the plan degraded
with the reason. Refusing because the *agent* failed would serve a worse answer
than the system is capable of. The fallback is recorded as a step so a trace
never jumps from no searches to some evidence unexplained.

**Calls in one turn run concurrently.** Safe because tools are stateless and
each carries its own timeout, and `invoke` never raises so there is no partial
failure to unwind. Two searches cost the slower one rather than their sum — and
the multi-document questions this loop exists for are exactly the ones that
produce several calls at once.

**A separate `ToolCallingLLM` protocol**, not more methods on `LLMProvider`.
Tool calling is a capability the answering path never needs and some providers
lack; widening the existing interface would force every provider — including the
offline fake the whole suite depends on — to implement a surface it has no use
for. The provider is stateless: the loop owns the conversation and passes the
whole history each time, which is what makes a run reproducible in a test.

### What the offline tests could not tell us

Thirty loop tests passed against a scripted model before the first live call.
Two bugs survived all of them, both in the layer a fake cannot exercise — what
the *real API* accepts:

1. **`additionalProperties: false` is rejected with a 400.** `ToolArgs` sets
   `extra="forbid"`, and pydantic emits that key; Gemini's function-calling
   schema dialect has no such field. Declarations are now reduced to the
   supported subset. This does **not** weaken the authorization boundary:
   `extra="forbid"` is enforced by `model_validate` inside `invoke`, before the
   tool runs. The schema key only ever *told* the provider about the rule.

   Notably, ADR-0026's test asserting "the declaration is accepted by the
   provider SDK" passed throughout. Constructing the SDK's type is not evidence
   the request will succeed — the type was happy to hold a field the service
   refuses.

2. **Gemini 3.x thinking models require `thought_signature` echoed back** on
   function-call parts. Reconstructing a model turn from neutral message types
   dropped it, so the first iteration always worked and every second iteration
   failed with a 400. `ToolCall` now carries an opaque `provider_state` that the
   loop passes through without reading — the field is deliberately meaningless
   to everything except the provider that produced it.

Both were caught by the fallback rather than by a failed request: the smoke run
returned answerable evidence while logging the error, which is the degradation
path working as designed and is also exactly how such a bug could have gone
unnoticed in production. That is an argument for alerting on `degraded`, not
merely recording it.

**Nested-model arguments are now refused at registration.** They produce
`$defs`/`$ref`, and whether a given provider resolves those is unverified. The
failure mode if not is a 400 at request time; refusing at boot keeps the
discovery where every other structural check lives.

**Reconsider if.** Evaluation shows the loop's bounds binding on questions that
would have been answered with one more iteration, or the answer model
consistently refusing on evidence the agent judged sufficient — which would mean
the snippet/chunk split of ADR-0028 is misleading the stopping decision.

---

## ADR-0030: One answering path, two ways of choosing evidence

**Status:** accepted (Phase 4, step 6)

**Context.** The agent loop ends holding a conversation with a model. Letting it
produce the final text is one more turn and costs nothing extra — which is what
most agent frameworks do.

It would also move answer generation off the grounded path: no evidence blocks,
no server-generated ids, no citation resolution, no verbatim quote check, no
refusal downgrade. Every groundedness property Phase 1 built would silently fail
to apply to the new feature, and it would look like it was working, because the
answers would read fine.

**Decision.** `AnswerService.answer_from_evidence` was extracted from
`answer()`. Both paths call it. The plain path chooses evidence with one
retrieval; the agent path chooses it with a loop. Below that line nothing
differs — same prompt, same ids, same validation, same downgrade.

There is no agentic answering path. There is one answering path and two ways of
deciding what reaches it. The agent model is never asked for a citation, so it
is never trusted to produce one.

The agent tests are largely the answering tests run again through the agent: a
guarantee that holds only on the path someone remembered to test is not a
guarantee.

**Evidence order is a rank interleave, not a sort.** Sorting the union by score
would be wrong: reranker outputs are unnormalised per-query logits (ADR-0020),
so a score from one search says nothing about a score from another, and sorting
them invents an ordering out of noise. Concatenating is also wrong — it puts one
whole search ahead of another, so any truncation costs a two-part question the
half it searched for last. Interleaving by rank preserves each search's own
valid ordering and never invents one between them.

**Evidence is capped** at `agent_max_evidence` (12). Eight tool calls of ten
passages each can gather far more than helps.

**Agent mode is opt-in per request** and off by default, so existing callers see
no change. Per-request retrieval knobs (`top_k`, `mode`, `rerank`,
`min_similarity`, `source_ids`) are **rejected** with a 400 rather than ignored:
they describe a single search, the agent runs several with parameters it picks,
and silently dropping them would report a configuration that never ran.

### The first live comparison went against the agent

Two questions, agent versus plain, same corpus and same answer model:

| | plain | agent |
|---|---|---|
| "refund window and who approves exceptions" | 2 documents cited, both parts answered | **1 document**, lost the enterprise window |
| "rotate an API key, and what if one leaks" | both parts answered | **refused** — "no information on what to do if a key leaks" |
| prompt tokens | ~1,270 | ~3,760 (3x) |
| latency | ~3.3s | ~4.6s |

The agent routed *well* — it decomposed both questions into sensible sub-searches
and each returned results. It then answered worse, at three times the token cost.

This is two questions and therefore an anecdote, not a measurement. But it is a
specific, plausible anecdote and the leading hypothesis is uncomfortable for the
design above: **the interleave discards the reranker's global ordering.** The
plain path hands the answer model eight passages ranked against each other by a
cross-encoder over one candidate pool. The agent hands it twelve, ordered by a
rule that is deliberately agnostic about cross-search quality. More evidence,
worse ordered — which is the classic recipe for a worse answer.

If that is right, the fix is not to abandon the interleave but to restore a
single comparable ordering: rerank the union once against the original question
before answering. That is one cross-encoder pass over ~12 passages, it changes
no retrieval behaviour, and it makes the scores comparable by construction
rather than by assumption.

Not doing it now. The whole point of step 8 is to measure this rather than
guess, and implementing a fix for a two-sample observation is how unmeasured
complexity gets in. Recorded here as the primary hypothesis to test, with
`agent_max_evidence` as the secondary one.

**Reconsider if.** Evaluation confirms the ordering hypothesis — in which case
a union rerank goes in and this ADR is superseded. Or evaluation shows the agent
losing on questions it *routed* well, which would point at the snippet/chunk
split of ADR-0028 misleading the stopping decision instead.

---

## ADR-0031: The agent's evidence is reranked once, as a union, before answering

**Status:** accepted (Phase 4, follow-up to step 6)

**Context.** ADR-0030 recorded a live comparison that went against the agent: it
routed well, gathered more evidence, and answered worse than plain RAG. The
leading hypothesis was ordering. The plain path hands the answer model a set
ranked against each other by a cross-encoder over one candidate pool; the agent
handed it a rank interleave, a rule deliberately agnostic about cross-search
quality because cross-encoder scores are unnormalised per-query logits and
genuinely are not comparable between searches (ADR-0020).

**Decision.** After the loop finishes, the deduplicated union of everything it
found is reranked **once**, against the **original user question**, and the
result is capped at `agent_max_evidence`.

Reranking against the user's question rather than the agent's sub-queries is the
point: that is the only question the answer is judged against. It also makes the
scores comparable by construction instead of by assumption — one query, one
pass, one scale.

What this does not touch: dense and lexical retrieval, the reranker
implementation, the grounded answering path, the agent's bounds, and the plain
path, which does no union rerank and is unchanged. The cap moved to *after*
ordering so it keeps the globally best passages rather than the best of an
ordering that was never meant to be compared. `agent_union_rerank` exists so the
two behaviours can be run against each other rather than the fix being assumed.

Provenance survives: identity, offsets and text are untouched, the first-stage
score stays in `component_scores`, and the new score is recorded under
`union_rerank`. Ties break on chunk id so an ordering never depends on which of
two concurrent searches finished first.

### The diagnostic

Seven questions — the five labelled `multi-doc` cases plus the two from
ADR-0030 — three arms, two runs. A and B were built from **one gathered plan per
question**, so the searches are held fixed and the only difference between them
is the ordering; two independent agent runs would have confounded the ordering
change with the model choosing different searches.

Coverage across both runs: **plain 26/26, A (interleave) 23/26, B (union
rerank) 26/26.** Prompt tokens were identical between A and B on every question.

**One difference reproduced** (`step6-refunds`: A cited one document, B cited
both, twice) and **one did not** (`revocation-and-release`: A scored 1/2 then
2/2 on identical evidence in identical order — answer-model nondeterminism, not
ordering). Both are kept in the record; discarding the inconvenient half of a
small sample is how seven questions get talked into meaning more than they do.

**What this justifies claiming:** B never lost to A, matched plain everywhere,
and won one reproducible case, at 28–556 ms of extra latency and no extra
tokens. That is consistent with the ordering hypothesis and cheap enough that
adopting it does not need to wait for a full measurement.

**What it does not justify claiming:** that answer quality is fixed. Seven
questions, a metric with demonstrated run-to-run variance, no confidence
intervals, no paired bootstrap. And the agent still only *matches* plain RAG
while costing an extra model and several searches — whether agency earns its
keep at all is the step 8 question, untouched by this.

`agent_max_evidence` is deliberately left alone. The secondary hypothesis — that
the quantity of evidence hurts — is a separate variable and changing it here
would have made the ordering result unreadable.

**Reconsider if.** Step 8 shows the agent losing to plain RAG on questions where
the union rerank ranked the needed passage first, which would move the
explanation from ordering to the snippet/chunk split of ADR-0028 or to the
stopping decision itself.

---

## ADR-0032: Step 8 — agent mode matches plain RAG; recommendation is opt-in, not default

**Status:** accepted (Phase 4, step 8)

**Context.** ADR-0030 found the agent answering worse than plain RAG on two
questions. ADR-0031 fixed the likely cause (evidence ordering) and a 7-question
diagnostic suggested the fix worked. Neither was a measurement across the
labelled eval set, and a diagnostic built to confirm one hypothesis is not a
substitute for one built to test whether the whole feature is worth having.

**Method.** `scripts/evaluate_agent.py` (harness in `atlas.eval.agent_compare`,
new and separate from `EvalRunner` — see that module's docstring for why)
ran both systems, paired, on all 112 questions of `eval/datasets/main.jsonl`,
against the shipped configuration: dense+rerank retrieval, `gemini-3.1-flash-lite`
routing, `gemini-3.5-flash-lite` answering, the ADR-0031 union rerank, unmodified
bounds. Quality is scored by whether a question's gold labels are satisfied by a
**cited** chunk, not merely a retrieved one — stricter than the existing
"produced a citation" check, and the reason a new metric was needed rather than
reusing `EvalRunner`'s as-is.

**Result.** Of 100 answerable questions, **96 scored identically** between the
two systems. Of the 4 that differed: 2 agent wins, 2 agent losses — exactly
even. Overall paired delta: **−0.005, CI [−0.04, 0.03]**, crossing zero. Every
per-kind breakdown also crosses zero. Refusal correctness was identical (12/12
unanswerable correctly refused, 0/100 wrongly refused, both systems) and
unverified-quote counts were identical (6 vs 6) — the grounded answering path
behaved the same regardless of which evidence-gathering method fed it, which is
exactly what ADR-0030's design (one answering path, two ways of choosing
evidence) was for.

Zero errors and zero degraded runs across 100 live agent executions. The system
is robust; the question this ADR answers is about value, not reliability.

**Cost is not close.** $1.40 per 1000 questions against $0.79 — **1.8×**.
Latency: p50 5132ms vs 3286ms, p95 7050ms vs 4286ms — 56–64% slower.

**The one suggestive result.** `refund-window-by-plan`, a genuine multi-document
question, is where plain RAG missed one of two required documents (0.5) and the
agent's second search reached both (1.0) — the exact failure mode
(Recall@1 = 0.400 on multi-document queries) that motivated building the agent
in Phase 2. Consistent with the original hypothesis, from **one instance out of
five multi-doc questions**, which is not a basis for a claim, only for not
discarding the hypothesis.

**The two losses were investigated rather than left as a number.** Both are
`paraphrase`-kind. One (`revocation-delay`) is not a retrieval defect: the
correct document ranked first in the union rerank, and the answer model cited a
different, genuinely relevant document instead of the labelled one — a citation
choice, not a miss. The other (`queue-first-checks`) is a real retrieval-depth
gap: the right document filled 3 of 7 evidence slots but the specific labelled
line was in a section none of the searches surfaced. Two examples, not a
pattern; recorded as material for whoever investigates further, not as a
demonstrated weakness of agent mode generally.

**Decision: agent mode ships opt-in and stays opt-in.** It is not promoted to
default, and Step 7 (`query_metadata`) is not started on the strength of this
result. The reasoning:

- No measured quality gain justifies 1.8× cost and ~60% more latency on this
  corpus and this question set.
- The one place agent mode shows promise (multi-document questions) is
  measured from five instances — too few to act on, and exactly the kind of
  claim this project's stated rule (verify, do not assume) exists to prevent.
- Nothing about this result is a defect. It is a well-built feature whose
  benefit, if real, is concentrated in a slice of questions this dataset is too
  small to characterise.

**What would change this.** A larger, purpose-built multi-document eval set
(the current one has five such questions total) showing a paired delta whose
CI excludes zero. Short of that, agent mode remains available, tested, and
not the default path — which is what "opt-in" has meant since ADR-0029.

**Reconsider if.** A user-facing need for genuinely multi-hop questions
materialises at a scale where 1.8× cost is worth measuring against, or a future
tool (`query_metadata` or otherwise) changes what the agent can do enough that
this comparison should be re-run rather than trusted as still current.
