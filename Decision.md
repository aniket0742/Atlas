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

## ADR-0002: No Kafka. A queue is deferred, and will probably be Postgres

**Status:** deferred (revisit in Phase 3)

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

**Decision.** Phase 1 ingests synchronously. Phase 3 introduces a Postgres job
table, not Kafka.

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

**Status:** accepted (Phase 1), superseded in Phase 3

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
