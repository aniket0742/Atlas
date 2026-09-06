# Architecture Decision Record

Decisions are recorded as made, reflecting what is **actually implemented**, not
what was planned. An unvalidated decision is labelled as such rather than
glossed over.

Each entry names what would make us reconsider it. A decision with no such
trigger is usually a preference wearing a decision's clothes.

**Status key:** `accepted` — implemented and in use. `provisional` — implemented
but not yet validated by measurement. `superseded` — replaced by a later ADR.
`implemented but not default` — the code ships and is selectable, and
measurement said not to switch it on.

That last status is worth noticing on its own. Lexical retrieval, hybrid
fusion, and agent mode are all implemented, tested, and **not** the default,
because evaluation did not support adopting them. The runs behind those
decisions are committed under `eval/baselines/`, including the rejected ones.

## Index

| # | Decision | Status |
|---|---|---|
| [0001](#adr-0001-postgres-with-pgvector-not-a-dedicated-vector-database) | Postgres with pgvector, not a dedicated vector database | accepted |
| [0002](#adr-0002-no-kafka-the-queue-is-postgres) | No Kafka; the queue is Postgres | accepted |
| [0003](#adr-0003-tenant_id-on-every-table-from-the-first-migration) | `tenant_id` on every table from the first migration | accepted |
| [0004](#adr-0004-embeddings-in-their-own-table-keyed-by-model) | Embeddings in their own table, keyed by model | accepted |
| [0005](#adr-0005-hand-written-sql-with-psycopg-3-no-orm) | Hand-written SQL with psycopg 3, no ORM | accepted |
| [0006](#adr-0006-plain-numbered-sql-migrations-not-alembic) | Plain numbered SQL migrations, not Alembic | accepted |
| [0007](#adr-0007-local-onnx-embeddings-hosted-llm) | Local ONNX embeddings, hosted LLM | accepted |
| [0008](#adr-0008-gemini-behind-a-two-method-provider-interface) | Gemini behind a provider interface | accepted |
| [0009](#adr-0009-structure-aware-chunking-with-exact-character-offsets) | Structure-aware chunking with exact character offsets | **provisional** |
| [0010](#adr-0010-groundedness-is-enforced-in-code-not-requested-in-the-prompt) | Groundedness enforced in code, not requested in the prompt | accepted |
| [0011](#adr-0011-eval-labels-anchor-to-documents-and-snippets-never-to-chunk-ids) | Eval labels anchor to documents and snippets, never chunk ids | accepted |
| [0012](#adr-0012-phase-1-ingests-synchronously-and-says-so) | Ingestion started synchronous | superseded by 0022/0023 |
| [0013](#adr-0013-the-similarity-floor-calibrated-at-060) | The similarity floor, calibrated at 0.60 | superseded by 0019 |
| [0014](#adr-0014-ndcg-counts-gain-once-per-label) | nDCG counts gain once per label | accepted |
| [0015](#adr-0015-a-static-inspection-console-not-a-frontend-framework) | A static inspection console, not a frontend framework | accepted |
| [0016](#adr-0016-the-api-runs-in-docker-compose) | The API runs in Docker Compose | accepted |
| [0017](#adr-0017-postgresql-full-text-search-not-bm25) | PostgreSQL full-text search, not BM25 | accepted |
| [0018](#adr-0018-hybrid-retrieval-implemented-measured-and-not-adopted) | Hybrid retrieval implemented, measured, not adopted | **not default** |
| [0019](#adr-0019-the-similarity-floor-is-a-query-level-gate-not-a-per-chunk-filter) | The similarity floor is a query-level gate | accepted |
| [0020](#adr-0020-cross-encoder-reranking-adopted) | Cross-encoder reranking, adopted | **accepted, default on** |
| [0021](#adr-0021-paired-bootstrap-replacing-comparison-of-independent-intervals) | Paired bootstrap, replacing independent intervals | accepted |
| [0022](#adr-0022-the-ingestion-queue-is-a-postgres-table) | The ingestion queue is a Postgres table | accepted |
| [0023](#adr-0023-post-v1documents-returns-202-and-the-break-is-not-versioned) | `POST /v1/documents` returns 202 | accepted |
| [0024](#adr-0024-two-model-roles-and-gemini-35-flash-lite-for-the-answer) | Two model roles; `gemini-3.5-flash-lite` answers | accepted |
| [0025](#adr-0025-gemini-31-flash-lite-routes-the-agents-tool-calls) | `gemini-3.1-flash-lite` routes tool calls | accepted |
| [0026](#adr-0026-tool-guarantees-live-in-the-registry-not-in-the-tools) | Tool guarantees live in the registry | accepted |
| [0027](#adr-0027-identity-is-carried-to-tools-never-derivable-by-them) | Identity is carried to tools, never derivable | accepted |
| [0028](#adr-0028-the-search-tool-shows-the-model-snippets-and-keeps-the-chunks) | The search tool shows snippets, keeps chunks | accepted |
| [0029](#adr-0029-the-agent-loop-gathers-evidence-it-does-not-write-the-answer) | The agent loop gathers evidence, does not answer | accepted |
| [0030](#adr-0030-one-answering-path-two-ways-of-choosing-evidence) | One answering path, two ways of choosing evidence | accepted |
| [0031](#adr-0031-the-agents-evidence-is-reranked-once-as-a-union-before-answering) | Agent evidence reranked once as a union | accepted |
| [0032](#adr-0032-step-8-agent-mode-matches-plain-rag-recommendation-is-opt-in-not-default) | Agent mode matches plain RAG; opt-in, not default | **not default** |
| [0033](#adr-0033-ci-gates-lint-and-both-test-suites-and-refuses-to-pass-on-a-skipped-suite) | CI gates lint and both test suites | accepted |

---

## ADR-0001: Postgres with pgvector, not a dedicated vector database

**Status:** accepted

**Problem.** Vector similarity search needs an index — a purpose-built vector
database (Qdrant/Weaviate/Milvus/Pinecone) or an index inside Postgres.

**Decision.** Postgres 16 with pgvector.

**Why.** Transactional consistency, not raw ANN performance. A document
version's chunks and embeddings must commit atomically with its metadata —
otherwise a crash mid-ingest leaves chunks retrieval can find but whose
document row still says `pending`. With pgvector that's one
`BEGIN`/`COMMIT`. An external store makes it a distributed write needing an
outbox or reconciliation job — real complexity for a problem this system
doesn't have at its size. Metadata filtering (tenant, source, version) is also
just a join.

Pinecone/hosted removes the operational cost but adds a paid dependency,
already ruled out.

**Trade-offs.** pgvector's HNSW is slower than a tuned dedicated engine at
scale, and Postgres applies the ANN index *before* filter predicates — the
under-fill problem in ADR-0003. Index build is single-threaded and memory
scales with `m`/`ef_construction`.

**Reconsider if.** The corpus exceeds ~10^7 chunks per tenant and p95 latency
stops meeting target after tuning `m`/`ef_construction`/`hnsw.ef_search`; or
filtered-recall under selective tenant predicates can't be fixed by partial
indexes. Both are measurable.

---

## ADR-0002: No Kafka. The queue is Postgres

**Status:** accepted — realised in ADR-0022

**Problem.** The original design put Kafka between the ingestion API and
processing workers.

**Decision.** A Postgres job table with `SELECT ... FOR UPDATE SKIP LOCKED`,
not Kafka or Redis Streams.

**Why.** Kafka earns its complexity at throughput and fan-out this system
doesn't have: one developer, a corpus of thousands of documents, one consumer
type. It also can't participate in the transaction that writes the document
row without an outbox table — at which point the outbox *is* a Postgres queue.
`SKIP LOCKED` handles concurrent workers correctly, makes enqueue atomic with
the write, and needs no new infrastructure: queue depth is a `COUNT(*)`, the
DLQ is a status column. Adding Kafka here would make the diagram look
impressive and the system worse — explicitly against this project's goals.

**Trade-offs.** Competes with query traffic for connections, generates table
churn for autovacuum, no replay of historical events, won't scale to millions
of jobs/hour.

**Reconsider if.** Sustained ingestion exceeds ~10^3 jobs/second, a second
independent consumer needs the same event stream, or replay becomes a
requirement. Redis Streams is the step before Kafka.

---

## ADR-0003: `tenant_id` on every table from the first migration

**Status:** accepted

**Problem.** The system runs with one tenant and no authentication. Does
tenancy wait until an identity layer exists?

**Decision.** No. `tenant_id` on every table and in every query from
migration `0001`, enforced at the repository boundary. Document/source/chunk
ids are `uuid5` values derived from a string that **includes the tenant**, so
two tenants uploading byte-identical files get different ids by construction.

Row-level security and schema-per-tenant were considered and rejected:
RLS needs per-request session variables and complicates pooling; schema-per-
tenant is strong isolation but painful at scale and on migrations.

**Why.** Tenancy is a schema property, not a feature. Retrofitting it means
touching every table, index and query at once, and the failure mode of missing
one is silent cross-tenant data disclosure — the worst bug this system could
have. Doing it now costs one column and one predicate per query; doing it
later costs a migration plus an unverifiable audit of every call site.
Authentication, RBAC and rate limiting are a separate concern and are *not*
pulled forward — only the schema is.

**Known gap.** The HNSW index is global, not per-tenant; pgvector applies the
`tenant_id` predicate after the ANN search, so a selective filter can return
fewer than `k` rows under many tenants even when more matching rows exist.
With one tenant this cannot occur. A fix (partial indexes per tenant, or
pgvector's iterative scan) would be chosen against measured filtered-recall,
not assumed — not needed at current scale.

**Reconsider if.** Tenant counts or compliance requirements make row-level
security or physical separation necessary. The column stays either way.

---

## ADR-0004: Embeddings in their own table, keyed by model

**Status:** accepted

**Decision.** A separate `chunk_embeddings(chunk_id, model, embedding)` table,
keyed on `(chunk_id, model)`, rather than a column on `chunks`.

**Why.** Comparing embedding models is the most basic retrieval comparison
this project makes, and a column on `chunks` forces either destructive
re-indexing or a nullable column per model. A separate table lets two models
coexist over *identical* chunks, so a comparison isolates the model instead of
confounding it with a re-chunk.

**Trade-offs.** One extra join per retrieval query. pgvector pins a fixed
dimensionality per column (`vector(384)`), so a model with different
dimensions needs a new table, not just a new row.

**Reconsider if.** Only one embedding model is ever used and the join shows up
as a measured cost — neither is true today.

---

## ADR-0005: Hand-written SQL with psycopg 3, no ORM

**Status:** accepted

**Decision.** psycopg 3 (async) with hand-written SQL in a repository module,
not an ORM.

**Why.** The queries that matter are pgvector distance operators and, later, a
hybrid lexical/vector fusion — neither expressible in an ORM without dropping
to raw SQL anyway. An ORM would add indirection over the part of the codebase
most worth reading, for conveniences (identity map, lazy loading) this system
doesn't use.

**Trade-offs.** No automatic schema/model consistency, no query composition
helpers, manual parameter binding — mitigated by keeping all SQL in one module
and taking `tenant_id` explicitly everywhere.

**Reconsider if.** The schema grows enough CRUD surface that hand-written SQL
becomes the bulk of the code rather than the interesting part of it.

---

## ADR-0006: Plain numbered SQL migrations, not Alembic

**Status:** accepted

**Decision.** Numbered `.sql` files applied in order, tracked in a
`schema_migrations` table with a checksum per applied file.

**Why.** The schema's interesting content is pgvector index DDL with tuning
parameters and a generated `tsvector`/GIN column — Alembic's autogenerate
models neither, so they'd be hand-written inside migrations regardless,
leaving Alembic as a dependency without leverage. Postgres has transactional
DDL, so each migration applies atomically with no framework help. The
checksum means editing an already-applied migration is refused, not silently
allowed to disagree with the database.

**Trade-offs.** No autogenerate, no downgrade path, no branch merging.
Forward-only is deliberate; rollback after a migration needs a compensating
migration.

**Reconsider if.** Multiple developers start producing concurrent migrations
needing branch merging, or downgrade becomes an operational requirement.

---

## ADR-0007: Local ONNX embeddings, hosted LLM

**Status:** accepted

**Problem.** A hard project constraint is no paid API dependency.

**Decision.** Embeddings run locally on CPU via `fastembed`
(`BAAI/bge-small-en-v1.5`, 384 dimensions, 512-token limit); generation goes
to Google Gemini's free tier; reranking later uses a local cross-encoder from
the same library.

**Why local embeddings.** Retrieval-quality work means re-embedding the corpus
for every comparison. If that costs money and network round-trips,
experiments get run once and conclusions get assumed. Local embeddings make
re-indexing free and repeatable, remove the network from the ingest path, and
are deterministic — an eval run from last month stays comparable with one from
today, unlike a hosted model that gets silently versioned underneath you.
`fastembed` over `sentence-transformers` for ONNX Runtime (keeps the worker
image small) and an exposed tokenizer (chunking budgets in real tokens).

**Trade-offs.** A 33M-parameter model is weaker than a large hosted one — a
measured gap, not an accepted one, since the provider interface allows
swapping. `fastembed` downloads a **quantised** ONNX build
(`qdrant/bge-small-en-v1.5-onnx-q`), which may differ slightly from the
original FP32 weights; any published number must name the build.

**Measured (single laptop CPU, not a benchmark).** ~350ms per ~250-token chunk
in a batch of ten — slow enough to matter for bulk ingestion, recorded as a
baseline rather than a claim.

**Reconsider if.** Retrieval quality against the eval set is materially worse
than a hosted embedding model *and* the no-paid-dependency constraint changes.

---

## ADR-0008: Gemini behind a two-method provider interface

**Status:** accepted

**Decision.** `LLMProvider` and `EmbeddingProvider` are `Protocol`s. The
Gemini implementation uses `client.models.generate_content` from
`google-genai` 2.21.0 — the stable, documented surface that accepts
`response_schema`, verified against the installed SDK rather than assumed.

**Why an interface at all.** Two concrete reasons: tests and CI must run with
no API key or network (a deterministic fake exists for both providers), and
the eval harness must hold retrieval constant while varying the model.

**Deliberately narrow.** One generation method, returning **structured output
only** — `generate_structured(...) -> (parsed, TokenUsage)`. No free-form text
method, because requiring a schema is what makes citation validation possible
(ADR-0010). Provider-specific concerns (safety settings, thinking budgets) are
constructor arguments, not interface parameters.

**Free-tier consequences.** HTTP 429 is a normal operating condition, so
exponential backoff with full jitter is built in from the start; other 4xx
errors fail fast since retrying them only burns quota. Free-tier model
availability isn't reliably documented, so `atlas models` lists what a key can
actually reach rather than the project asserting which models are free.

**Reconsider if.** Rate limits make development impractical, at which point a
local model via Ollama is the fallback preserving the no-paid-dependency
constraint.

---

## ADR-0009: Structure-aware chunking with exact character offsets

**Status:** provisional — not yet validated against alternatives

**Problem.** Where to cut documents. Fixed-size windows, fixed-token windows,
recursive splitting and structure-aware splitting were all considered.

**Decision.** Split on structure (markdown headings, then blank-line blocks),
then pack into token-budgeted windows measured with the embedding model's own
tokenizer. A chunk never spans a section boundary; oversized blocks split on
sentence boundaries, then hard-split only as a last resort.

Two invariants the rest of the system depends on: `chunk.text` equals an exact
slice of `document.content` (citations resolve by slicing, so any deviation
surfaces as a wrong quote), and `chunk.token_count` never exceeds the
embedding model's max — nothing is silently truncated at embed time.

**Why.** Fixed-size windows cut mid-sentence and mid-table. Most of the target
corpus (markdown, docs, READMEs) carries explicit structure, so respecting it
is cheap. The no-spanning rule exists because a chunk crossing a section
boundary gets labelled with only one section's heading path — an actual defect
found by the chunking tests, not a hypothetical. Undersized chunks are
**merged backwards within their section, never dropped**: a short section
("## Contact" plus an address) is small but exactly what users ask about, and
a dropped chunk is unretrievable content with no signal it's missing.

**Trade-offs.** More complex than a character splitter. Structure detection is
markdown-specific; PDFs and plain text degrade to blank-line blocks. Sentence
splitting is regex-based and mis-splits on abbreviations.

**Not yet justified.** The parameters (320 target tokens, 64 overlap) are
starting values. Whether this beats naive fixed-size chunking on this corpus
is unmeasured, and the status stays `provisional` until it is.

**Reconsider if.** Measurement shows no advantage over fixed-size chunking, or
shows parent/child retrieval materially better.

---

## ADR-0010: Groundedness is enforced in code, not requested in the prompt

**Status:** accepted

**Problem.** Preventing fabricated answers and fabricated sources.

**Decision.** The model returns a JSON object (`answer`, `citations[]`,
`sufficient_evidence`), treated as a *proposal* and validated: (1) a cited id
must be one of the server-generated ids supplied for that question, or it's
discarded; (2) the quote must appear verbatim in the cited chunk — a citation
that resolves but doesn't match is **kept and flagged**, not dropped, since
it's usually paraphrase and occasionally fabrication, both worth counting; (3)
an answer claiming sufficient evidence with no resolvable citation is
converted into a refusal. Retrieval returning nothing above the similarity
floor never reaches the model at all.

**Why.** A prompt asking the model to cite its sources is a request; these are
guarantees. Asking a confidently-wrong model to self-report is not a control.

**Prompt injection.** Retrieved documents are attacker-influenced — anyone who
can get a document into the corpus can put instructions in it. No complete
defence is claimed. What's implemented: evidence is delimited in tagged blocks
declared as untrusted data, and citation ids are server-generated so injected
text can't mint a source. Tool authorization (ADR-0027) is likewise derived
from the caller's identity and never from retrieved text.

**Residual risk, stated plainly.** A document asserting "the refund window is
900 days" is faithfully reported as saying so. The property enforced is
faithfulness to sources; source trustworthiness is a different, out-of-scope
problem.

**Trade-offs.** Structured output constrains phrasing. Strict validation can
downgrade a correct answer to a refusal when the model cites sloppily — a
deliberate bias toward false refusals over false confidence, measured by the
eval harness.

**Reconsider if.** Measured incorrect-refusal rate on answerable questions
outweighs the groundedness benefit.

---

## ADR-0011: Eval labels anchor to documents and snippets, never to chunk ids

**Status:** accepted

**Decision.** A relevance label names a document by its stable external id
plus, optionally, a text snippet that must appear in the retrieved chunk —
never a chunk id.

**Why.** Chunk ids derive from `(document, version, ordinal)`, so any change to
chunking, overlap or token budget renumbers every chunk and silently
invalidates the dataset. Since the harness exists to compare chunking
strategies, chunk-id labels would be destroyed by the first experiment they
were built to evaluate. Document-plus-snippet labels survive re-chunking,
re-embedding and re-ingestion, and stay readable to a human.

**Trade-offs.** Coarser than chunk-level labelling — a label is satisfied by
any chunk from the right document containing the snippet. Substring matching
means a label breaks if the source document is edited; the test suite asserts
every shipped label still matches its corpus file.

**Reconsider if.** Chunking stabilises permanently and finer-grained labels are
needed to discriminate between close configurations.

---

## ADR-0012: Phase 1 ingests synchronously, and says so

**Status:** superseded by ADR-0022 and ADR-0023. Retained because the
reasoning for starting synchronous, and the two properties that made moving
off it cheap, explain how that later work went.

**Decision.** `POST /v1/documents` ran the full pipeline inline and returned
201 once queryable.

**Why.** The pipeline's behaviour was still being established; running it
inline kept it debuggable and made failures immediate. Deterministic ids and a
content-hash short-circuit were already in place, so moving the call site
behind a queue later would not touch the pipeline itself.

**Trade-offs.** A large PDF blocked its own request; no retry on failure.

**Reconsider.** Done — the endpoint now enqueues and returns 202 (ADR-0023).

---

## ADR-0013: The similarity floor, calibrated at 0.60

**Status:** superseded by ADR-0019 — calibrated by experiment E2

**Problem.** Retrieval always returns its top-k, so a question with no answer
in the corpus still yields k confident-looking chunks.

**Decision.** A cosine floor of **0.60**; below it Atlas refuses without
calling the model.

**How it was chosen.** `scripts/calibrate_floor.py` swept thresholds on the
19-query smoke set. Measured with `bge-small-en-v1.5`: answerable questions'
best-relevant-chunk score ranged 0.669–0.879; unanswerable questions' best
score overall ranged 0.569–0.639. The distributions didn't overlap, and
0.60 sits comfortably below the weakest genuine answer while still catching
one unanswerable query before any token is spent — deliberately conservative,
since the floor is a cheap pre-filter and the model's own
`sufficient_evidence` judgement plus citation validation (ADR-0010) are the
real controls.

**Reconsider if.** The eval set grows (this separation is very likely an
artefact of a small synthetic corpus), the embedding model changes (absolute
cosine values aren't comparable across models), or the measured
false-refusal rate rises. — Superseded: see ADR-0019.

---

## ADR-0014: nDCG counts gain once per label

**Status:** accepted

**Problem.** The first baseline run reported **nDCG@8 = 1.0164** — impossible,
since nDCG is normalised to [0, 1].

**Cause.** DCG summed gain at every rank holding a relevant chunk, while IDCG
was computed from the number of *labels*. Because chunks overlap, several
chunks can satisfy the same label, letting DCG exceed IDCG.

**Decision.** nDCG takes one position per distinct label — the earliest rank
at which it was satisfied. Precision still counts every relevant chunk in the
top k (a statement about the retrieved list, not label coverage); MRR is
unaffected.

**Why it matters beyond the arithmetic.** A score above 1.0 was caught
immediately; the same bug at smaller magnitude would have silently inflated
every number, and later comparisons would have run against a corrupted
baseline. A regression test now asserts nDCG stays within [0, 1] for inputs
where several chunks share a label.

---

## ADR-0015: A static inspection console, not a frontend framework

**Status:** accepted

**Problem.** The system needed to be demonstrable through a browser without
frontend work displacing the backend/retrieval work that is the point of the
project.

**Decision.** Three static files (`index.html`, `app.css`, `app.js`) served by
the existing FastAPI app — no React/Vite, no separate static container.

**Why.** Same-origin means no CORS and no second port. More usefully, it makes
the console a plain client of the documented API — **it cannot display
anything the API does not already return**, which keeps the API honest rather
than letting a capable frontend paper over gaps. The whole console is a few
hundred lines; a framework's advantages start well past that.

**Security note.** Document titles, headings and chunk text are all
attacker-influenced (they originate in uploaded files). The console builds
every node with `textContent`, never `innerHTML` on response data — stored XSS
is structurally impossible rather than a review item.

**Not included.** Streaming. `/v1/query` returns one complete JSON response,
so the console shows a spinner rather than a fake typewriter animation over an
already-complete response.

**Reconsider if.** The console outgrows one file per concern, or a genuinely
interactive view needs real state management.

---

## ADR-0016: The API runs in Docker Compose

**Status:** accepted

**Problem.** Postgres and Redis were containerised while the API ran on the
host, so "run Atlas" meant containers *and* a host Python environment.

**Decision.** A `Dockerfile` for the API and an `api` service in
`docker-compose.yml`; `docker compose up` starts everything, applies
migrations, and serves the console.

**What this forced.** `ATLAS_DATABASE_URL` is overridden inside compose to the
`postgres` service name, since `localhost` means the container itself.
The ONNX model cache is a named volume (`atlas_models`) so the ~67MB download
survives image rebuilds rather than baking into every layer. `.env` is
`.dockerignore`d — the key arrives as a runtime environment variable and never
enters the image, verified against image history. The package is installed
editable (`pip install -e .`) so the migration runner's relative path
resolution keeps working with the source bind-mounted for live-editing.

**Trade-offs.** Image build takes a few minutes (mostly `onnxruntime`).
No orchestration, no non-root user, no pinned base-image digest — a
development image, not a deployment one.

---

## ADR-0017: PostgreSQL full-text search, not BM25

**Status:** accepted

**Problem.** Dense retrieval is weak on rare literal tokens — error codes,
env var names, header names. Measured: Recall@1 was 0.615 for `identifier`
queries against 0.871 for `paraphrase`.

**Decision.** PostgreSQL FTS: a `text_search` tsvector generated column with a
GIN index, ranked by `ts_rank_cd`. A real BM25 extension (`pg_search`/
ParadeDB) pins the project to a non-standard image and its release cycle; an
in-process library (`rank_bm25`) can't compose with the tenant predicate
inside the query, exactly the property ADR-0003 protects.

**Naming, deliberately.** This is **not BM25** and isn't called that anywhere
in the codebase. `ts_rank_cd` rewards lexeme count and proximity but
implements neither BM25's term saturation nor its length normalisation.

**Two implementation details that decide whether it works at all.**
`plainto_tsquery` must be rewritten from `&` (AND) to `|` (OR) — the default
requires every lexeme in a twelve-word question to appear in one chunk and
reliably matches nothing. And the generated column must use two-argument
`to_tsvector` with a pinned config; the one-argument form is only STABLE,
which a generated column rejects.

**Trade-offs.** English-only stemming, no BM25-quality ranking. A GIN index
composes normally with the tenant predicate via bitmap scan, so ADR-0003's
under-fill problem doesn't apply here.

**Reconsider if.** Measurement shows lexical ranking quality is the limiting
factor — see ADR-0018, where it measured *worse* than dense.

---

## ADR-0018: Hybrid retrieval implemented, measured, and not adopted

**Status:** implemented but **not default**

**Problem.** Does combining dense and lexical retrieval improve results?

**Fusion method: Reciprocal Rank Fusion.** Cosine similarity and `ts_rank_cd`
sit on incomparable scales, and per-query min-max normalisation would make
every query's top hit score 1.0 regardless of quality — destroying the signal
that a query has no good match at all. RRF combines ranks, on the same scale
by construction, at the cost of discarding score magnitude.

**Adoption: dense remains default.** Measured on 100 answerable queries,
paired bootstrap against dense:

| configuration | Recall@1 | nDCG@1 | Recall@8 | nDCG@8 |
|---|---|---|---|---|
| dense (baseline) | 0.780 | 0.800 | 0.980 | 0.895 |
| lexical | 0.620 | 0.640 | 0.955 | 0.805 |
| hybrid | 0.720 | 0.740 | 0.990 | 0.881 |

Lexical alone is **significantly worse** (Recall@1 −0.160, CI
[−0.250, −0.070]). Hybrid shows **no measured difference** from dense at
either depth (Recall@1 −0.060, CI [−0.130, +0.010]; nDCG@8 −0.014, CI
[−0.046, +0.018]). Adopting it as default would be adding a subsystem for its
own sake — the specific failure this project set out to avoid.

**What the aggregate hides, and why the code stays.** Per-kind Recall@1:

| query kind | n | dense | lexical | hybrid |
|---|---|---|---|---|
| identifier | 13 | 0.615 | 0.538 | **0.769** |
| conceptual | 13 | 0.615 | **0.846** | 0.692 |
| lookup | 36 | **0.861** | 0.667 | 0.750 |
| paraphrase | 31 | **0.871** | 0.548 | 0.742 |

Hybrid helps identifier queries substantially and hurts paraphrase/lookup, and
this corpus is ~two-thirds paraphrase-or-lookup. On an identifier-heavy corpus
the conclusion could invert, which is why the mode is kept, selectable, rather
than deleted.

**Reconsider if.** A deployment's query mix is identifier-heavy, or weighted
fusion (tuning the dense/lexical contributions rather than treating them
equally) is measured and beats plain RRF.

---

## ADR-0019: The similarity floor is a query-level gate, not a per-chunk filter

**Status:** accepted, supersedes the mechanism in ADR-0013

**Problem.** The Phase 1 floor filtered individual chunks by cosine
similarity. Fusion breaks that: an RRF score is a sum of reciprocal ranks and
a reranker score is an unnormalised logit — a cosine threshold has no meaning
against either, and applying one anyway would silently change refusal
behaviour while every retrieval metric kept looking correct.

**Decision.** The floor is evaluated on the **dense candidates, before
fusion**, as one query-level decision: if no dense candidate reaches it, the
query is treated as having no evidence and answering refuses; otherwise the
final ranking is returned untouched. In `lexical` mode there is no dense score
and no gate — that mode is for measurement, not serving.

**Why this is better than a patch.** It matches what the floor was always
for — deciding whether the corpus can answer at all, a property of the query
rather than of each chunk — and survives any future change to the final
ranking method.

**Value.** 0.55, interim and explicitly **not a validated optimum**. The 0.60
calibrated on 5 documents stopped separating once the corpus grew to 33: 10 of
12 unanswerable queries now score above it, and the distributions overlap
outright. The floor is now a crash barrier for pathological queries; the
model's `sufficient_evidence` judgement and citation validation (ADR-0010) are
the real controls.

---

## ADR-0020: Cross-encoder reranking, adopted

**Status:** accepted, on by default

**Problem.** Both retrieval methods score a query and a passage independently
and compare the results — neither ever sees the pair together.

**Decision.** `Xenova/ms-marco-MiniLM-L-6-v2` (~80MB) via fastembed, reranking
the top 30 first-stage candidates. Local and free, consistent with ADR-0007.

**Measured**, paired bootstrap against dense:

| configuration | Recall@1 | nDCG@8 | retrieval p50 |
|---|---|---|---|
| dense | 0.780 | 0.895 | 77ms |
| dense + rerank | **0.850** | **0.939** | 750ms |

nDCG@8 **+0.044, CI [+0.009, +0.081] — significant.** Recall@1 +0.070 with CI
[+0.000, +0.150], not significant on its own. The gain is broad, not
concentrated in one slice: identifier 0.615→0.923, conceptual 0.615→0.846,
lookup 0.861→0.917.

**The cost, stated plainly.** Retrieval goes from 77ms to ~750ms — ~10x in
isolation, but +23% end to end once generation's ~2.8s is counted.
`ATLAS_RERANK_ENABLED=false` reverts it.

**A negative result worth recording.** `dense+rerank` and `hybrid+rerank` are
indistinguishable (Recall@1 differs by exactly 0.0000; nDCG@8 at k=8 is
marginally *worse* for hybrid, −0.010, CI [−0.030, +0.000]). A cross-encoder
reading the pair directly subsumes what lexical matching contributed, so
running both pays twice for one effect — why the shipped default is dense +
rerank, not hybrid + rerank.

**Reconsider if.** Latency becomes the binding constraint, or a larger
reranker (`BAAI/bge-reranker-base`, ~13x the size) is measured to justify its
cost.

---

## ADR-0021: Paired bootstrap, replacing comparison of independent intervals

**Status:** accepted, corrects the rule registered before these experiments

**Problem.** The pre-registered decision rule was: adopt a configuration only
if it beats the incumbent with **non-overlapping 95% confidence intervals**.
Wrong before any number existed — configurations are evaluated on the *same*
queries, so the comparison is paired, and independent intervals are dominated
by variance *between queries* rather than the difference between
configurations. Two configurations can differ on nearly every query and still
produce comfortably overlapping intervals.

**Decision.** A paired bootstrap over per-query differences. "The interval
excludes zero" is the statement that configurations actually differ.

**Kept honest three ways.** The change was made for a stated methodological
reason, not a disappointing result; it was applied symmetrically, and its
first effect was to make a **negative** result significant (lexical retrieval,
scored "overlapping" by the old test, is significantly *worse* than dense,
CI [−0.250, −0.070]); and both tests are still printed side by side.

**What this does not fix.** 100 queries over one synthetic corpus labelled by
one person — this removes a statistical error, not the generalisation
problem.

**Reconsider if.** The eval set grows enough that a correction for multiple
comparisons becomes worthwhile — the current sweep tests several
configurations against one baseline without one, a real if minor weakness.

---

## ADR-0022: The ingestion queue is a Postgres table

**Status:** accepted — realises the decision deferred in ADR-0002

**Problem.** Ingestion ran inside the HTTP request: a large PDF blocked its
caller, a failure had no retry, and query traffic competed with embedding for
the same process.

**Decision.** An `ingest_jobs` table claimed with
`SELECT ... FOR UPDATE SKIP LOCKED`, drained by a separate worker process — the
property that justifies it is that `POST /v1/documents` creates the source and
the job **in one transaction**, which a broker cannot participate in without
an outbox (and an outbox *is* a Postgres queue with an extra hop).

**Design decisions worth recording individually:**

- **Payload lives in the job row, as `bytea`.** Postgres TOASTs and compresses
  anything past ~2KB out of line; object storage would add a second place the
  payload can disagree with the job after a crash. The payload clears on
  success and is **retained on death** — a dead-letter queue whose entries
  can't be replayed isn't one.
- **Attempts count at claim time, not completion.** A job that crashes its
  worker never reaches a completion handler; counting there would let a
  poison document cycle forever (claim, crash, lease expires, requeue).
  Counting at claim means it exhausts its budget and reaches dead-letter,
  which is what an operator needs to see.
- **Leases, not heartbeats.** A claimed job carries `locked_at`; a reaper
  returns expired leases to `pending`. Cheaper than a heartbeat protocol, at
  the cost of up to 300 seconds of delay after a crash — a lease that must
  exceed the slowest plausible document, so a worker embedding a 200-page PDF
  is never stolen from mid-work.
- **Three transactions per job** (claim, work, completion), not one held
  across the work — holding a claim transaction for the tens of seconds a
  large document takes would pin a connection and block the reaper.
- **Duplicate suppression via a partial unique index over `pending` rows** —
  re-uploading while an earlier upload is still queued replaces that job
  rather than adding a second. Correctness under duplicate delivery comes from
  deterministic ids (ADR-0003), not this index; suppression is an
  optimisation on top.

**Trade-offs.** Competes with query traffic for connections, no replay,
polling costs up to one poll interval before a job starts (`LISTEN/NOTIFY`
would remove that but needs a dedicated connection per worker plus a fallback
poll anyway — not the bottleneck when ingesting a document takes far longer).

**Reconsider if.** Sustained ingestion exceeds ~10^3 jobs/second, a second
independent consumer needs the same stream, or payload sizes make in-row
storage untenable.

---

## ADR-0023: `POST /v1/documents` returns 202, and the break is not versioned

**Status:** accepted

**Problem.** The endpoint returned 201 once a document was queryable. Behind a
queue it can only return "accepted" — the document isn't searchable yet.

**Decision.** `POST /v1/documents` now returns **202 Accepted** with a
`job_id` and `document_id`. No `/v2`: there are no external consumers, so a
compatibility shim would be cost paid for a beneficiary that doesn't exist,
leaving two ingestion paths to test where one is dead weight.

**The `document_id` is returned despite no work having happened** — it's
derived from `(tenant, source, external_id)` via uuid5 rather than assigned by
the worker, so it's knowable at enqueue time and a caller can hold it and poll.

**A behavioural consequence worth stating.** Document-type validation moved
into the worker, since parsing is now deferred work — files the endpoint
previously rejected with 415/422 now surface as a dead-lettered job instead.
The failure isn't hidden, but a client that treats 202 as success will report
success for a document that never indexes; the console polls the job rather
than trusting the 202.

**Reconsider if.** The API gains external consumers, at which point versioning
stops being ceremony.

---

## ADR-0024: Two model roles, and `gemini-3.5-flash-lite` for the answer

**Status:** accepted

**Problem.** One model served every purpose. The agent's tool-routing turns
are high-volume, cheap and latency-sensitive; the final grounded answer is
low-volume and where quality is visible — different jobs.

**Decision.** Two roles behind the existing provider abstraction:

| role | model | why |
|---|---|---|
| final answer | `gemini-3.5-flash-lite` | no measurably better option; see below |
| agent / tool routing | `gemini-3.1-flash-lite` | equal routing quality, fastest, cheapest |

`get_llm()` and `get_agent_llm()` build both from one
`_build_llm(settings, model)` — no new provider type.

**One API key covers both.** Free-tier quota is scoped per project **per
model**, verified by exhausting one model to a 429 and finding the other still
served on the same key — splitting roles adds headroom rather than dividing
one budget.

**Why `gemini-3.5-flash` (the old default) was dropped.** Strictly dominated:
$1.50/$9.00 per 1M tokens against $0.75/$3.75 for `gemini-3.8-flash`, plus a
20 requests/**day** free-tier cap that can't support a 112-query evaluation
run. It had never been re-examined since being chosen early on.

**Why the answer model is the cheap one.** Measured on 112 queries, retrieval
held constant (`eval/baselines/answer-models/`):

| model | unverified quotes | p50 | $/1k queries |
|---|---|---|---|
| gemini-3.1-flash-lite | 14 | 2,254 ms | $0.54 |
| **gemini-3.5-flash-lite** | 3 | 2,507 ms | $0.71 |
| gemini-3.7-flash | 2 | 3,016 ms | $2.44 |
| gemini-3.8-flash | 1 | 3,968 ms | $2.83 |

All four scored 12/12 refusing unanswerable questions, 0 wrongly refused,
100/100 citation coverage — the ADR-0010 guarantees hold regardless of model;
only verbatim-quoting fidelity varies.

**The ranking isn't readable from one run.** `gemini-3.5-flash-lite` produced
5, then 8, then 3 unverified quotes across three identical-configuration
runs. Paired bootstrap on per-query counts against the selected default:
`gemini-3.1-flash-lite` +0.098 (CI [+0.045, +0.161], **significantly worse**);
`gemini-3.7-flash` −0.009 (CI [−0.027, +0.000], no difference);
`gemini-3.8-flash` −0.018 (CI [−0.045, +0.000], no difference). The pricier
models are **not measurably better** at the one thing separating them, while
costing 3.4–4× more and running 20–58% slower. The first single-run table
would have supported the opposite conclusion, from noise.

**What this cost to measure honestly.** Cost was first computed from a
blended 85/15 input:output split, wrong by 15–35% since `3.7`/`3.8` emit 3–4×
more output tokens (billed thinking tokens, invisible in the response text).
The runner now records input/output separately. Input tokens were identical
(141,584) across all four models — confirmation retrieval was genuinely held
constant.

**Reconsider if.** The eval set grows enough to resolve differences the
current n cannot, quoting fidelity degrades in production, or promotional
Flash pricing expires (2027-01-01), doubling `3.6`/`3.7`/`3.8` cost.

---

## ADR-0025: `gemini-3.1-flash-lite` routes the agent's tool calls

**Status:** accepted

**Problem.** The routing role decides when to call a tool and what to search
for — high-volume, latency-sensitive, cheap per call, the opposite profile
from writing the final answer (settled separately in ADR-0024).

**Decision.** `gemini-3.1-flash-lite`, measured against
`gemini-3.5-flash-lite` and `gemini-3.7-flash` on 16 routing cases with real
retrieval (`eval/baselines/agent-routing/`):

| model | selection accuracy | unnecessary searches | multi-doc coverage | latency | $/1k |
|---|---|---|---|---|---|
| **gemini-3.1-flash-lite** | 16/16 | 0 | **4/4** | **2,568 ms** | **$0.36** |
| gemini-3.5-flash-lite | 16/16 | 0 | 3/4 | 2,724 ms | $0.58 |
| gemini-3.7-flash | 16/16 | 0 | 3/4 | 4,684 ms | $1.83 |

**The first benchmark could not justify any choice.** An initial 8-case set
scored **8/8 for every candidate**, including models costing 15x more — not
evidence of equivalence, evidence the benchmark was too easy to measure
anything. It was rebuilt around routing's actual failure modes (vocabulary
mismatch, questions that only look like lookups, answers already in the
question, cross-domain comparisons, terse input). Selection accuracy still
saturated at 16/16 — routing over a single tool is genuinely easy, recorded as
the finding rather than papered over.

**What did separate them: multi-document coverage.**
`gemini-3.1-flash-lite` was the only model to issue two genuinely different
queries for a comparison spanning two documents and reach both; the others
collapsed it into one query. A real capability difference on exactly the
query class measured hardest elsewhere (multi-doc Recall@1 stuck at 0.400 for
every retrieval configuration). Persistence cut the other way — on an
unanswerable question `gemini-3.7-flash` searched four times over 10.5s before
giving up, against one or two for the lite models, reaching the same correct
conclusion. More persistence bought nothing but latency.

**Reconsider if.** The full agent evaluation (112 queries against the real
tool set, rather than 16 synthetic cases against one tool) separates these
models on selection quality — see ADR-0032, which ran that evaluation and did
not change this choice.

---

## ADR-0026: Tool guarantees live in the registry, not in the tools

**Status:** accepted

**Problem.** Every tool needs argument validation, a timeout, an
authorization check and structured logging.

**Decision.** A tool implements exactly one method, `execute`. Validation,
timeouts, permission checks and logging happen in `ToolRegistry.invoke`,
around the call.

**Why.** If validation is the tool's job, the third tool someone adds forgets
it, and the failure is a malformed database query rather than a clean
rejection. Placing the guard in the registry means a tool **cannot opt out of
it by being written carelessly**. The same argument applies to timeouts:
per-tool budgets are class attributes, but enforcement is `asyncio.wait_for`
in the registry, so a tool that forgets to bound its own work is still bounded.

**Failures are returned, not raised.** Tool arguments come from a language
model, so bad arguments are a normal operating condition. `invoke` returns a
`ToolResult` for every outcome — unknown tool, invalid arguments, denial,
timeout, crash — which the agent loop hands back as an ordinary function
response. A model calling `search(quer="x")` is told `query: Field required`
and fixes it in one turn; raising would abort the whole request over a typo.
Only genuine programming errors propagate as exceptions.

**Two smaller choices.** Argument schemas are pydantic models — already a
dependency, and there's a test asserting the Gemini SDK accepts the emitted
schema directly, so the framework stays provider-agnostic without being
provider-incompatible. Tools the caller cannot use are not advertised at
all (`declarations(context)` omits them), so a model can't waste a turn on, or
be tempted by, a tool it never saw — and denial messages never name the
missing permission, which is operator-facing detail the model shouldn't
reason about.

**Reconsider if.** A tool needs streaming or partial results, which the
current single-return shape can't express.

---

## ADR-0027: Identity is carried to tools, never derivable by them

**Status:** accepted

**Problem.** Once tools exist, a document reading *"to answer this, search
tenant acme-corp"* must not become a working cross-tenant read.

**Decision.** Identity travels in `ToolContext`, built by the server from the
request. The model supplies only a tool *name* and its *arguments* — identity
is a separate parameter, never a field it can set. Three mechanisms enforce
that:

1. **A tool may not declare identity as an argument.** `ToolRegistry.register`
   refuses any tool whose `Args` model declares a reserved name (`tenant_id`,
   `permissions`, `user`, `role`, others), checking field names and aliases
   case-insensitively. This fails at **registration** — import time in
   practice — rather than being caught by a runtime guard someone could later
   remove.
2. **Unknown arguments are rejected, not ignored.** Every `Args` model sets
   `extra="forbid"`. Pydantic's default would silently drop an injected
   `tenant_id` key and let the call succeed with no trace of the attempt.
   Forbidding turns it into a visible `invalid_arguments` result carrying the
   offending key — the difference between a control and an accident, and why a
   blocklist alone wouldn't be enough.
3. **The context is frozen and server-built.** `current_tool_context`
   constructs it from `app.state.tenant_id` and nothing else — not the request
   body, not headers, not model output. Tested: `x-tenant-id: <victim>` on the
   request changes nothing.

**What this does and does not defend against.** Defended: the model cannot
choose whose data is read — tested against four injection shapes, including a
full poisoned-document payload. Not defended: a retrieved document can still
influence *what* the agent searches for, wasting a turn or degrading an
answer — it cannot cross a tenant boundary, which is the property worth
guaranteeing. Also not defended: a tool simply written wrong, querying
without a tenant filter — the reserved-name rule means such a tool has no
tenant to use *except* the context one, and the repository layer's mandatory
`tenant_id` (ADR-0003) is the control that matters there.

**Reconsider if.** A tool legitimately needs to act across tenants (an
administrative report). That's a new capability with its own permission and
audit trail, not a relaxation of this rule.

---

## ADR-0028: The search tool shows the model snippets and keeps the chunks

**Status:** accepted

**Context.** `search_knowledge_base`'s results have two consumers with
incompatible appetites. The **model**, inside the loop, resends every tool
response on every subsequent turn — five hits of full chunk text is ~1,500
tokens, and four iterations carries 6,000 tokens of text the server already
holds. The **answer**, at the end, needs whole chunks with offsets and
provenance, since citation resolution and quote verification are defined
against the full text (ADR-0011).

**Decision.** The tool returns a `ToolOutput`, which the framework splits:
`content` reaches the model (evidence id, document, section, relevance, a
480-character snippet); `artifacts` stays server-side, excluded from
`for_model()` — the full `RetrievedChunk` objects. The exclusion is a property
of the type, not caller discipline. The agent loop accumulates evidence from
`artifacts` across iterations, deduplicated by chunk id.

**Why the split rather than the alternatives.** Full text everywhere costs the
tokens above. Ids-only with a re-read at answer time is a second query for
rows already in memory, and a window where re-ingestion could change what the
ids point at.

**What this trades away.** The model judges *sufficiency* from truncated
text, so it can stop searching believing a snippet answers a question the
full chunk would have shown it doesn't. That affects when the loop **stops**,
never what it cites — citations always resolve against full chunk text.
480 characters is a guess (~40% of a typical chunk), not a measured value.

**The tool changes no retrieval behaviour** — same modes, fusion, reranker,
gate. If the agent beats plain RAG, it does so by choosing what to search for
and searching more than once, not by retrieving differently.

**Arguments deliberately absent.** `query` and `top_k` only — no source filter
or similarity floor, both evidence policy a model shouldn't set for itself. No
tenant (ADR-0027 refuses it at registration).

**A related leak found while building this.** `Tool.declaration()` was
shipping the `Args` class docstring as the JSON-schema description (pydantic
derives one from the other) — ~500 tokens of maintainer-facing rationale sent
to the model every request. The declaration now strips it.

**Reconsider if.** Evaluation shows the loop stopping early on questions whose
answer sat past the 480-character cut.

---

## ADR-0029: The agent loop gathers evidence; it does not write the answer

**Status:** accepted

**Context.** A tool-calling loop is the obvious place to also generate the
final answer, since one more turn is free. That would move answer generation
off the grounded path entirely — no evidence blocks, no server-generated
citation ids.

**Decision.** The loop returns *evidence and a trace*. The agent model decides
what to search for and when it has enough; the answer model writes the
response through the unchanged grounded path. The agent model is never asked
for a citation.

**Four bounds, each catching what the others miss.** Iterations (4) cap
reasoning depth. Total tool calls (8) bound work an iteration cap can't, since
one turn may request several calls at once — observed on the first live run.
A wall-clock budget (60s), checked before each new step, catches every
individual step being within limits while the whole is too slow. Per-tool
timeouts already exist in the registry. Hitting a bound isn't an error — the
loop returns what it found and records which bound stopped it.

**Degrading rather than failing.** When the model path yields no evidence for
any reason (provider down, model didn't search, every search empty, a bound
hit first), the loop runs one plain search on the original question — exactly
what plain RAG would do — and marks the plan degraded with the reason.
Refusing because the *agent* failed would serve a worse answer than the system
is capable of.

**Calls in one turn run concurrently** — safe because tools are stateless with
their own timeouts, and `invoke` never raises. A separate `ToolCallingLLM`
protocol, not more methods on `LLMProvider`, since tool calling is a
capability the answering path never needs and the offline fake would
otherwise need to implement a surface it has no use for.

**What the offline tests could not tell us.** Thirty loop tests passed against
a scripted model before the first live call; two bugs survived all of them,
both in what a fake can't exercise — what the *real API* accepts.
`additionalProperties: false` (from `extra="forbid"`) is rejected by Gemini's
function-calling schema with a 400; declarations are now reduced to the
supported subset (this does **not** weaken the authorization boundary, which
is enforced by `model_validate` before the tool runs — the schema key only
ever told the provider about the rule). And Gemini 3.x thinking models require
an opaque `thought_signature` echoed back on function-call parts;
reconstructing a turn from neutral message types dropped it, so the first
iteration always worked and every second failed with a 400 — `ToolCall` now
carries that state through without reading it. Both were caught by the
fallback rather than a failed request, which is an argument for alerting on
`degraded`, not merely recording it.

**Reconsider if.** Evaluation shows the loop's bounds binding on questions one
more iteration would have answered, or the answer model consistently refusing
on evidence the agent judged sufficient.

---

## ADR-0030: One answering path, two ways of choosing evidence

**Status:** accepted

**Context.** Letting the agent loop produce the final text itself would move
answer generation off the grounded path — no evidence blocks, no citation
resolution, no refusal downgrade. Every groundedness property would silently
stop applying to the new feature, and it would look like it was working
because the answers would read fine.

**Decision.** `AnswerService.answer_from_evidence` was extracted from
`answer()`. Both the plain path (one retrieval) and the agent path (a loop)
call it; below that line nothing differs. There is no agentic answering path,
only one answering path and two ways of deciding what reaches it. The agent
tests are largely the answering tests run again through the agent — a
guarantee that holds only on the path someone remembered to test is not a
guarantee.

**Evidence order is a rank interleave, not a sort.** Sorting the union by
score would be wrong: reranker outputs are unnormalised per-query logits
(ADR-0020), so a score from one search says nothing about a score from
another. Concatenating is also wrong — it puts one whole search ahead of
another, costing a two-part question the half searched for last. Interleaving
by rank preserves each search's own valid ordering without inventing one
between them. Evidence is capped at `agent_max_evidence` (12).

**Agent mode is opt-in per request**, off by default. Per-request retrieval
knobs are **rejected** with a 400 rather than silently ignored — they
describe a single search, and dropping them would report a configuration that
never ran.

**The first live comparison went against the agent.** Two questions: the
agent decomposed both into sensible sub-searches, gathered more evidence, and
answered *worse* — one document instead of two on a multi-part question, a
refusal where plain answered, at 3x the tokens (~1,270 → ~3,760) and ~40%
more latency (~3.3s → ~4.6s). Two questions is an anecdote,
but the leading hypothesis was uncomfortable: **the interleave discards the
reranker's global ordering.** Plain hands the answer model eight passages
ranked against each other over one candidate pool; the agent hands it twelve,
ordered by a rule deliberately agnostic about cross-search quality. More
evidence, worse ordered.

Not fixed here — implementing a fix for a two-sample observation is how
unmeasured complexity gets in. Recorded as the hypothesis for ADR-0031 to
test.

**Reconsider if.** Evaluation confirms the ordering hypothesis (a union rerank
follows — see ADR-0031), or the agent loses on questions it *routed* well,
pointing at ADR-0028's snippet/chunk split instead.

---

## ADR-0031: The agent's evidence is reranked once, as a union, before answering

**Status:** accepted, follow-up to ADR-0030

**Context.** ADR-0030's live comparison went against the agent; the leading
hypothesis was evidence ordering, since cross-encoder scores from separate
searches are not comparable (ADR-0020).

**Decision.** After the loop finishes, the deduplicated union of everything it
found is reranked **once**, against the **original user question**, and
capped at `agent_max_evidence` afterward (so the cap keeps the globally best
passages, not the best of an incomparable ordering). Reranking against the
user's question rather than the agent's sub-queries is the point — that's the
only question the answer is judged against, and it makes scores comparable by
construction. Nothing about retrieval, the reranker, or the plain path
changes; `agent_union_rerank` exists so both behaviours can be compared
directly. Provenance survives untouched — identity, offsets, text — with the
first-stage score kept alongside the new `union_rerank` score.

**The diagnostic.** Seven questions, three arms, two runs, with the agent's
searches held fixed (one gathered plan per question) so the only difference
between arms is ordering. Coverage across both runs: plain 26/26, interleave
23/26, union rerank 26/26. One difference reproduced (a two-document question
went from citing one to citing both); one did not (answer-model
nondeterminism on identical evidence). Both are kept in the record.

**What this justifies claiming:** the union rerank never lost to the
interleave, matched plain everywhere, and won one reproducible case, for
28–556ms of extra latency and no extra tokens. **What it doesn't justify:**
that answer quality is fixed overall — seven questions, a metric with
demonstrated run-to-run variance, no confidence intervals. Whether agency
earns its keep at all is the ADR-0032 question, untouched by this.

**Reconsider if.** The full evaluation shows the agent losing on questions
where the union rerank ranked the needed passage first — which would move the
explanation to ADR-0028's snippet/chunk split or the stopping decision itself.

---

## ADR-0032: Step 8 — agent mode matches plain RAG; recommendation is opt-in, not default

**Status:** accepted — the agent evaluation

**Context.** ADR-0030 found the agent answering worse than plain RAG on two
questions; ADR-0031's fix and 7-question diagnostic suggested it worked.
Neither was a measurement across the full labelled eval set, and a diagnostic
built to confirm one hypothesis isn't a substitute for one built to test
whether the whole feature is worth having.

**Method.** A new, separate harness (`atlas.eval.agent_compare`,
`scripts/evaluate_agent.py`) ran both systems, paired, on all 112 questions of
`eval/datasets/main.jsonl`, against the shipped configuration: dense+rerank
retrieval, `gemini-3.1-flash-lite` routing, `gemini-3.5-flash-lite` answering,
the ADR-0031 union rerank. Quality is scored by whether a question's gold
labels are satisfied by a **cited** chunk, not merely a retrieved one —
stricter than the existing "produced a citation" check.

**Result.** Of 100 answerable questions, **96 scored identically** between the
two systems. Of the 4 that differed: 2 agent wins, 2 agent losses — exactly
even. Overall paired delta: **−0.005, CI [−0.04, 0.03]**, crossing zero; every
per-kind breakdown also crosses zero. Refusal correctness and unverified-quote
counts were identical between systems (12/12 unanswerable correctly refused,
0/100 wrongly refused, 6 vs 6 unverified) — exactly what ADR-0030's shared
answering path was for. Zero errors, zero degraded runs across 100 live agent
executions: the system is robust; the question this ADR answers is about
value, not reliability.

**Cost is not close.** $1.40 per 1000 questions against $0.79 — **1.8×**.
Latency: p50 5132ms vs 3286ms, p95 7050ms vs 4286ms — 56–64% slower.

**The one suggestive result.** `refund-window-by-plan`, a genuine
multi-document question, is where plain RAG missed one of two required
documents (citation-recall 0.5) and the agent's second search reached both
(1.0) — the exact failure mode (Recall@1 = 0.400 on multi-document queries)
that originally motivated the agent. One instance out of five multi-doc
questions, consistent with the hypothesis but not a basis for a claim.

**The two losses were investigated, not left as a number.** Both
`paraphrase`-kind. One is a citation choice, not a miss: the correct document
ranked first in the union rerank, but the answer model cited a different,
genuinely relevant document instead of the labelled one. The other is a real
retrieval-depth gap: the right document filled 3 of 7 evidence slots, but the
specific labelled line sat in a section none of the searches surfaced. Two
examples, not a pattern.

**Decision: agent mode ships opt-in and stays opt-in.** No measured quality
gain justifies 1.8× cost and ~60% more latency on this corpus. The one place
it shows promise (multi-document questions) rests on five instances — too few
to act on, and exactly the kind of claim this project's rule (verify, don't
assume) exists to prevent. Nothing here is a defect; it's a well-built feature
whose benefit, if real, is concentrated in a slice this dataset is too small
to characterise.

**Reconsider if.** A larger, purpose-built multi-document eval set (the
current one has five such questions total) shows a paired delta whose CI
excludes zero.

---

## ADR-0033: CI gates lint and both test suites, and refuses to pass on a skipped suite

**Status:** accepted

**Decision.** Three parallel jobs on push and pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): `ruff check`, the
unit tests, and the integration tests against a `pgvector/pgvector:pg16`
service container — the same image as `docker-compose.yml`, since plain
Postgres would fail at migration 0001 rather than at some subtler later point.

**No secrets are needed, and that's a design property, not a convenience.**
Every test runs against the deterministic offline providers, so the full
suite — ingestion, retrieval, the tool framework, the agent loop, the grounded
answering path — executes with no paid call and no API key. A fork's pull
request gets the same gate with access to nothing: the payoff on the provider
`Protocol` from ADR-0007.

**A skipped integration suite fails the build.** The suite calls `pytest.skip`
when no database is reachable — correct locally, dangerous in CI: a broken
service container yields a suite that skips every test and *exits zero*.
Verified rather than assumed: pointing the DSN at a dead port produces
"41 skipped" and a zero exit status, which GitHub would report green. Two
guards catch this differently: a precheck that must succeed before the suite
can run, and a junit-XML assertion after that catches anything in between.

**mypy is not a gate.** Configured `strict`, it reports 75 pre-existing
errors, mostly `Any` from untyped third-party SDKs. A gate that's never been
green isn't a gate, and loosening the config to pass would be worse than not
running it — it runs as informational output only.

**The evaluation harness does not run in CI.** It calls real models and costs
real money (the four-model answer comparison alone was $0.73). Its outputs are
frozen under `eval/baselines/`, which is what makes them comparable across
time; regenerating on every push would cost money and destroy that property.

**Two bugs found and fixed while building this**, both exposing latent issues
CI was the first thing to actually exercise. `ATLAS_LLM_PROVIDER=fake` — the
setting Atlas's own missing-key error message names — crashed agent mode,
because `FakeLLMProvider` had no `generate_with_tools`; it now issues one
search and finishes, chosen by declaration *shape* rather than by name so
adding a tool can't silently break it. And the integration suite's own
reachability check opened a connection through the pooled `Database` class,
whose connect callback registers pgvector's type — which needs
`CREATE EXTENSION vector` to already exist, and that only happens inside the
migration step the reachability check was supposed to run *before*. It worked
locally by accident, since a dev Postgres volume stays migrated across
sessions; CI's freshly created container was the first genuinely unmigrated
database it ever ran against. Fixed by checking reachability with a bare
connection that asks nothing more than "can I connect."
