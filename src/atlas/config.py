"""Runtime configuration.

Every tunable that affects retrieval quality lives here rather than being
inlined at a call site, so that the eval harness can sweep it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ATLAS_",
        extra="ignore",
    )

    # --- Database ---------------------------------------------------------
    database_url: str = "postgresql://atlas:atlas@localhost:5432/atlas"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10

    # --- LLM --------------------------------------------------------------
    llm_provider: Literal["gemini", "fake"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 60.0
    # 8192, not 2048. The response schema's size scales with the number of
    # citations times quote length: top_k citations each carrying a verbatim
    # quote can far exceed a 2k budget. When it does, the model is truncated
    # mid-JSON, nothing parses, and the whole answer is lost -- observed
    # intermittently during the first real --with-answers run.
    #
    # This is a ceiling, not a reservation: unused budget costs nothing, so a
    # generous value is free insurance against a failure mode that destroys the
    # entire response rather than degrading it.
    llm_max_output_tokens: int = 8192
    # Grounded answering is an extraction task, not a creative one.
    llm_temperature: float = 0.0

    # --- Embeddings -------------------------------------------------------
    embedding_provider: Literal["fastembed", "fake"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: Path = Path(".models")

    # --- Chunking ---------------------------------------------------------
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 32

    # --- Retrieval --------------------------------------------------------
    # Measured, not assumed. Hybrid was implemented and evaluated (E5) and showed
    # no improvement over dense at either k=1 or k=8, so it is not the default.
    # It stays selectable for measurement and for identifier-heavy corpora, where
    # the per-kind breakdown does favour it. See ADR-0018.
    retrieval_mode: Literal["dense", "lexical", "hybrid"] = "dense"
    retrieval_top_k: int = 8
    # How deep each component retrieves before fusion. Fusing two lists of
    # length top_k can only reorder those top_k; the gain comes from a chunk one
    # component ranked 12th and the other ranked 2nd.
    retrieval_candidates: int = 30
    # RRF damping constant. 60 is the value from Cormack et al. (2009) and the
    # usual default; exposed so it can be swept rather than trusted.
    rrf_k: int = 60

    # --- Reranking --------------------------------------------------------
    # On by default: the only configuration measured to beat the dense baseline
    # (nDCG@8 +0.044, paired 95% CI [+0.009, +0.081]). It costs roughly 680ms of
    # extra retrieval latency, which is about +23% on a request whose generation
    # step already takes ~2.8s. Set ATLAS_RERANK_ENABLED=false to return to
    # ~77ms retrieval. See ADR-0020.
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Cross-encoder cost is linear in this number, so it is the main latency
    # dial. Measured in experiment E6.
    rerank_candidates: int = 30

    # --- Evaluation -------------------------------------------------------
    # Eval queries run concurrently. Bounded rather than unlimited because
    # retrieval embeds and reranks on the local CPU, so past a point this trades
    # throughput for contention. Set to 1 to reproduce the old serial behaviour,
    # or lower it if running against a rate-limited free-tier key.
    eval_concurrency: int = 8

    # --- Agent (Phase 4) --------------------------------------------------
    # A separate model role for tool-routing turns. Free-tier quota is scoped
    # per project PER MODEL (quotaId GenerateRequestsPerMinutePerProjectPerModel),
    # verified by exhausting one model and finding another still served on the
    # same key -- so splitting roles adds throughput rather than sharing it.
    #
    # Measured on this project: gemini-3.5-flash 5 RPM, the flash-lite models 15
    # RPM. Those are observations, not guarantees; Google states limits vary by
    # project and standing, and the reported quotaValue was not even stable
    # across our own runs. Sustained throughput is often latency-bound anyway.
    #
    # The final-answer model is deliberately NOT changed: answer quality is the
    # product, and routing is the cheap, high-volume role.
    # gemini-3.1-flash-lite over gemini-3.5-flash-lite: both scored 8/8 on tool
    # selection with 0 unnecessary searches and reached every needed document on
    # the multi-document cases, so quality did not separate them. Latency did --
    # 2.7s mean against 12.4s, with no quota waiting in either run. See
    # scripts/validate_agent_model.py and ADR-0024.
    agent_model: str = "gemini-3.1-flash-lite"

    # --- Ingestion workers ------------------------------------------------
    # Jobs claimed and run concurrently per worker process. Embedding releases
    # the GIL inside ONNX Runtime so these overlap, but beyond the core count
    # they contend rather than speed up.
    worker_concurrency: int = 4
    worker_poll_interval_seconds: float = 1.0
    # How long a claimed job may stay `running` before the reaper assumes its
    # worker died. Must exceed the slowest plausible document; a 200-page PDF
    # embedding on CPU can take minutes.
    worker_lease_seconds: int = 300
    worker_reap_interval_seconds: float = 60.0
    # Attempts per job before it moves to the dead-letter state. Counted at
    # claim time, so a job that crashes its worker still consumes budget.
    ingest_max_attempts: int = 4

    # --- Refusal ----------------------------------------------------------
    # Interim permissive setting, NOT a validated optimum. The 0.60 calibrated
    # in Phase 1 on a 5-document corpus stopped separating once the corpus grew
    # to 33 documents: 10 of 12 unanswerable queries now score above it and the
    # distributions overlap outright. 0.55 is a crash barrier for pathological
    # queries; the model's own sufficient_evidence judgement and citation
    # validation are the real controls. Revisited with numbers in Phase 3+.
    # See ADR-0013 and ADR-0019.
    min_similarity: float = 0.55

    # --- API --------------------------------------------------------------
    # Phase 1 has no auth. Every request is attributed to this tenant so that
    # the tenant-scoped data path is exercised from day one rather than bolted
    # on in Phase 5.
    default_tenant_slug: str = "default"
    max_upload_bytes: int = 20 * 1024 * 1024

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
