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
    llm_max_output_tokens: int = 2048
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
