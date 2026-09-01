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
    retrieval_top_k: int = 8
    # Calibrated in Phase 1 experiment E2, not guessed. See ADR-0013 and
    # scripts/calibrate_floor.py.
    min_similarity: float = 0.60

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
