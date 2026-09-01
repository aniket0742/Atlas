"""Provider construction from settings.

Providers are cached per-process. The embedding model holds an ONNX session and
several hundred MB of resident memory once warm; constructing one per request
would be a straightforward way to run out of RAM.
"""

from __future__ import annotations

from functools import lru_cache

from atlas.config import Settings, get_settings
from atlas.providers.base import EmbeddingProvider, LLMProvider


@lru_cache(maxsize=4)
def _build_embedder(provider: str, model: str, cache_dir: str) -> EmbeddingProvider:
    if provider == "fake":
        from atlas.providers.fake import FakeEmbeddingProvider

        return FakeEmbeddingProvider()

    from atlas.providers.local_embeddings import FastEmbedProvider

    return FastEmbedProvider(model_name=model, cache_dir=cache_dir)


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    return _build_embedder(s.embedding_provider, s.embedding_model, str(s.model_cache_dir))


def get_llm(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.llm_provider == "fake":
        from atlas.providers.fake import FakeLLMProvider

        return FakeLLMProvider()

    from atlas.providers.gemini import GeminiProvider

    return GeminiProvider(
        api_key=s.gemini_api_key,
        model=s.llm_model,
        timeout_seconds=s.llm_timeout_seconds,
        max_output_tokens=s.llm_max_output_tokens,
        temperature=s.llm_temperature,
    )
