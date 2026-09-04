"""Provider construction from settings.

Providers are cached per-process. The embedding model holds an ONNX session and
several hundred MB of resident memory once warm; constructing one per request
would be a straightforward way to run out of RAM.
"""

from __future__ import annotations

from functools import lru_cache

from atlas.config import Settings, get_settings
from atlas.providers.base import EmbeddingProvider, LLMProvider, RerankProvider


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


def _build_llm(s: Settings, model: str) -> LLMProvider:
    if s.llm_provider == "fake":
        from atlas.providers.fake import FakeLLMProvider

        return FakeLLMProvider()

    from atlas.providers.gemini import GeminiProvider

    return GeminiProvider(
        api_key=s.gemini_api_key,
        model=model,
        timeout_seconds=s.llm_timeout_seconds,
        max_output_tokens=s.llm_max_output_tokens,
        temperature=s.llm_temperature,
    )


def get_llm(settings: Settings | None = None) -> LLMProvider:
    """The model that writes the final, cited answer."""
    s = settings or get_settings()
    return _build_llm(s, s.llm_model)


def get_agent_llm(settings: Settings | None = None) -> LLMProvider:
    """The model that decides which tools to call.

    A distinct role, not merely a distinct setting. Routing is high-volume and
    cheap; answering is low-volume and is where quality shows. They also draw on
    separate free-tier quotas, so splitting them adds headroom instead of
    competing for one budget.

    One API key covers both: quota is scoped per project per model, verified
    empirically rather than assumed.
    """
    s = settings or get_settings()
    return _build_llm(s, s.agent_model)


@lru_cache(maxsize=4)
def _build_reranker(provider: str, model: str, cache_dir: str) -> RerankProvider:
    if provider == "fake":
        from atlas.providers.reranker import FakeReranker

        return FakeReranker()

    from atlas.providers.reranker import FastEmbedReranker

    return FastEmbedReranker(model_name=model, cache_dir=cache_dir)


def get_reranker(settings: Settings | None = None) -> RerankProvider | None:
    """Build the reranker, or None when reranking is disabled.

    Returning None rather than a no-op keeps "reranking is off" visible in the
    retrieval result and in the health endpoint, instead of silently running a
    stage that does nothing.
    """
    s = settings or get_settings()
    if not s.rerank_enabled:
        return None
    provider = "fake" if s.embedding_provider == "fake" else "fastembed"
    return _build_reranker(provider, s.rerank_model, str(s.model_cache_dir))
