"""Local embedding provider backed by fastembed (ONNX Runtime, CPU).

Why local and not a hosted embedding API:

  * Cost. Retrieval experiments in Phase 2 mean re-embedding the whole corpus
    repeatedly. Paying per token for that turns "measure whether reranking
    helps" into a budget decision, which is precisely the trap the spec warns
    about. Local embeddings make re-indexing free, so experiments are cheap.
  * Determinism. The same text produces the same vector forever, so an eval run
    from last week is comparable to one from today. Hosted embedding models are
    versioned and get deprecated underneath you.
  * No network in the ingest hot path.

The trade-off is quality: a 33M-parameter model is weaker than a large hosted
embedding model. That is measured, not assumed -- the eval harness can swap the
provider and report the delta.

fastembed rather than sentence-transformers: it runs on ONNX Runtime instead of
pulling in PyTorch, which keeps the eventual worker image small. It exposes the
tokenizer, which chunking needs to avoid silent truncation.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastembed import TextEmbedding


class FastEmbedProvider:
    """Embeds with a locally-cached ONNX model.

    The model is loaded lazily on first use: importing this module must not
    trigger a 67MB download, because the API process imports it at startup and
    the CLI imports it to print help.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | str = ".models",
        *,
        threads: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(cache_dir)
        self._threads = threads
        self._model: TextEmbedding | None = None
        # Model load is not thread-safe and FastAPI may call this from several
        # threadpool workers at once.
        self._lock = threading.Lock()

        meta = self._metadata(model_name)
        self._dimensions = int(meta["dim"])
        # fastembed reports the truncation limit only in prose. Rather than
        # parse a description string, take the family default and clamp
        # conservatively -- chunking treats this as a ceiling, so erring low
        # costs a little recall density and erring high causes silent
        # truncation, which is the failure that actually hurts.
        self._max_tokens = 512

    @staticmethod
    def _metadata(model_name: str) -> dict:
        for model in TextEmbedding.list_supported_models():
            if model["model"] == model_name:
                return model
        supported = ", ".join(sorted(m["model"] for m in TextEmbedding.list_supported_models()))
        raise ValueError(f"Unsupported embedding model {model_name!r}. Available: {supported}")

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    kwargs = {"model_name": self._model_name, "cache_dir": self._cache_dir}
                    if self._threads is not None:
                        kwargs["threads"] = self._threads
                    self._model = TextEmbedding(**kwargs)
        return self._model

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """Token count as the *model* sees it, including its special tokens.

        fastembed's token_count returns the sum over all supplied texts, so it is
        called with a single string. The count includes [CLS]/[SEP], which makes
        it two tokens pessimistic per call -- the right direction to be wrong in,
        since this is used as a budget ceiling.
        """
        if not text:
            return 0
        return int(self._ensure_model().token_count(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        return [v.tolist() for v in model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        # query_embed applies whatever query-side prefix the model family
        # expects. For BGE v1.5 the prefix matters less than it did for v1, but
        # routing queries and passages through different calls is the correct
        # shape regardless of which model is configured.
        return next(iter(model.query_embed([text]))).tolist()
