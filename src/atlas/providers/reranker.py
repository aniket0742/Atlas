"""Local cross-encoder reranking via fastembed (ONNX Runtime, CPU).

Same constraint as the embedding model: no paid dependency, and re-running an
experiment must be free (ADR-0007). `Xenova/ms-marco-MiniLM-L-6-v2` is ~80MB and
is the default because a larger reranker on CPU spends latency this system has
not yet shown it can afford. `BAAI/bge-reranker-base` is stronger and about 13x
the size; whether that trade is worth it is measured in E6, not assumed.

Reranking is off by default. It is enabled only if it earns its latency against
the measured hybrid baseline.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder


class FastEmbedReranker:
    """Cross-encoder reranker with a lazily loaded ONNX model."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: Path | str = ".models",
        *,
        threads: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(cache_dir)
        self._threads = threads
        self._model: TextCrossEncoder | None = None
        # Model construction is not thread-safe and FastAPI can reach this from
        # several threadpool workers at once.
        self._lock = threading.Lock()

        supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
        if model_name not in supported:
            raise ValueError(
                f"Unsupported reranker {model_name!r}. Available: {', '.join(sorted(supported))}"
            )

    def _ensure_model(self) -> TextCrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    kwargs: dict = {
                        "model_name": self._model_name,
                        "cache_dir": self._cache_dir,
                    }
                    if self._threads is not None:
                        kwargs["threads"] = self._threads
                    self._model = TextCrossEncoder(**kwargs)
        return self._model

    @property
    def model_id(self) -> str:
        return self._model_name

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._ensure_model().rerank(query, passages)]


class FakeReranker:
    """Deterministic offline reranker for tests.

    Scores by lexical overlap between query and passage. Not a model, but it
    produces a stable, query-dependent ordering, which is what the reranking
    plumbing needs to be tested against without a model download.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    @property
    def model_id(self) -> str:
        return "fake-overlap-reranker"

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, len(passages)))
        import re

        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms:
            return [0.0] * len(passages)
        scores = []
        for passage in passages:
            words = re.findall(r"[a-z0-9]+", passage.lower())
            if not words:
                scores.append(0.0)
                continue
            hits = sum(1 for w in words if w in terms)
            scores.append(hits / len(words) ** 0.5)
        return scores
