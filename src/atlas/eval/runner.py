"""Evaluation runner.

Produces a JSON report containing the metrics *and* the configuration that
produced them. That pairing is the point: a retrieval number without the
chunking parameters, embedding model and top_k that produced it cannot be
compared against anything, and a directory of such numbers is worse than none
because it invites false comparisons.

Two modes:

  * retrieval-only (default) -- no LLM calls, so it is free, fast, and can be
    run on every change. This is what Phase 2's hybrid-vs-dense comparison uses.
  * with-answers -- also generates answers to measure refusal behaviour and
    citation validity. Costs one model call per query, so it is run
    deliberately, not on every change.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.answer.service import AnswerService
from atlas.config import Settings
from atlas.eval import metrics
from atlas.eval.dataset import EvalQuery, load
from atlas.retrieval.service import Retriever


@dataclass(slots=True)
class QueryReport:
    query_id: str
    question: str
    answerable: bool
    scores: dict[str, Any] | None
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    # Populated only in with-answers mode.
    refused: bool | None = None
    refusal_reason: str | None = None
    citation_count: int | None = None
    unverified_quotes: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None


class EvalRunner:
    def __init__(
        self,
        retriever: Retriever,
        settings: Settings,
        answerer: AnswerService | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings
        self._answerer = answerer

    async def run(
        self,
        tenant_id: uuid.UUID,
        dataset_path: Path,
        *,
        k: int | None = None,
        with_answers: bool = False,
        label: str | None = None,
    ) -> dict[str, Any]:
        queries = load(dataset_path)
        top_k = k or self._settings.retrieval_top_k

        if with_answers and self._answerer is None:
            raise ValueError("with_answers requires an AnswerService")

        reports: list[QueryReport] = []
        started = time.perf_counter()

        for query in queries:
            report = await self._run_one(tenant_id, query, top_k, with_answers)
            reports.append(report)

        return self._assemble(
            reports=reports,
            queries=queries,
            top_k=top_k,
            dataset_path=dataset_path,
            with_answers=with_answers,
            label=label,
            wall_seconds=time.perf_counter() - started,
        )

    async def _run_one(
        self, tenant_id: uuid.UUID, query: EvalQuery, top_k: int, with_answers: bool
    ) -> QueryReport:
        # Retrieval metrics are computed on the *unfiltered* candidates. The
        # similarity floor is an answering policy, not a retrieval property, and
        # mixing them would make a threshold change look like a retrieval
        # regression.
        result = await self._retriever.retrieve(
            tenant_id, query.question, top_k=top_k, min_similarity=0.0
        )
        ranked = result.candidates

        retrieved_rows = [
            {
                "rank": i,
                "document": chunk.document_external_id,
                "score": round(chunk.score, 4),
                "heading_path": chunk.heading_path,
            }
            for i, chunk in enumerate(ranked)
        ]

        scores: dict[str, Any] | None = None
        if query.answerable:
            relevant_positions: list[int] = []
            matched_labels: set[int] = set()
            for position, chunk in enumerate(ranked):
                hit = False
                for label_index, gold in enumerate(query.labels):
                    if gold.matches(chunk.document_external_id, chunk.text):
                        matched_labels.add(label_index)
                        hit = True
                if hit:
                    relevant_positions.append(position)

            scores = asdict(
                metrics.score_query(
                    query_id=query.id,
                    relevant_positions=relevant_positions,
                    matched_labels=len(matched_labels),
                    total_labels=len(query.labels),
                    retrieved=len(ranked),
                    k=top_k,
                )
            )

        report = QueryReport(
            query_id=query.id,
            question=query.question,
            answerable=query.answerable,
            scores=scores,
            retrieved=retrieved_rows,
        )

        if with_answers and self._answerer is not None:
            t0 = time.perf_counter()
            answer = await self._answerer.answer(tenant_id, query.question, top_k=top_k)
            report.latency_ms = (time.perf_counter() - t0) * 1000
            report.refused = answer.refused
            report.refusal_reason = answer.refusal_reason
            report.citation_count = len(answer.citations)
            report.unverified_quotes = sum(1 for c in answer.citations if not c.quote_verified)
            report.total_tokens = answer.usage.total_tokens

        return report

    def _assemble(
        self,
        *,
        reports: list[QueryReport],
        queries: list[EvalQuery],
        top_k: int,
        dataset_path: Path,
        with_answers: bool,
        label: str | None,
        wall_seconds: float,
    ) -> dict[str, Any]:
        answerable = [r for r in reports if r.answerable and r.scores]

        def column(name: str) -> list[float]:
            return [float(r.scores[name]) for r in answerable if r.scores]

        summary: dict[str, Any] = {}
        for name in ("recall_at_k", "precision_at_k", "reciprocal_rank", "ndcg_at_k"):
            values = column(name)
            low, high = metrics.bootstrap_ci(values)
            key = "mrr" if name == "reciprocal_rank" else name
            summary[key] = {
                "mean": round(metrics.mean(values), 4),
                "ci95": [round(low, 4), round(high, 4)],
            }

        if with_answers:
            unanswerable = [r for r in reports if not r.answerable]
            answerable_answered = [r for r in reports if r.answerable and r.refused is not None]
            # The metric that matters most for groundedness: when the corpus
            # cannot answer, does the system say so instead of inventing?
            summary["refusal"] = {
                "unanswerable_queries": len(unanswerable),
                "correctly_refused": sum(1 for r in unanswerable if r.refused),
                "answerable_queries": len(answerable_answered),
                "incorrectly_refused": sum(1 for r in answerable_answered if r.refused),
            }
            cited = [r for r in answerable_answered if not r.refused]
            summary["citations"] = {
                "answers_with_citations": sum(1 for r in cited if (r.citation_count or 0) > 0),
                "answers_scored": len(cited),
                "unverified_quotes": sum(r.unverified_quotes or 0 for r in cited),
            }
            latencies = [r.latency_ms for r in reports if r.latency_ms is not None]
            if latencies:
                latencies.sort()
                summary["latency_ms"] = {
                    "p50": round(latencies[len(latencies) // 2], 1),
                    "p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1),
                    "max": round(latencies[-1], 1),
                }
            summary["tokens"] = {
                "total": sum(r.total_tokens or 0 for r in reports),
            }

        return {
            "label": label,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": {
                "path": str(dataset_path),
                "queries": len(queries),
                "answerable": sum(1 for q in queries if q.answerable),
                "unanswerable": sum(1 for q in queries if not q.answerable),
            },
            # Everything that could change the numbers. A report without this is
            # not comparable to any other report.
            "config": {
                "k": top_k,
                "embedding_model": self._retriever.embedder.model_id,
                "llm_model": self._settings.llm_model if with_answers else None,
                "chunk_target_tokens": self._settings.chunk_target_tokens,
                "chunk_overlap_tokens": self._settings.chunk_overlap_tokens,
                "chunk_min_tokens": self._settings.chunk_min_tokens,
                "min_similarity": self._settings.min_similarity,
                "retrieval": "dense",
                "python": platform.python_version(),
            },
            "summary": summary,
            "wall_seconds": round(wall_seconds, 2),
            "queries": [asdict(r) for r in reports],
        }


def write_report(report: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{report.get('label') or 'run'}.json"
    path = directory / name
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
