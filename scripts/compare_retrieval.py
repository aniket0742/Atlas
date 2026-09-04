"""Experiments E5 and E6: compare retrieval configurations.

E5 — dense vs lexical vs hybrid.
E6 — hybrid vs hybrid + cross-encoder reranking, including latency.

Retrieval-only: no LLM calls, so this is free to re-run and costs no quota.

## Decision rule, and a correction to it

The rule registered before Step 4 was "adopt only if the candidate beats the
incumbent at k=1 with non-overlapping 95% confidence intervals". That rule is
statistically wrong, and it was wrong before any number existed.

Both configurations are evaluated on the *same* queries, so the comparison is
paired. Independent intervals ignore the pairing and are dominated by variance
*between queries* -- some questions are simply harder -- rather than by the
difference between configurations. Two configurations can differ on nearly every
query and still produce overlapping intervals.

The corrected rule: adopt only if the **paired** bootstrap interval on the
per-query difference excludes zero. This is applied symmetrically to every
configuration, including where it makes a candidate's *deficit* significant. The
original unpaired comparison is still printed alongside so the change in method
is visible rather than quietly swapped in.

Reranking additionally has to justify its latency. A quality gain costing
seconds per query is a bad trade for an interactive system.

Run:  python scripts/compare_retrieval.py [dataset.jsonl]
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atlas  # noqa: F401,E402  (installs the Windows event loop policy)
from atlas.config import get_settings  # noqa: E402
from atlas.db import repository as repo  # noqa: E402
from atlas.db.pool import Database  # noqa: E402
from atlas.eval import metrics  # noqa: E402
from atlas.eval.dataset import load  # noqa: E402
from atlas.eval.runner import EvalRunner, write_report  # noqa: E402
from atlas.providers.factory import get_embedder  # noqa: E402
from atlas.providers.reranker import FastEmbedReranker  # noqa: E402
from atlas.retrieval.service import Retriever  # noqa: E402

DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/datasets/main.jsonl")
DEPTHS = (1, 3, 5, 8)
CONFIGS = [
    ("dense", "dense", False),
    ("lexical", "lexical", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]
LATENCY_SAMPLE = 20


def per_query(report: dict, metric: str) -> dict[str, float]:
    """Query id -> metric value, so configurations can be aligned for pairing."""
    return {
        q["query_id"]: float(q["scores"][metric])
        for q in report["queries"]
        if q.get("scores")
    }


def aligned(a: dict, b: dict, metric: str) -> tuple[list[float], list[float]]:
    ids = sorted(set(per_query(a, metric)) & set(per_query(b, metric)))
    left, right = per_query(a, metric), per_query(b, metric)
    return [left[i] for i in ids], [right[i] for i in ids]


def overlaps(a: dict, b: dict) -> bool:
    return a["ci95"][0] <= b["ci95"][1] and b["ci95"][0] <= a["ci95"][1]


async def measure_latency(retriever, tenant, questions, mode, rerank) -> tuple[float, float]:
    """Median and p95 wall time for a single retrieval call, in ms."""
    # One warm-up so a lazy model load is not counted as query latency.
    await retriever.retrieve(tenant, questions[0], top_k=8, mode=mode, rerank=rerank)
    samples = []
    for question in questions:
        started = time.perf_counter()
        await retriever.retrieve(tenant, question, top_k=8, mode=mode, rerank=rerank)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return samples[len(samples) // 2], samples[min(len(samples) - 1, int(len(samples) * 0.95))]


async def main() -> None:
    settings = get_settings()
    queries = load(DATASET)
    db = Database(settings.database_url)
    await db.open()
    try:
        async with db.transaction() as conn:
            tenant = await repo.ensure_tenant(conn, settings.default_tenant_slug)

        reranker = FastEmbedReranker(
            model_name=settings.rerank_model, cache_dir=str(settings.model_cache_dir)
        )
        retriever = Retriever(db, get_embedder(settings), settings, reranker=reranker)
        runner = EvalRunner(retriever, settings)

        reports: dict[tuple[str, int], dict] = {}
        for name, mode, rerank in CONFIGS:
            for k in DEPTHS:
                reports[(name, k)] = await runner.run(
                    tenant, DATASET, k=k, mode=mode, rerank=rerank, label=f"{name}-k{k}"
                )
                if k in (1, 8):
                    write_report(reports[(name, k)], Path("eval/results"))

        n = reports[("dense", 1)]["dataset"]["answerable"]
        print("=" * 100)
        print(f"E5 / E6 — retrieval comparison on {DATASET}")
        print(f"{n} answerable queries, {len(queries) - n} unanswerable")
        print("=" * 100)

        for metric in ("recall_at_k", "mrr", "ndcg_at_k"):
            print(f"\n{metric}  (mean [unpaired 95% CI])")
            print("  " + f"{'config':<16}" + "".join(f"{'k=' + str(k):>22}" for k in DEPTHS))
            print("  " + "-" * (16 + 22 * len(DEPTHS)))
            for name, _, _ in CONFIGS:
                row = f"  {name:<16}"
                for k in DEPTHS:
                    e = reports[(name, k)]["summary"][metric]
                    row += f"{e['mean']:.3f} [{e['ci95'][0]:.2f}-{e['ci95'][1]:.2f}]".rjust(22)
                print(row)

        print("\n" + "=" * 100)
        print("Paired comparison against dense at k=1  (the corrected test)")
        print("=" * 100)
        print(f"  {'config':<16}{'metric':<14}{'delta':>9}{'paired 95% CI':>22}   verdict")
        print("  " + "-" * 96)
        base_report = reports[("dense", 1)]
        for name, _, _ in CONFIGS[1:]:
            cand_report = reports[(name, 1)]
            for metric_key, field in (("recall_at_k", "recall_at_k"), ("mrr", "reciprocal_rank")):
                left, right = aligned(base_report, cand_report, field)
                delta, (low, high) = metrics.paired_bootstrap_delta(left, right)
                significant = low > 0 or high < 0
                verdict = (
                    ("BETTER" if delta > 0 else "WORSE") if significant else "no difference"
                )
                unpaired = (
                    "unpaired: overlap"
                    if overlaps(
                        base_report["summary"][metric_key], cand_report["summary"][metric_key]
                    )
                    else "unpaired: separate"
                )
                print(
                    f"  {name:<16}{metric_key:<14}{delta:>+9.4f}"
                    f"{f'[{low:+.4f}, {high:+.4f}]':>22}   {verdict}  ({unpaired})"
                )

        print("\n" + "=" * 100)
        print("Recall@1 by query kind — where the difference actually lives")
        print("=" * 100)
        kinds = sorted(reports[("dense", 1)]["summary"]["by_kind"])
        print("  " + f"{'kind':<14}{'n':>4}" + "".join(f"{c[0]:>16}" for c in CONFIGS))
        print("  " + "-" * (18 + 16 * len(CONFIGS)))
        for kind in kinds:
            row = f"  {kind:<14}{reports[('dense', 1)]['summary']['by_kind'][kind]['n']:>4}"
            for name, _, _ in CONFIGS:
                row += f"{reports[(name, 1)]['summary']['by_kind'][kind]['recall_at_k']:>16.3f}"
            print(row)

        print("\n" + "=" * 100)
        print(f"Latency — single retrieval call, {LATENCY_SAMPLE} queries, no LLM")
        print("=" * 100)
        sample = [q.question for q in queries[:LATENCY_SAMPLE]]
        print(f"  {'config':<16}{'p50':>10}{'p95':>10}")
        print("  " + "-" * 36)
        for name, mode, rerank in CONFIGS:
            p50, p95 = await measure_latency(retriever, tenant, sample, mode, rerank)
            print(f"  {name:<16}{p50:>9.1f}ms{p95:>9.1f}ms")
        print(
            f"\n  Reranking cost is linear in rerank_candidates (currently "
            f"{settings.rerank_candidates})."
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
