"""Experiment E2: calibrate the retrieval similarity floor.

The floor decides when Atlas refuses *before* calling the model. Too low and an
unanswerable question still gets a full evidence set, which is where fabrication
comes from. Too high and correct answers get refused.

The key property that makes this measurable for free: the floor acts on retrieval
scores, so its effect is computable without a single LLM call. For each candidate
threshold we can ask directly:

  * unanswerable queries where *every* chunk falls below the floor
    -> guaranteed correct refusal, with no model call and no tokens spent
  * answerable queries where every chunk falls below the floor
    -> guaranteed wrong refusal
  * answerable queries where every *relevant* chunk falls below the floor
    -> the model is handed only irrelevant evidence, which is the setup for a
       confident wrong answer

Run:  python scripts/calibrate_floor.py [dataset.jsonl]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atlas  # noqa: F401,E402  (installs the Windows event loop policy)
from atlas.config import get_settings  # noqa: E402
from atlas.db import repository as repo  # noqa: E402
from atlas.db.pool import Database  # noqa: E402
from atlas.eval.dataset import load  # noqa: E402
from atlas.providers.factory import get_embedder  # noqa: E402
from atlas.retrieval.service import Retriever  # noqa: E402

DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/datasets/main.jsonl")
TOP_K = 8
FLOORS = [round(0.30 + 0.02 * i, 2) for i in range(21)]  # 0.30 .. 0.70


async def main() -> None:
    settings = get_settings()
    queries = load(DATASET)

    db = Database(settings.database_url)
    await db.open()
    try:
        async with db.transaction() as conn:
            tenant = await repo.ensure_tenant(conn, settings.default_tenant_slug)

        retriever = Retriever(db, get_embedder(settings), settings)

        # Retrieve once per query with no floor; the sweep is then pure arithmetic.
        best_overall: dict[str, float] = {}
        best_relevant: dict[str, float] = {}

        for query in queries:
            result = await retriever.retrieve(
                tenant, query.question, top_k=TOP_K, min_similarity=0.0
            )
            ranked = result.candidates
            best_overall[query.id] = ranked[0].score if ranked else 0.0
            relevant = [
                c.score
                for c in ranked
                if any(g.matches(c.document_external_id, c.text) for g in query.labels)
            ]
            best_relevant[query.id] = max(relevant) if relevant else 0.0
    finally:
        await db.close()

    answerable = [q for q in queries if q.answerable]
    unanswerable = [q for q in queries if not q.answerable]

    print("=" * 78)
    print("Score distributions (cosine similarity, bge-small-en-v1.5)")
    print("=" * 78)

    print("\nANSWERABLE - score of the best *relevant* chunk")
    print("  (the floor must stay BELOW these or the answer is lost)")
    rel = sorted((best_relevant[q.id], q.id) for q in answerable)
    for score, qid in rel:
        print(f"    {score:.4f}  {qid}")
    print(f"  min={rel[0][0]:.4f}  max={rel[-1][0]:.4f}")

    print("\nUNANSWERABLE - score of the best chunk retrieved")
    print("  (the floor must stay ABOVE these to refuse without a model call)")
    un = sorted(((best_overall[q.id], q.id) for q in unanswerable), reverse=True)
    for score, qid in un:
        print(f"    {score:.4f}  {qid}")
    print(f"  min={un[-1][0]:.4f}  max={un[0][0]:.4f}")

    separable = rel[0][0] > un[0][0]
    print("\n" + "-" * 78)
    print(f"Lowest answerable relevant score : {rel[0][0]:.4f}  ({rel[0][1]})")
    print(f"Highest unanswerable score       : {un[0][0]:.4f}  ({un[0][1]})")
    print(f"Linearly separable by a floor?   : {separable}")
    if not separable:
        print("  -> No single threshold both keeps every answer and blocks every")
        print("     unanswerable question. The floor cannot be the only control;")
        print("     the model's own sufficient_evidence judgement carries the rest.")

    print("\n" + "=" * 78)
    print("Floor sweep")
    print("=" * 78)
    print(f"{'floor':>6} {'auto-refused':>14} {'wrongly refused':>17} {'evidence lost':>15}")
    print(f"{'':>6} {'(of ' + str(len(unanswerable)) + ' unansw.)':>14} "
          f"{'(of ' + str(len(answerable)) + ' answerable)':>17} {'(answerable)':>15}")
    print("-" * 78)

    for floor in FLOORS:
        auto_refused = sum(1 for q in unanswerable if best_overall[q.id] < floor)
        wrongly_refused = sum(1 for q in answerable if best_overall[q.id] < floor)
        evidence_lost = sum(1 for q in answerable if best_relevant[q.id] < floor)
        flag = ""
        if wrongly_refused == 0 and evidence_lost == 0 and auto_refused > 0:
            flag = "  <- safe and useful"
        elif evidence_lost > 0 and wrongly_refused == 0:
            flag = "  <- starts losing evidence"
        print(f"{floor:>6.2f} {auto_refused:>14} {wrongly_refused:>17} "
              f"{evidence_lost:>15}{flag}")

    print("\nInterpretation: the best floor is the highest value that still shows")
    print("0 wrongly refused and 0 evidence lost. Anything above that trades a")
    print("real answer for a refusal.")


if __name__ == "__main__":
    asyncio.run(main())
