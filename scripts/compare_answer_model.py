"""Which model should write the final, cited answer?

The routing benchmark (scripts/validate_agent_model.py) says nothing about this.
Routing is a classification problem over one tool and every candidate saturated
it. Answering is different work: read supplied evidence, produce a claim, cite it
verbatim, and refuse when the evidence does not support an answer.

This holds retrieval completely constant and varies only the answer model, then
scores the properties this system actually promises:

  * refusal correctness in BOTH directions -- refusing the unanswerable is easy
    if you refuse everything, so wrongly-refused answerable queries matter too
  * citation coverage -- did non-refused answers cite anything resolvable
  * unverified quotes -- citations resolving to a real chunk whose quote was not
    found verbatim in it; paraphrase at best, fabrication at worst
  * latency and real token cost

This is the `--with-answers` eval that had never been run. Until now every
answer-quality claim in this project was unmeasured.

Run:  python scripts/compare_answer_model.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atlas  # noqa: F401,E402
from atlas.answer.service import AnswerService  # noqa: E402
from atlas.config import get_settings  # noqa: E402
from atlas.db import repository as repo  # noqa: E402
from atlas.db.pool import Database  # noqa: E402
from atlas.eval.runner import EvalRunner, write_report  # noqa: E402
from atlas.providers.factory import get_embedder, get_reranker  # noqa: E402
from atlas.providers.gemini import GeminiProvider  # noqa: E402
from atlas.retrieval.service import Retriever  # noqa: E402

DATASET = Path("eval/datasets/main.jsonl")

# USD per 1M tokens, paid tier. Flash promo pricing doubles 2027-01-01.
PRICING = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
}
CANDIDATES = list(PRICING)


async def main() -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    await db.open()
    try:
        async with db.transaction() as conn:
            tenant = await repo.ensure_tenant(conn, settings.default_tenant_slug)

        # Retrieval is built once and shared, so every model sees identical
        # evidence. Anything that differs between rows is the model.
        retriever = Retriever(db, get_embedder(settings), settings, reranker=get_reranker(settings))

        rows = []
        for model in CANDIDATES:
            llm = GeminiProvider(
                api_key=settings.gemini_api_key,
                model=model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_output_tokens=settings.llm_max_output_tokens,
                temperature=settings.llm_temperature,
            )
            runner = EvalRunner(retriever, settings, AnswerService(db, retriever, llm, settings))
            print(f"running {model} ...", flush=True)
            report = await runner.run(
                tenant, DATASET, with_answers=True, label=f"answers-{model}"
            )
            write_report(report, Path("eval/results"))
            # Quality above ran concurrently, so its per-query latency includes
            # contention and is not what a single user would experience. Sample
            # a few queries serially for a latency figure that means something.
            sample = [q["question"] for q in report["queries"][:6]]
            service = AnswerService(db, retriever, llm, settings)
            timings = []
            for question in sample:
                t0 = time.perf_counter()
                await service.answer(tenant, question)
                timings.append((time.perf_counter() - t0) * 1000)
            timings.sort()
            rows.append((model, report, timings[len(timings) // 2]))
            s = report["summary"]
            print(f"  done: refusal {s['refusal']}, citations {s['citations']}", flush=True)

        print("\n" + "=" * 104)
        print("Answer model comparison -- retrieval held constant (dense + rerank)")
        print(f"{DATASET}, {rows[0][1]['dataset']['queries']} queries, "
              f"concurrency={settings.eval_concurrency}")
        print("=" * 104)
        header = (f"{'model':<24}{'refused OK':>12}{'wrong refuse':>14}"
                  f"{'cited':>10}{'unverified':>12}{'p50 ms':>9}{'$/1k Q':>9}")
        print(header)
        print("-" * len(header))
        for model, report, serial_p50 in rows:
            s = report["summary"]
            r, c = s["refusal"], s["citations"]
            lat = serial_p50
            # Token totals are summed across every answered query in the run.
            total_tok = s["tokens"]["total"]
            n = report["dataset"]["queries"]
            # Split unavailable in the summary; approximate with the observed
            # ~85/15 input:output ratio measured on this pipeline.
            in_price, out_price = PRICING[model]
            cost = (total_tok * 0.85 * in_price + total_tok * 0.15 * out_price) / 1e6 / n
            refused = f"{r['correctly_refused']}/{r['unanswerable_queries']}"
            cited = f"{c['answers_with_citations']}/{c['answers_scored']}"
            print(f"{model:<24}{refused:>12}{r['incorrectly_refused']:>14}"
                  f"{cited:>10}{c['unverified_quotes']:>12}"
                  f"{lat:>9.0f}{cost * 1000:>9.2f}")
        print("\nrefused OK   = unanswerable queries correctly refused (higher better)")
        print("wrong refuse = answerable queries wrongly refused (lower better)")
        print("unverified   = citations whose quote was not found verbatim (lower better)")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
