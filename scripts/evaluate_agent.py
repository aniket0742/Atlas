"""Step 8: does agent mode beat plain RAG, and does it justify its cost?

Runs both systems on every question in the labelled eval set, paired, and scores
both by whether their *citations* satisfy the question's gold labels -- not
merely whether they retrieved the right chunk (see
`atlas.eval.agent_compare` for why that distinction matters).

Every setting here matches the shipped default: dense retrieval, reranking on,
`gemini-3.1-flash-lite` routing, `gemini-3.5-flash-lite` answering, the union
rerank from ADR-0031. This is not a search over configurations -- it is a
measurement of the configuration this project is about to ship.

Run:  python scripts/evaluate_agent.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atlas  # noqa: F401,E402
from atlas.agent.knowledge_base import SearchKnowledgeBaseTool  # noqa: E402
from atlas.agent.loop import AgentPlanner  # noqa: E402
from atlas.agent.service import AgentAnswerService  # noqa: E402
from atlas.agent.tools import ToolContext, ToolRegistry  # noqa: E402
from atlas.answer.service import AnswerService  # noqa: E402
from atlas.config import get_settings  # noqa: E402
from atlas.db import repository as repo  # noqa: E402
from atlas.db.pool import Database  # noqa: E402
from atlas.eval.agent_compare import run_paired, summarize, to_json  # noqa: E402
from atlas.eval.runner import write_report  # noqa: E402
from atlas.providers.factory import get_agent_llm, get_embedder, get_llm, get_reranker  # noqa: E402
from atlas.retrieval.service import Retriever  # noqa: E402

DATASET = Path("eval/datasets/main.jsonl")

# USD per 1M tokens, paid tier -- same table as scripts/compare_answer_model.py
# and scripts/validate_agent_model.py. Kept local rather than imported because
# these scripts are one-shot tools, not a shared library; duplication here is
# cheaper than a shared import for three lines that rarely change.
PRICING = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


def cost(input_tok: int, output_tok: int, model: str, n: int) -> float:
    in_price, out_price = PRICING[model]
    return (input_tok * in_price + output_tok * out_price) / 1e6 / n


async def main() -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    await db.open()
    try:
        async with db.transaction() as conn:
            tenant = await repo.ensure_tenant(conn, settings.default_tenant_slug)

        # One retriever, one reranker instance, shared by both systems -- so
        # nothing about retrieval itself differs between them; only how each
        # system decides what to retrieve.
        reranker = get_reranker(settings)
        retriever = Retriever(db, get_embedder(settings), settings, reranker=reranker)
        answerer = AnswerService(db, retriever, get_llm(settings), settings)

        registry = ToolRegistry([SearchKnowledgeBaseTool(retriever)])
        planner = AgentPlanner(get_agent_llm(settings), registry, settings)
        agent = AgentAnswerService(planner, answerer, settings, reranker=reranker)

        print(f"running plain vs agent on {DATASET} ...", flush=True)
        reports = await run_paired(
            tenant,
            DATASET,
            answerer,
            agent,
            settings,
            context_factory=lambda tenant_id: ToolContext(tenant_id=tenant_id),
        )
        summary = summarize(reports)

        config = {
            "dataset": str(DATASET),
            "queries": len(reports),
            "answer_model": settings.llm_model,
            "agent_model": settings.agent_model,
            "retrieval_mode": settings.retrieval_mode,
            "rerank_enabled": settings.rerank_enabled,
            "agent_union_rerank": settings.agent_union_rerank,
            "agent_max_evidence": settings.agent_max_evidence,
            "agent_max_iterations": settings.agent_max_iterations,
            "agent_max_tool_calls": settings.agent_max_tool_calls,
            "eval_concurrency": settings.eval_concurrency,
        }
        full = to_json(reports, summary, config=config)
        path = write_report({**full, "label": "agent-vs-plain"}, Path("eval/results"))
        print(f"wrote {path}", flush=True)

        # -- console summary -----------------------------------------------
        n = summary["overall"]["n"]
        print("\n" + "=" * 96)
        print(f"Plain RAG vs agent mode -- {len(reports)} questions, {n} answerable, paired")
        print("=" * 96)

        pr = summary["overall"]["plain_citation_recall"]
        ar = summary["overall"]["agent_citation_recall"]
        print(f"citation recall   plain {pr['mean']:.3f} {pr['ci95']}   "
              f"agent {ar['mean']:.3f} {ar['ci95']}")
        if "paired_delta_agent_minus_plain" in summary["overall"]:
            d = summary["overall"]["paired_delta_agent_minus_plain"]
            print(f"paired delta (agent - plain): {d['mean']:+.3f}  ci95 {d['ci95']}")

        print("\nby kind:")
        for kind, block in summary["by_kind"].items():
            pr, ar = block["plain_citation_recall"], block["agent_citation_recall"]
            print(f"  {kind:<14} n={block['n']:<4} "
                  f"plain {pr['mean']:.3f}  agent {ar['mean']:.3f}")

        rp, ra = summary["refusal"]["plain"], summary["refusal"]["agent"]
        print(f"\nrefusal (unanswerable correctly refused): "
              f"plain {rp['correctly_refused']}/{rp['unanswerable_queries']}  "
              f"agent {ra['correctly_refused']}/{ra['unanswerable_queries']}")
        print(f"refusal (answerable wrongly refused):     "
              f"plain {rp['incorrectly_refused']}  agent {ra['incorrectly_refused']}")
        print(f"unverified quotes: plain {summary['unverified_quotes']['plain']}  "
              f"agent {summary['unverified_quotes']['agent']}")
        print(f"errors: plain {summary['errors']['plain']}  agent {summary['errors']['agent']}")

        beh = summary["agent_behaviour"]
        print(f"\nagent behaviour: degraded {beh['degraded_count']}/{n} "
              f"({beh['degraded_rate']:.1%}), bound-hit {beh['bound_hit_count']}, "
              f"tool_calls mean={beh['tool_calls_mean']} max={beh['tool_calls_max']}")
        print("tool calls by kind:", beh["tool_calls_by_kind"])

        lat = summary["latency_ms"]
        print("\nlatency ms (concurrent, includes contention):")
        print(f"  plain p50={lat['plain']['p50']} p95={lat['plain']['p95']}")
        print(f"  agent p50={lat['agent']['p50']} p95={lat['agent']['p95']}")

        tok = summary["tokens"]
        plain_cost = cost(tok["plain"]["input"], tok["plain"]["output"], settings.llm_model, n)
        agent_cost = (
            cost(tok["agent"]["agent_model_input"], tok["agent"]["agent_model_output"],
                 settings.agent_model, n)
            + cost(tok["agent"]["answer_model_input"], tok["agent"]["answer_model_output"],
                   settings.llm_model, n)
        )
        print(f"\ncost per 1000 questions: plain ${plain_cost * 1000:.2f}  "
              f"agent ${agent_cost * 1000:.2f}  "
              f"(agent is {agent_cost / plain_cost:.1f}x plain)")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
