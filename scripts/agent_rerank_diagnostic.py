"""Diagnostic: does reranking the union once actually change anything?

Three arms on the same questions with the same answer model:

  plain    -- no agent at all, the existing baseline
  A        -- agent, interleaved evidence (behaviour before the fix)
  B        -- agent, union reranked once against the original question

The agent's *searches* are held fixed within a question by running A and B from
one gathered plan, so any difference between them is the ordering and nothing
else. That is the whole point: comparing two independent agent runs would
confound the ordering change with the model choosing different searches.

Questions are the five labelled multi-doc cases from the eval set, which carry
ground-truth documents, plus the two from the step 6 observation.
"""

import asyncio
import json
import sys

from atlas.agent.knowledge_base import SearchKnowledgeBaseTool, evidence_from
from atlas.agent.loop import AgentPlanner
from atlas.agent.service import AgentAnswerService
from atlas.agent.tools import ToolContext, ToolRegistry
from atlas.answer.service import AnswerService
from atlas.config import get_settings
from atlas.db import repository as repo
from atlas.db.pool import Database
from atlas.providers.factory import get_agent_llm, get_embedder, get_llm, get_reranker
from atlas.retrieval.service import Retriever

DATASET = "eval/datasets/main.jsonl"
OUT = "eval/results/agent-rerank-diagnostic.json"
EXTRA = [
    {
        "id": "step6-refunds",
        "question": "What is the refund window and who approves exceptions to it?",
        "labels": [
            {"document": "policies/billing.md"},
            {"document": "policies/refunds-enterprise.md"},
        ],
    },
    {
        "id": "step6-apikey",
        "question": "How do I rotate an API key, and what should I do if one leaks?",
        "labels": [{"document": "security/secrets-management.md"}],
    },
]


def load_questions():
    rows = []
    with open(DATASET, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("//"):
                rows.append(json.loads(line))
    multi = [r for r in rows if r.get("kind") == "multi-doc"]
    return multi + EXTRA


def report(row, label, answer, expected):
    cited = sorted({c.document_external_id for c in answer.citations})
    covered = sum(1 for doc in expected if doc in cited)
    order = []
    if answer.agent_trace:
        order = [e["document"] for e in answer.agent_trace["evidence"]["order"]]
    return {
        "id": row["id"],
        "arm": label,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "cited_documents": cited,
        "expected_documents": expected,
        "documents_covered": f"{covered}/{len(expected)}",
        "citations": len(answer.citations),
        "unverified_quotes": sum(1 for c in answer.citations if not c.quote_verified),
        "prompt_tokens": answer.usage.prompt_tokens,
        "output_tokens": answer.usage.output_tokens,
        "total_ms": round(answer.timings_ms.get("total_ms", 0)),
        "evidence_order": order,
        "answer": (answer.text or "")[:400],
    }


async def main() -> int:
    settings = get_settings()
    db = Database(settings.database_url)
    await db.open()
    results = []
    try:
        async with db.connection() as conn:
            tenant = await repo.ensure_tenant(conn, settings.default_tenant_slug)
        context = ToolContext(tenant_id=tenant)

        reranker = get_reranker(settings)
        retriever = Retriever(db, get_embedder(settings), settings, reranker=reranker)
        answerer = AnswerService(db, retriever, get_llm(settings), settings)
        registry = ToolRegistry([SearchKnowledgeBaseTool(retriever)])
        planner = AgentPlanner(get_agent_llm(settings), registry, settings)
        service = AgentAnswerService(planner, answerer, settings, reranker=reranker)

        for row in load_questions():
            question = row["question"]
            expected = sorted({label["document"] for label in row["labels"]})
            print("=" * 78, flush=True)
            print(row["id"], "|", question, flush=True)

            plain = await answerer.answer(tenant, question)
            results.append(report(row, "plain", plain, expected))

            # One gathered plan, two orderings. Holding the searches fixed is
            # what makes this a test of the ordering rather than of the model.
            plan = await planner.gather(question, context)
            union = evidence_from(
                [r for step in plan.steps for r in step.tool_results]
            )
            searches = [
                call["arguments"].get("query")
                for step in plan.trace()["steps"]
                for call in step["tool_calls"]
            ]
            print("  searches:", searches, flush=True)
            print("  union size:", len(union), flush=True)

            for arm, use_rerank in (("A-interleave", False), ("B-union-rerank", True)):
                settings.agent_union_rerank = use_rerank
                evidence, info = await service._order(question, list(union))
                answer = await answerer.answer_from_evidence(
                    tenant, question, evidence, provenance={}
                )
                answer.agent_trace = {"evidence": {**info, "order": [
                    {"document": c.document_external_id, "score": round(float(c.score), 3)}
                    for c in evidence
                ]}}
                entry = report(row, arm, answer, expected)
                entry["searches"] = searches
                entry["union_size"] = len(union)
                entry["rerank_ms"] = info["rerank_ms"]
                results.append(entry)
                print(f"  {arm}: covered={entry['documents_covered']} "
                      f"refused={entry['refused']} tokens={entry['prompt_tokens']} "
                      f"order={entry['evidence_order'][:5]}", flush=True)
    finally:
        await db.close()

    out = OUT
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
