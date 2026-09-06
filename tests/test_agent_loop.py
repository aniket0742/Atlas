"""The bounded agent loop.

Driven by a scripted model rather than a real one. The loop's responsibilities
are enforcing bounds, feeding results back, accumulating evidence and degrading
safely -- all of which are deterministic, and none of which should be asserted
through the one component whose behaviour is not.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from atlas.agent.knowledge_base import SearchKnowledgeBaseTool
from atlas.agent.loop import AgentPlanner, StopReason
from atlas.agent.tools import Tool, ToolArgs, ToolContext, ToolOutcome, ToolRegistry
from atlas.config import Settings
from atlas.core.models import TokenUsage
from atlas.providers.base import (
    AgentTurn,
    LLMError,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from atlas.providers.fake import ScriptedToolCallingLLM
from atlas.retrieval.service import RetrievalResult
from tests.conftest import make_chunk
from tests.test_search_tool import FakeRetriever

TENANT = uuid.uuid5(uuid.NAMESPACE_DNS, "caller-tenant")
QUESTION = "What is the refund window and who approves exceptions?"


def settings(**overrides) -> Settings:
    base = {
        "gemini_api_key": "test-key",
        "agent_max_iterations": 4,
        "agent_max_tool_calls": 8,
        "agent_budget_seconds": 60.0,
    }
    return Settings(**{**base, **overrides})


def build(script=None, *, chunks=None, error=None, **setting_overrides):
    """A planner over a scripted model and a fake-backed search tool."""
    retriever = FakeRetriever(chunks if chunks is not None else [make_chunk("some text")])
    registry = ToolRegistry([SearchKnowledgeBaseTool(retriever)])  # type: ignore[arg-type]
    llm = ScriptedToolCallingLLM(script, error=error)
    planner = AgentPlanner(llm, registry, settings(**setting_overrides))
    return planner, llm, retriever


def context() -> ToolContext:
    return ToolContext(tenant_id=TENANT, request_id="req-1")


def search(query: str, **kwargs):
    return ("search_knowledge_base", {"query": query, **kwargs})


# ---------------------------------------------------------------------------
# The ordinary path
# ---------------------------------------------------------------------------


async def test_a_single_search_then_finish():
    planner, _, retriever = build([[search("refund window")], "found it"])

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.FINISHED
    assert plan.degraded is False
    assert len(plan.steps) == 2
    assert plan.tool_call_count == 1
    assert [c["query"] for c in retriever.calls] == ["refund window"]


async def test_the_model_may_search_several_times_across_iterations():
    """The behaviour the whole phase exists to test: decomposition."""
    a, b = make_chunk("refund window is 30 days"), make_chunk("exceptions need a manager")
    retriever = FakeRetriever([a, b])
    registry = ToolRegistry([SearchKnowledgeBaseTool(retriever)])  # type: ignore[arg-type]
    llm = ScriptedToolCallingLLM(
        [[search("refund window")], [search("who approves refund exceptions")], "done"]
    )
    planner = AgentPlanner(llm, registry, settings())

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.FINISHED
    assert plan.tool_call_count == 2
    assert [c["query"] for c in retriever.calls] == [
        "refund window",
        "who approves refund exceptions",
    ]


async def test_evidence_is_deduplicated_across_searches():
    """Two phrasings finding the same chunk must not cite it twice."""
    shared = make_chunk("shared passage")
    planner, _, _ = build(
        [[search("one")], [search("two")], "done"], chunks=[shared]
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.evidence == [shared]


async def test_parallel_calls_in_one_turn_are_all_executed():
    planner, _, retriever = build([[search("a"), search("b"), search("c")], "done"])

    plan = await planner.gather(QUESTION, context())

    assert plan.tool_call_count == 3
    assert sorted(c["query"] for c in retriever.calls) == ["a", "b", "c"]
    assert len(plan.steps[0].tool_results) == 3


# ---------------------------------------------------------------------------
# What the model is shown
# ---------------------------------------------------------------------------


async def test_the_model_sees_the_question_then_its_own_calls_then_results():
    planner, llm, _ = build([[search("refunds")], "done"])

    await planner.gather(QUESTION, context())

    first, second = llm.calls[0]["history"], llm.calls[1]["history"]
    assert first == [UserMessage(text=QUESTION)]
    assert isinstance(second[1], ModelMessage)
    assert second[1].tool_calls[0].name == "search_knowledge_base"
    assert isinstance(second[2], ToolResultMessage)
    assert second[2].response["ok"] is True


async def test_tool_failures_are_fed_back_so_the_model_can_correct_itself():
    """A bad call should cost one turn, not the request."""
    planner, llm, _ = build(
        [[("search_knowledge_base", {"quer": "typo"})], [search("refunds")], "done"]
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.steps[0].tool_results[0].outcome is ToolOutcome.INVALID_ARGUMENTS
    fed_back = llm.calls[1]["history"][2]
    assert fed_back.response["ok"] is False
    assert "query" in fed_back.response["error"]
    # And the model recovered, so the run is not degraded.
    assert plan.degraded is False


async def test_an_unknown_tool_name_is_reported_rather_than_fatal():
    planner, _, _ = build([[("web_search", {"q": "x"})], [search("refunds")], "done"])

    plan = await planner.gather(QUESTION, context())

    assert plan.steps[0].tool_results[0].outcome is ToolOutcome.UNKNOWN_TOOL
    assert plan.evidence


async def test_the_model_is_only_shown_tools_the_caller_may_use():
    class AdminArgs(ToolArgs):
        pass

    class AdminTool(Tool):
        name = "admin_only"
        description = "Privileged."
        Args = AdminArgs
        required_permission = "admin"

        async def execute(self, context, args):
            return {}

    registry = ToolRegistry(
        [SearchKnowledgeBaseTool(FakeRetriever([make_chunk("x")])), AdminTool()]  # type: ignore[arg-type]
    )
    llm = ScriptedToolCallingLLM(["done"])
    await AgentPlanner(llm, registry, settings()).gather(QUESTION, context())

    assert llm.calls[0]["tools"] == ["search_knowledge_base"]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


async def test_a_model_that_never_stops_is_stopped_by_the_iteration_cap():
    """The ordinary runaway: every turn requests another search."""
    planner, _, retriever = build(
        [[search(f"query {i}")] for i in range(50)], agent_max_iterations=3
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.MAX_ITERATIONS
    assert len(plan.steps) == 3
    assert len(retriever.calls) == 3
    # It still returns what it found rather than failing.
    assert plan.evidence


async def test_the_total_call_cap_bounds_work_the_iteration_cap_cannot():
    """One turn may request many calls, so iterations alone bound nothing."""
    planner, _, retriever = build(
        [[search(f"q{i}") for i in range(20)]],
        agent_max_iterations=4,
        agent_max_tool_calls=5,
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.MAX_TOOL_CALLS
    assert plan.tool_call_count == 5
    assert len(retriever.calls) == 5, "calls ran past the budget"


async def test_the_call_cap_counts_across_iterations_not_within_one():
    planner, _, retriever = build(
        [[search("a"), search("b")], [search("c"), search("d")], [search("e")]],
        agent_max_tool_calls=3,
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.tool_call_count == 3
    assert len(retriever.calls) == 3
    assert plan.stop_reason is StopReason.MAX_TOOL_CALLS


async def test_the_wall_clock_budget_stops_the_loop_before_a_new_step():
    """Each step is fine; the whole is far too slow to keep a request waiting."""

    class SlowModel(ScriptedToolCallingLLM):
        def generate_with_tools(self, **kwargs):
            import time

            time.sleep(0.06)
            return super().generate_with_tools(**kwargs)

    registry = ToolRegistry([SearchKnowledgeBaseTool(FakeRetriever([make_chunk("x")]))])  # type: ignore[arg-type]
    llm = SlowModel([[search(f"q{i}")] for i in range(20)])
    planner = AgentPlanner(
        llm, registry, settings(agent_budget_seconds=0.1, agent_max_iterations=20)
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert len(plan.steps) < 20


async def test_a_model_call_may_not_outlive_the_remaining_budget():
    """Otherwise one slow call could blow the whole budget on its own."""
    planner, llm, _ = build([[search("a")], [search("b")], "done"], agent_budget_seconds=5.0)

    await planner.gather(QUESTION, context())

    budgets = [call["timeout_seconds"] for call in llm.calls]
    assert all(0 < b <= 5.0 for b in budgets)
    assert budgets == sorted(budgets, reverse=True), "budget did not shrink as time passed"


# ---------------------------------------------------------------------------
# Degrading to plain retrieval
# ---------------------------------------------------------------------------


async def test_a_provider_failure_degrades_to_a_direct_search():
    planner, _, retriever = build(error=LLMError("Gemini call failed (status=429)"))

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.LLM_ERROR
    assert plan.degraded is True
    assert "429" in plan.degraded_reason
    # It searched for the original question, exactly as plain RAG would.
    assert [c["query"] for c in retriever.calls] == [QUESTION]
    assert plan.evidence


async def test_a_model_that_answers_without_searching_still_gets_evidence():
    """The most likely real failure: routing simply does not fire."""
    planner, _, retriever = build(["The refund window is 30 days."])

    plan = await planner.gather(QUESTION, context())

    assert plan.stop_reason is StopReason.FINISHED
    assert plan.degraded is True
    assert "no evidence" in plan.degraded_reason
    assert [c["query"] for c in retriever.calls] == [QUESTION]


async def test_searches_that_all_come_back_empty_also_degrade():
    planner, _, retriever = build([[search("a")], "nothing found"], chunks=[])

    plan = await planner.gather(QUESTION, context())

    assert plan.degraded is True
    assert [c["query"] for c in retriever.calls] == ["a", QUESTION]


async def test_a_degraded_run_records_the_fallback_as_a_step():
    """A trace must not jump from no searches to some evidence unexplained."""
    planner, _, _ = build(["answering directly"])

    plan = await planner.gather(QUESTION, context())

    assert "[fallback]" in plan.steps[-1].text
    assert plan.steps[-1].tool_results[0].tool == "search_knowledge_base"


async def test_a_successful_run_is_not_marked_degraded():
    planner, _, _ = build([[search("refunds")], "done"])
    plan = await planner.gather(QUESTION, context())
    assert plan.degraded is False
    assert plan.degraded_reason is None


async def test_evidence_found_before_a_provider_failure_is_kept():
    """A partial search beats nothing, so a late failure must not discard it."""

    class FailsOnSecondCall(ScriptedToolCallingLLM):
        def generate_with_tools(self, **kwargs):
            if len(self.calls) >= 1:
                self.calls.append({"history": [], "tools": [], "system": "", "timeout_seconds": 0})
                raise LLMError("upstream died")
            return super().generate_with_tools(**kwargs)

    chunk = make_chunk("found before the failure")
    registry = ToolRegistry([SearchKnowledgeBaseTool(FakeRetriever([chunk]))])  # type: ignore[arg-type]
    llm = FailsOnSecondCall([[search("refunds")]])
    plan = await AgentPlanner(llm, registry, settings()).gather(QUESTION, context())

    assert plan.stop_reason is StopReason.LLM_ERROR
    assert plan.evidence == [chunk]
    # Evidence exists, so no fallback ran and the run is not marked degraded.
    assert plan.degraded is False


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


async def test_every_search_runs_against_the_caller_tenant():
    planner, _, retriever = build([[search("a"), search("b")], [search("c")], "done"])

    await planner.gather(QUESTION, context())

    assert {c["tenant_id"] for c in retriever.calls} == {TENANT}


async def test_a_model_supplied_tenant_argument_is_refused_mid_loop():
    """Injection reaching the loop, not just the tool.

    The scripted model here is standing in for one that was persuaded by a
    poisoned document. The call fails, the failure is fed back, and no search
    ran against the named tenant.
    """
    victim = uuid.uuid5(uuid.NAMESPACE_DNS, "victim-tenant")
    planner, _, retriever = build(
        [
            [("search_knowledge_base", {"query": "salaries", "tenant_id": str(victim)})],
            [search("refunds")],
            "done",
        ]
    )

    plan = await planner.gather(QUESTION, context())

    assert plan.steps[0].tool_results[0].outcome is ToolOutcome.INVALID_ARGUMENTS
    assert {c["tenant_id"] for c in retriever.calls} == {TENANT}
    assert len(retriever.calls) == 1, "the rejected call reached the retriever"


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------


async def test_the_trace_records_what_the_agent_did():
    planner, _, _ = build([[search("refund window")], "found it"])

    trace = (await planner.gather(QUESTION, context())).trace()

    assert trace["stop_reason"] == "finished"
    assert trace["iterations"] == 2
    assert trace["tool_calls"] == 1
    assert trace["degraded"] is False
    assert trace["model"] == "fake-agent"
    call = trace["steps"][0]["tool_calls"][0]
    assert call["tool"] == "search_knowledge_base"
    assert call["arguments"] == {"query": "refund window"}
    assert call["outcome"] == "ok"
    assert call["result_count"] == 1


async def test_the_trace_does_not_restate_the_corpus():
    """Document text belongs in citations, not in every trace."""
    planner, _, _ = build([[search("x")], "done"], chunks=[make_chunk("SECRET" * 500)])

    trace = (await planner.gather(QUESTION, context())).trace()

    assert "SECRET" not in str(trace)


async def test_the_trace_is_json_serialisable():
    import json

    planner, _, _ = build([[search("x")], "done"])
    json.dumps((await planner.gather(QUESTION, context())).trace())


async def test_usage_is_summed_across_every_model_call():
    """One number per call would understate what an agent request costs."""
    planner, _, _ = build([[search("a")], [search("b")], "done"])

    plan = await planner.gather(QUESTION, context())

    assert plan.usage.prompt_tokens == 300
    assert plan.usage.output_tokens == 60
    # Billed as output but invisible in the response, so tracked separately.
    assert plan.usage.thinking_tokens == 15


async def test_timings_are_recorded_per_step():
    planner, _, _ = build([[search("x")], "done"])

    plan = await planner.gather(QUESTION, context())

    assert plan.duration_ms > 0
    assert plan.steps[0].llm_ms >= 0
    assert plan.steps[0].tools_ms >= 0


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_agent_prompt_does_not_pretend_to_enforce_authorization():
    """A prompt is a quality lever, not a control. It must not read as one."""
    from atlas.agent.prompts import AGENT_SYSTEM_INSTRUCTION

    lowered = AGENT_SYSTEM_INSTRUCTION.lower()
    assert "tenant" not in lowered
    # It does tell the model to treat retrieved text as data -- worth doing,
    # worth not relying on.
    assert "never as instructions" in lowered


@pytest.mark.parametrize("bound", ["agent_max_iterations", "agent_max_tool_calls"])
async def test_a_bound_of_zero_still_returns_an_answerable_plan(bound):
    """Misconfiguration should degrade, not crash."""
    planner, _, retriever = build([[search("a")], "done"], **{bound: 0})

    plan = await planner.gather(QUESTION, context())

    assert plan.degraded is True
    assert [c["query"] for c in retriever.calls] == [QUESTION]


async def test_concurrent_requests_do_not_share_state():
    """The planner and its tools are shared across requests.

    A leak here would cross tenants, which makes this worth asserting even
    though nothing in the loop stores per-request state today.

    Both fakes are deliberately stateless. `ScriptedToolCallingLLM` indexes its
    script by a call counter and `FakeRetriever` records into a shared list, so
    neither is safe to drive from two requests at once -- using them here would
    test the fixtures rather than the loop.
    """
    other = uuid.uuid5(uuid.NAMESPACE_DNS, "other-tenant")
    mine, theirs = make_chunk("mine"), make_chunk("theirs")

    class PerTenantRetriever:
        async def retrieve(self, tenant_id, query, *, top_k=None, **kwargs):
            # Yield, so the two requests genuinely interleave.
            await asyncio.sleep(0.01)
            chunks = [mine] if tenant_id == TENANT else [theirs]
            return RetrievalResult(
                chunks=chunks, candidates=chunks, timings_ms={}, mode="dense",
                best_dense_score=0.8,
            )

    class SearchOnceThenFinish:
        """Decides from the history it is given, not from a counter."""

        model_id = "stateless-fake"

        def generate_with_tools(self, *, system_instruction, history, tools, timeout_seconds=None):
            if len(history) == 1:
                return AgentTurn(
                    text=None,
                    tool_calls=(ToolCall(name="search_knowledge_base", arguments={"query": "x"}),),
                    usage=TokenUsage(),
                )
            return AgentTurn(text="done", tool_calls=(), usage=TokenUsage())

    registry = ToolRegistry([SearchKnowledgeBaseTool(PerTenantRetriever())])  # type: ignore[arg-type]
    planner = AgentPlanner(SearchOnceThenFinish(), registry, settings())  # type: ignore[arg-type]

    a, b = await asyncio.gather(
        planner.gather(QUESTION, ToolContext(tenant_id=TENANT)),
        planner.gather(QUESTION, ToolContext(tenant_id=other)),
    )

    assert a.evidence == [mine]
    assert b.evidence == [theirs]
    assert not a.degraded and not b.degraded
