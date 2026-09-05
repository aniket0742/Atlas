"""The bounded agent loop.

## What this does, and what it deliberately does not

The loop gathers **evidence**. It does not write the answer.

A model is given the question and the tool declarations, and is asked to search
until it has what it needs. Each turn it may request tool calls; the loop runs
them through `ToolRegistry.invoke`, feeds the results back, and repeats. When
the model stops asking for tools -- or a bound is hit -- the loop returns the
accumulated chunks and a trace of how they were found.

Writing the answer stays where it already was: the grounded path, with the
answer model, evidence blocks, citation resolution and quote verification
(ADR-0011). This split is a direct consequence of the two-model decision
(ADR-0024): routing is cheap and high-volume, answering is the product. It also
means the agent model is never trusted to produce a citation, because it is
never asked for one.

## Every loop needs a reason it cannot run forever

A model that keeps calling tools is not a hypothetical -- it is the ordinary
failure mode of a tool-calling loop, and on a metered API it is the expensive
one. Four independent bounds, because each catches something the others do not:

* **iterations** -- the model round-trips. Caps reasoning depth.
* **tool calls in total** -- one iteration can request several calls at once, so
  an iteration cap alone does not bound the work.
* **wall-clock budget** -- catches the case where each individual step is within
  its limits but the whole is far too slow for a request to wait on. Checked
  before starting anything new, never mid-flight.
* **per-tool timeout** -- already enforced by the registry.

Hitting a bound is not an error. The loop stops and returns the evidence it has,
because a partial answer that cites real sources is worth more than a failure,
and the trace records which bound stopped it.

## Degrading to plain retrieval

Everything about the agent path can fail: the API can be down, the model can
refuse to call anything, the whole feature can be misconfigured. None of that
should turn into "no answer", because Atlas could always answer this question
without an agent.

So when the model path produces no evidence for any reason, the loop runs a
single direct search on the original question -- exactly what plain RAG would
have done -- and marks the result degraded. The trace says why. A caller can
serve the answer and an operator can still see that the agent did not work.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from atlas.agent.knowledge_base import evidence_from
from atlas.agent.prompts import AGENT_SYSTEM_INSTRUCTION
from atlas.agent.tools import ToolContext, ToolRegistry, ToolResult
from atlas.config import Settings
from atlas.core.models import RetrievedChunk, TokenUsage
from atlas.providers.base import (
    AgentMessage,
    LLMError,
    ModelMessage,
    ToolCall,
    ToolCallingLLM,
    ToolResultMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)

#: The tool the loop falls back to when the model path yields nothing.
FALLBACK_TOOL = "search_knowledge_base"


class StopReason(StrEnum):
    """Why the loop stopped. Recorded on every run, including successful ones."""

    #: The model stopped asking for tools. The only reason that is not a bound.
    FINISHED = "finished"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOOL_CALLS = "max_tool_calls"
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: The provider failed and did not recover. Evidence gathered before the
    #: failure is kept -- a partial search still beats nothing.
    LLM_ERROR = "llm_error"


@dataclass(slots=True)
class AgentStep:
    """One iteration: what the model said, what it called, what came back."""

    iteration: int
    text: str | None
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_ms: float = 0.0
    tools_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Trace shape for the API and the console.

        Tool *content* is excluded on purpose: it is document text, it is large,
        and it is already reachable through the citations the answer resolves.
        A trace should show what the agent did, not restate the corpus.
        """
        return {
            "iteration": self.iteration,
            "text": self.text,
            "llm_ms": round(self.llm_ms, 1),
            "tools_ms": round(self.tools_ms, 1),
            "tool_calls": [
                {
                    "tool": result.tool,
                    "arguments": result.arguments,
                    "outcome": result.outcome.value,
                    "duration_ms": round(result.duration_ms, 1),
                    "result_count": _result_count(result),
                    "error": result.error,
                }
                for result in self.tool_results
            ],
        }


@dataclass(slots=True)
class AgentPlan:
    """The evidence the agent gathered, and the record of how."""

    evidence: list[RetrievedChunk]
    steps: list[AgentStep]
    stop_reason: StopReason
    usage: TokenUsage
    duration_ms: float
    #: True when the evidence came from the fallback search rather than the
    #: model's own tool use. The answer is still served; the trace says so.
    degraded: bool = False
    degraded_reason: str | None = None
    model: str = ""

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_results) for step in self.steps)

    def trace(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "iterations": len(self.steps),
            "tool_calls": self.tool_call_count,
            "stop_reason": self.stop_reason.value,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "evidence_count": len(self.evidence),
            "duration_ms": round(self.duration_ms, 1),
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.output_tokens,
                "thinking_tokens": self.usage.thinking_tokens,
            },
            "steps": [step.summary() for step in self.steps],
        }


class AgentPlanner:
    """Runs the loop. Holds no per-request state."""

    def __init__(
        self,
        llm: ToolCallingLLM,
        registry: ToolRegistry,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._settings = settings

    async def gather(self, question: str, context: ToolContext) -> AgentPlan:
        started = time.perf_counter()
        settings = self._settings
        deadline = started + settings.agent_budget_seconds

        history: list[AgentMessage] = [UserMessage(text=question)]
        # Declarations are filtered by the caller's permissions, so a tool the
        # caller cannot use is never advertised and cannot be attempted.
        declarations = self._registry.declarations(context)

        steps: list[AgentStep] = []
        usage = TokenUsage()
        results: list[ToolResult] = []
        stop_reason = StopReason.MAX_ITERATIONS
        degraded_reason: str | None = None

        for iteration in range(1, settings.agent_max_iterations + 1):
            if time.perf_counter() >= deadline:
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break

            t0 = time.perf_counter()
            try:
                turn = await asyncio.to_thread(
                    lambda: self._llm.generate_with_tools(
                        system_instruction=AGENT_SYSTEM_INSTRUCTION,
                        history=list(history),
                        tools=declarations,
                        # Never let one call outlive the whole loop's budget.
                        timeout_seconds=max(1.0, deadline - time.perf_counter()),
                    )
                )
            except LLMError as exc:
                logger.warning("agent loop: provider failed at iteration %s: %s", iteration, exc)
                stop_reason = StopReason.LLM_ERROR
                degraded_reason = f"{type(exc).__name__}: {exc}"
                break
            llm_ms = (time.perf_counter() - t0) * 1000

            usage = _accumulate(usage, turn.usage)
            step = AgentStep(iteration=iteration, text=turn.text, llm_ms=llm_ms)
            steps.append(step)

            if not turn.wants_tools:
                # The model considers itself done. Whether it is right is not
                # this loop's judgement to make -- the grounded answering path
                # decides whether the evidence actually supports an answer.
                stop_reason = StopReason.FINISHED
                break

            calls = self._within_call_budget(turn.tool_calls, len(results))
            truncated = len(calls) < len(turn.tool_calls)

            history.append(ModelMessage(text=turn.text, tool_calls=tuple(calls)))

            t0 = time.perf_counter()
            step_results = await self._run_calls(calls, context)
            step.tools_ms = (time.perf_counter() - t0) * 1000

            step.tool_results = step_results
            results.extend(step_results)

            for call, result in zip(calls, step_results, strict=True):
                history.append(
                    ToolResultMessage(
                        name=call.name, response=result.for_model(), call_id=call.id
                    )
                )

            if truncated:
                stop_reason = StopReason.MAX_TOOL_CALLS
                break

        evidence = evidence_from(results)
        plan = AgentPlan(
            evidence=evidence,
            steps=steps,
            stop_reason=stop_reason,
            usage=usage,
            duration_ms=(time.perf_counter() - started) * 1000,
            model=getattr(self._llm, "model_id", ""),
        )

        if not evidence:
            await self._fall_back(plan, question, context, degraded_reason, stop_reason)

        plan.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "agent question=%r iterations=%s tool_calls=%s stop=%s evidence=%s "
            "degraded=%s duration_ms=%.0f tenant=%s request=%s",
            question[:120],
            len(plan.steps),
            plan.tool_call_count,
            plan.stop_reason.value,
            len(plan.evidence),
            plan.degraded,
            plan.duration_ms,
            context.tenant_id,
            context.request_id or "-",
        )
        return plan

    # -- bounds ------------------------------------------------------------

    def _within_call_budget(self, calls: tuple[ToolCall, ...], used: int) -> list[ToolCall]:
        """Trim a turn's calls to what the total budget still allows.

        A model can request several calls in one turn, so the iteration cap
        alone does not bound the work done. Trimming rather than rejecting the
        whole turn keeps whatever the model asked for first, which is usually
        its best guess.
        """
        remaining = max(0, self._settings.agent_max_tool_calls - used)
        return list(calls[:remaining])

    async def _run_calls(
        self, calls: list[ToolCall], context: ToolContext
    ) -> list[ToolResult]:
        """Execute one turn's calls concurrently.

        Safe because tools are stateless and each carries its own timeout. The
        gain is real: two searches in a turn cost the slower one rather than
        their sum, and the multi-document questions this loop exists for are
        exactly the ones that produce several calls at once.

        `invoke` never raises, so there is no partial-failure case to unwind --
        a failed call is a result like any other and is fed back to the model.
        """
        if not calls:
            return []
        return list(
            await asyncio.gather(
                *(
                    self._registry.invoke(call.name, call.arguments, context)
                    for call in calls
                )
            )
        )

    # -- degradation -------------------------------------------------------

    async def _fall_back(
        self,
        plan: AgentPlan,
        question: str,
        context: ToolContext,
        degraded_reason: str | None,
        stop_reason: StopReason,
    ) -> None:
        """One plain search on the original question, as plain RAG would do.

        Reached whenever the model path produced no evidence: the provider
        failed, the model answered without searching, every search came back
        empty, or a bound cut it off first. In all of those Atlas can still do
        the ordinary thing, and refusing because the *agent* failed would be a
        worse answer than the system is capable of.
        """
        result = await self._registry.invoke(
            FALLBACK_TOOL, {"query": question}, context
        )
        plan.evidence = evidence_from([result])
        plan.degraded = True
        plan.degraded_reason = degraded_reason or f"no evidence gathered ({stop_reason.value})"
        # Recorded as a step so the trace shows the fallback ran rather than
        # leaving an unexplained jump from no searches to some evidence.
        plan.steps.append(
            AgentStep(
                iteration=len(plan.steps) + 1,
                text="[fallback] direct search on the original question",
                tool_results=[result],
                tools_ms=result.duration_ms,
            )
        )
        logger.warning(
            "agent degraded to plain retrieval: %s (tenant=%s)",
            plan.degraded_reason,
            context.tenant_id,
        )


def _accumulate(total: TokenUsage, turn: TokenUsage) -> TokenUsage:
    """Sum usage across turns.

    An agent request costs several model calls, so a per-call number understates
    it. Thinking tokens are kept separate because they are billed as output but
    never appear in the response -- folding them in would hide where the cost
    actually went.
    """
    return TokenUsage(
        prompt_tokens=total.prompt_tokens + turn.prompt_tokens,
        output_tokens=total.output_tokens + turn.output_tokens,
        thinking_tokens=total.thinking_tokens + turn.thinking_tokens,
        total_tokens=total.total_tokens + turn.total_tokens,
    )


def _result_count(result: ToolResult) -> int | None:
    content = result.content
    if isinstance(content, dict) and isinstance(content.get("result_count"), int):
        return content["result_count"]
    return None


__all__ = [
    "AgentPlan",
    "AgentPlanner",
    "AgentStep",
    "StopReason",
    "FALLBACK_TOOL",
]
