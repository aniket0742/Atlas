"""The tool framework.

No database and no model. These pin the guarantees the registry makes on behalf
of every tool, especially the ones a tool author could otherwise forget.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import FrozenInstanceError

import pytest
from pydantic import Field

from atlas.agent.tools import Tool, ToolArgs, ToolContext, ToolOutcome, ToolRegistry

TENANT = uuid.uuid4()


def context(**kwargs) -> ToolContext:
    return ToolContext(tenant_id=TENANT, **kwargs)


class SearchArgs(ToolArgs):
    query: str = Field(description="A focused search phrase.")
    limit: int = Field(default=5, ge=1, le=20)


class RecordingTool(Tool):
    """Records whether it ran, so tests can assert it did *not*."""

    name = "search"
    description = "Search the knowledge base."
    Args = SearchArgs
    timeout_seconds = 5.0

    def __init__(self) -> None:
        self.calls: list[tuple[ToolContext, SearchArgs]] = []

    async def execute(self, context: ToolContext, args: SearchArgs):
        self.calls.append((context, args))
        return {"hits": [args.query] * args.limit}


class SlowTool(Tool):
    name = "slow"
    description = "Sleeps past its budget."
    Args = SearchArgs
    timeout_seconds = 0.05

    def __init__(self) -> None:
        self.cancelled = False

    async def execute(self, context: ToolContext, args: SearchArgs):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "never"


class ExplodingTool(Tool):
    name = "explode"
    description = "Raises."
    Args = SearchArgs

    async def execute(self, context: ToolContext, args: SearchArgs):
        raise RuntimeError("upstream is down")


class PrivilegedTool(RecordingTool):
    name = "privileged"
    description = "Needs a capability."
    required_permission = "admin"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_declaration_carries_name_description_and_schema():
    tool = RecordingTool()
    decl = tool.declaration()
    assert decl["name"] == "search"
    assert decl["description"]
    assert decl["parameters"]["required"] == ["query"]
    assert sorted(decl["parameters"]["properties"]) == ["limit", "query"]


def test_declaration_strips_pydantic_titles():
    """`title` is bookkeeping that only costs prompt tokens."""
    decl = RecordingTool().declaration()
    assert "title" not in decl["parameters"]
    for prop in decl["parameters"]["properties"].values():
        assert "title" not in prop


def test_declaration_is_accepted_by_the_provider_sdk():
    """The framework stays provider-agnostic, but must actually fit one.

    A neutral dict is worthless if the SDK rejects it, so this asserts the
    contract rather than assuming it.
    """
    types = pytest.importorskip("google.genai.types")
    decl = RecordingTool().declaration()
    fd = types.FunctionDeclaration(
        name=decl["name"], description=decl["description"], parameters=decl["parameters"]
    )
    assert fd.parameters.required == ["query"]


def test_registering_the_same_name_twice_is_refused():
    registry = ToolRegistry([RecordingTool()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(RecordingTool())


def test_registry_lists_names_sorted():
    registry = ToolRegistry([PrivilegedTool(), RecordingTool()])
    assert registry.names() == ["privileged", "search"]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_valid_call_executes_and_returns_content():
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    result = await registry.invoke("search", {"query": "refunds", "limit": 2}, context())

    assert result.ok
    assert result.outcome is ToolOutcome.OK
    assert result.content == {"hits": ["refunds", "refunds"]}
    assert result.duration_ms >= 0
    assert len(tool.calls) == 1


async def test_defaults_are_applied_before_execute_sees_them():
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    await registry.invoke("search", {"query": "x"}, context())

    _, args = tool.calls[0]
    assert args.limit == 5, "execute() should receive a fully-populated model"


async def test_the_context_reaches_the_tool_unchanged():
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    ctx = context(permissions=frozenset({"admin"}), request_id="req-1")

    await registry.invoke("search", {"query": "x"}, ctx)

    seen, _ = tool.calls[0]
    assert seen is ctx
    assert seen.tenant_id == TENANT


# ---------------------------------------------------------------------------
# Failures are returned, never raised
# ---------------------------------------------------------------------------


async def test_unknown_tool_returns_a_result_naming_the_real_ones():
    """Models invent tool names; a dead end should become a correctable turn."""
    registry = ToolRegistry([RecordingTool()])

    result = await registry.invoke("serach", {"query": "x"}, context())

    assert result.outcome is ToolOutcome.UNKNOWN_TOOL
    assert not result.ok
    assert "search" in result.error


async def test_invalid_arguments_do_not_reach_the_tool():
    """The security-relevant half: validation happens *before* execution."""
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    result = await registry.invoke("search", {"quer": "typo"}, context())

    assert result.outcome is ToolOutcome.INVALID_ARGUMENTS
    assert tool.calls == [], "the tool ran despite invalid arguments"


async def test_validation_message_is_actionable():
    registry = ToolRegistry([RecordingTool()])

    result = await registry.invoke("search", {}, context())

    # The model has to be able to fix it from this string alone.
    assert "query" in result.error
    assert "required" in result.error.lower()
    # Pydantic's default rendering carries URLs that cost tokens and help nobody.
    assert "https://" not in result.error


async def test_constraint_violations_are_rejected_too():
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    result = await registry.invoke("search", {"query": "x", "limit": 999}, context())

    assert result.outcome is ToolOutcome.INVALID_ARGUMENTS
    assert tool.calls == []


async def test_timeout_is_per_tool_and_cancels_the_work():
    tool = SlowTool()
    registry = ToolRegistry([tool])

    result = await registry.invoke("slow", {"query": "x"}, context())

    assert result.outcome is ToolOutcome.TIMEOUT
    assert "0.05s" in result.error
    # A timeout that leaves the coroutine running would leak work per call.
    assert tool.cancelled, "the tool coroutine was not cancelled"


async def test_a_raising_tool_becomes_an_error_result():
    registry = ToolRegistry([ExplodingTool()])

    result = await registry.invoke("explode", {"query": "x"}, context())

    assert result.outcome is ToolOutcome.ERROR
    assert "upstream is down" in result.error


async def test_arguments_are_recorded_as_the_model_supplied_them():
    """A trace should show what was attempted, not what happened to parse."""
    registry = ToolRegistry([RecordingTool()])

    result = await registry.invoke("search", {"quer": "typo"}, context())

    assert result.arguments == {"quer": "typo"}


async def test_missing_arguments_are_treated_as_empty_not_a_crash():
    registry = ToolRegistry([RecordingTool()])
    result = await registry.invoke("search", None, context())
    assert result.outcome is ToolOutcome.INVALID_ARGUMENTS


# ---------------------------------------------------------------------------
# Permissions (enforcement detail; the boundary itself is the next step)
# ---------------------------------------------------------------------------


async def test_a_tool_requiring_permission_is_denied_without_it():
    tool = PrivilegedTool()
    registry = ToolRegistry([tool])

    result = await registry.invoke("privileged", {"query": "x"}, context())

    assert result.outcome is ToolOutcome.DENIED
    assert tool.calls == [], "a denied tool was executed anyway"


async def test_the_denial_message_does_not_leak_the_permission_name():
    """The authorization model should not become text the model reasons about."""
    registry = ToolRegistry([PrivilegedTool()])
    result = await registry.invoke("privileged", {"query": "x"}, context())
    assert "admin" not in result.error


async def test_a_permitted_caller_may_use_the_tool():
    tool = PrivilegedTool()
    registry = ToolRegistry([tool])

    result = await registry.invoke(
        "privileged", {"query": "x"}, context(permissions=frozenset({"admin"}))
    )

    assert result.ok
    assert len(tool.calls) == 1


def test_declarations_hide_tools_the_caller_cannot_use():
    """Better not to advertise a tool than to advertise and then deny it."""
    registry = ToolRegistry([RecordingTool(), PrivilegedTool()])

    without = [d["name"] for d in registry.declarations(context())]
    with_perm = [
        d["name"]
        for d in registry.declarations(context(permissions=frozenset({"admin"})))
    ]

    assert without == ["search"]
    assert with_perm == ["privileged", "search"]


def test_declarations_without_a_context_list_everything():
    registry = ToolRegistry([RecordingTool(), PrivilegedTool()])
    assert [d["name"] for d in registry.declarations()] == ["privileged", "search"]


# ---------------------------------------------------------------------------
# Context immutability
# ---------------------------------------------------------------------------


def test_context_cannot_be_mutated_by_a_tool():
    """A tool must not be able to widen its own scope."""
    ctx = context()
    with pytest.raises(FrozenInstanceError):
        ctx.tenant_id = uuid.uuid4()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Payload sent back to the model
# ---------------------------------------------------------------------------


async def test_for_model_carries_the_result_on_success():
    registry = ToolRegistry([RecordingTool()])
    result = await registry.invoke("search", {"query": "x", "limit": 1}, context())
    payload = result.for_model()
    assert payload["ok"] is True
    assert payload["result"] == {"hits": ["x"]}


async def test_for_model_carries_the_error_so_the_model_can_correct_itself():
    registry = ToolRegistry([RecordingTool()])
    result = await registry.invoke("search", {}, context())
    payload = result.for_model()
    assert payload["ok"] is False
    assert "query" in payload["error"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


async def test_every_call_logs_name_outcome_and_duration(caplog):
    registry = ToolRegistry([RecordingTool()])
    with caplog.at_level(logging.INFO, logger="atlas.agent.tools"):
        await registry.invoke("search", {"query": "refunds"}, context(request_id="req-9"))

    line = caplog.text
    assert "tool=search" in line
    assert "outcome=ok" in line
    assert "duration_ms=" in line
    assert "request=req-9" in line
    assert "refunds" in line


async def test_failures_log_at_warning_with_the_error(caplog):
    registry = ToolRegistry([RecordingTool()])
    with caplog.at_level(logging.INFO, logger="atlas.agent.tools"):
        await registry.invoke("search", {}, context())

    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "outcome=invalid_arguments" in caplog.text


async def test_long_arguments_are_truncated_in_logs(caplog):
    """A model can emit a very long string; logs should not carry all of it."""
    registry = ToolRegistry([RecordingTool()])
    with caplog.at_level(logging.INFO, logger="atlas.agent.tools"):
        await registry.invoke("search", {"query": "x" * 5000}, context())

    assert "truncated" in caplog.text
    assert len(caplog.text) < 2000


# ---------------------------------------------------------------------------
# ToolOutput: splitting what the model sees from what the server keeps
# ---------------------------------------------------------------------------


class SplittingTool(Tool):
    name = "splitting"
    description = "Returns a summary to the model and the bulk to the server."
    Args = SearchArgs

    async def execute(self, context: ToolContext, args: SearchArgs):
        from atlas.agent.tools import ToolOutput

        return ToolOutput(content={"count": 2}, artifacts=["full text a", "full text b"])


async def test_tool_output_splits_content_from_artifacts():
    registry = ToolRegistry([SplittingTool()])

    result = await registry.invoke("splitting", {"query": "x"}, context())

    assert result.ok
    assert result.content == {"count": 2}
    assert result.artifacts == ["full text a", "full text b"]


async def test_artifacts_are_excluded_from_the_model_payload():
    """The exclusion is the whole point, so it is a property of the type."""
    registry = ToolRegistry([SplittingTool()])

    payload = (await registry.invoke("splitting", {"query": "x"}, context())).for_model()

    assert payload == {"ok": True, "result": {"count": 2}}
    assert "full text a" not in str(payload)


async def test_a_plain_return_value_still_works_and_has_no_artifacts():
    """Only tools that need the split pay for it."""
    registry = ToolRegistry([RecordingTool()])

    result = await registry.invoke("search", {"query": "x", "limit": 1}, context())

    assert result.content == {"hits": ["x"]}
    assert result.artifacts is None


class DocumentedArgs(ToolArgs):
    """Internal rationale that must never be sent to the model.

    Explaining a design decision to a maintainer costs nothing here and
    hundreds of prompt tokens if it leaks into the declaration.
    """

    query: str = Field(description="A focused search phrase.")


class DocumentedTool(RecordingTool):
    name = "documented"
    Args = DocumentedArgs


def test_the_args_docstring_does_not_leak_into_the_declaration():
    params = DocumentedTool().declaration()["parameters"]

    assert "description" not in params
    # Field descriptions are deliberate prompt text and must survive.
    assert params["properties"]["query"]["description"] == "A focused search phrase."
