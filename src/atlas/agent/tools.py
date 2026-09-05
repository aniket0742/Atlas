"""Tool framework for the agent.

No tools live here -- this is the machinery every tool runs inside. It is built
before any tool exists so that the guarantees are structural rather than
something each tool author has to remember.

## Where the guarantees live

A tool implements one method: `execute`. It does **not** validate its own
arguments, enforce its own timeout, check its own permissions, or write its own
logs. All of that happens in `ToolRegistry.invoke`, around the call.

That placement is the point. If validation were the tool's job, the third tool
someone adds would forget it, and the failure would be a malformed database
query rather than a clean rejection. Putting the guard in the registry means a
tool cannot opt out of it by being written carelessly.

## Failures are returned, not raised

Tool arguments are produced by a language model, so bad arguments are a *normal
operating condition*, not an exception. A model that calls `search(quer="x")`
should be told what it got wrong and given the chance to retry -- that is a
one-turn recovery. Raising instead would abort the whole request over a typo.

So `invoke` returns a `ToolResult` for every outcome including failure, and the
agent loop feeds it back to the model as an ordinary function response. The only
things that propagate as exceptions are programming errors in Atlas itself.

## Authorization is carried, never derived

`ToolContext` is built by the server from the authenticated request and passed
in. The model supplies only a tool *name* and its *arguments*, so identity is
not on the path the model can influence -- it is a separate parameter, not a
field it could set.

Three things make that structural rather than aspirational:

1. `ToolContext` is frozen, so a tool cannot widen its own scope.
2. A tool may not declare `tenant_id` (or any other reserved name) as an
   argument. Registration refuses it, so the attack has nowhere to land.
3. Argument models forbid unknown fields, so an injected extra key is a visible
   `invalid_arguments` failure rather than a silently ignored one.

The residual risk is unchanged from ADR-0010 and worth stating plainly: a
retrieved document can still say "search for X instead", and the model may
comply. What it cannot do is change *whose* data is searched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

# Arguments are logged for debugging, but a model can emit a very long string.
_MAX_LOGGED_ARGS = 300

#: Tool names the provider will accept.
_VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

#: Argument names a tool may never declare.
#:
#: This is the load-bearing half of the authorization boundary. Everything that
#: identifies *who* a call acts for arrives in `ToolContext`, which the server
#: builds. If a tool could declare `tenant_id` as an argument, the model would
#: be able to supply one -- and a document saying "search tenant acme" would
#: become an instruction the model could follow. Registration refuses such a
#: tool, so the attack has nowhere to land rather than being caught later by a
#: check somebody might forget.
RESERVED_ARGUMENT_NAMES = frozenset(
    {
        "tenant",
        "tenant_id",
        "permissions",
        "permission",
        "context",
        "request_id",
        "user",
        "user_id",
        "principal",
        "scope",
        "scopes",
        "role",
        "roles",
    }
)


class ToolArgs(BaseModel):
    """Base class for every tool's arguments.

    `extra="forbid"` is the reason this exists. Pydantic's default is to ignore
    unknown fields, so a model emitting
    `{"query": "x", "tenant_id": "someone-else"}` would have the extra key
    silently dropped -- the call would succeed, and nothing would record that
    something tried to widen its own scope.

    Forbidding turns that into a visible `invalid_arguments` result that lands in
    the trace and the logs. The attempt fails *loudly*, which is the difference
    between a control and an accident.
    """

    model_config = ConfigDict(extra="forbid")


class ToolOutcome(StrEnum):
    """Why a tool call ended the way it did.

    Distinct values rather than a boolean because the agent loop reacts
    differently to each: an invalid-arguments result is worth letting the model
    retry, a denial is not, and a timeout may be worth retrying once with a
    narrower request.
    """

    OK = "ok"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    DENIED = "denied"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Who this tool call acts on behalf of.

    Frozen deliberately. This is built by the API layer from the authenticated
    request and passed down; a tool receives it read-only and cannot widen its
    own scope. Model output and retrieved document text never reach it.
    """

    tenant_id: uuid.UUID
    # Capability names the caller holds. Empty means "no special capabilities",
    # which is the correct default for an unauthenticated Phase 4 request.
    permissions: frozenset[str] = frozenset()
    # Correlates a tool call back to the request that caused it.
    request_id: str | None = None

    def has(self, permission: str) -> bool:
        return permission in self.permissions


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """A tool's return value when the model should not see all of it.

    Retrieval is the motivating case. The model needs snippets to decide
    whether to search again; the *answer* needs the full chunks with their
    offsets and provenance, and putting those in the function response would
    spend thousands of tokens per call restating text the server already holds.

    So a tool may return one of these instead of a bare value: `content` goes
    back to the model, `artifacts` stays server-side and is carried on the
    `ToolResult` for the agent loop to collect. A tool that has no such split
    just returns its value directly.
    """

    content: Any
    artifacts: Any = None


@dataclass(slots=True)
class ToolResult:
    """The outcome of one tool call, and the unit of the agent's trace."""

    tool: str
    outcome: ToolOutcome
    duration_ms: float
    # Arguments as the model supplied them, before validation, so a trace shows
    # what was actually attempted rather than what was successfully parsed.
    arguments: dict[str, Any] = field(default_factory=dict)
    content: Any = None
    error: str | None = None
    # Server-side payload from a `ToolOutput`. Deliberately excluded from
    # `for_model()`: this is the half the model must not see, so the exclusion
    # is a property of the type rather than of each caller's discipline.
    artifacts: Any = None

    @property
    def ok(self) -> bool:
        return self.outcome is ToolOutcome.OK

    def for_model(self) -> dict[str, Any]:
        """The payload sent back as the function response.

        Errors are included in full because the model is expected to act on
        them: "query: Field required" is a correctable instruction, whereas a
        bare failure flag leaves it guessing.
        """
        if self.ok:
            return {"ok": True, "result": self.content}
        return {"ok": False, "error": self.error or self.outcome.value}


class Tool(ABC):
    """One capability the agent can invoke.

    Subclasses declare metadata and implement `execute`. Everything protective
    happens in the registry, so `execute` can assume its arguments are already
    validated and that a timeout is already running.
    """

    #: Stable identifier the model calls. Must match ^[a-zA-Z0-9_-]+$.
    name: ClassVar[str]
    #: Shown to the model. This is prompt text and materially affects routing --
    #: it should say when to use the tool, not merely what it does.
    description: ClassVar[str]
    #: Arguments model. Must subclass `ToolArgs`, which forbids unknown fields.
    #: Doubles as the JSON schema sent to the provider and as the validator
    #: applied before execution.
    Args: ClassVar[type[ToolArgs]]
    #: Per-tool, because a local database query and a remote API call do not
    #: deserve the same budget.
    timeout_seconds: ClassVar[float] = 10.0
    #: When set, `ToolContext.permissions` must contain it. None means the tool
    #: is available to any caller that can reach the agent at all.
    required_permission: ClassVar[str | None] = None

    @abstractmethod
    async def execute(self, context: ToolContext, args: Any) -> Any:
        """Do the work and return a JSON-serialisable result.

        `args` is a validated instance of `Args`. Raising is acceptable: the
        registry converts exceptions into an ERROR result rather than letting
        them escape into the agent loop.
        """

    def declaration(self) -> dict[str, Any]:
        """Provider-agnostic description of this tool.

        Deliberately a plain dict rather than a provider type. The Gemini SDK
        accepts this shape directly, and keeping it neutral means adding a
        second provider does not touch any tool.
        """
        schema = _for_provider(self.Args.model_json_schema())
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }


#: JSON-schema keys stripped before a schema is sent to a provider.
#:
#: `additionalProperties` is the load-bearing entry and the reason this function
#: exists. `ToolArgs` sets `extra="forbid"`, so pydantic emits
#: `additionalProperties: false` -- and the Gemini API rejects the declaration
#: outright with a 400, because its function-calling schema dialect is a subset
#: of JSON Schema that has no such field.
#:
#: Stripping it does **not** weaken the boundary. `extra="forbid"` is enforced
#: by `model_validate` inside `invoke`, before the tool runs; the schema key
#: only ever *told* the provider about the rule. An extra key still produces a
#: visible `invalid_arguments` failure. What is lost is advance notice to the
#: model, which is a prompt-quality matter, not a control.
#:
#: `title` and the class-docstring `description` are dropped for cost: pydantic
#: generates both, neither is written for the model, and the docstring in
#: particular is maintainer-facing rationale that would cost hundreds of tokens
#: on every request.
_STRIPPED_SCHEMA_KEYS = ("additionalProperties", "title", "$schema")


def _for_provider(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a pydantic JSON schema to what a provider will accept."""
    cleaned = _walk(schema)
    # Root only. This is the `Args` class docstring; the per-field descriptions
    # nested below it are deliberate prompt text and are kept.
    cleaned.pop("description", None)
    return cleaned


def _walk(node: Any) -> Any:
    """Strip provider-unsupported keywords everywhere in the tree.

    Nested because an object-typed argument produces a nested schema (and a
    `$defs` entry), and a rejected keyword anywhere fails the whole declaration.

    The `properties`/`$defs` special case is load-bearing: keys there are
    *names*, not schema keywords. Filtering them blindly would delete an
    argument a tool legitimately called `title`.
    """
    if isinstance(node, list):
        return [_walk(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRIPPED_SCHEMA_KEYS:
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            cleaned[key] = {name: _walk(sub) for name, sub in value.items()}
        else:
            cleaned[key] = _walk(value)
    return cleaned


def _has_ref(node: Any) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_ref(item) for item in node)
    return False



class ToolRegistry:
    """Holds the available tools and is the only way to call one."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool, refusing any that could accept identity from the model.

        These checks fail at registration -- import time in practice -- rather
        than at call time. A tool that could take a `tenant_id` argument is a
        design error, and a design error should be impossible to deploy, not
        caught by a runtime guard on the day someone probes it.
        """
        if not _VALID_NAME.match(tool.name or ""):
            raise ValueError(
                f"Tool name {tool.name!r} must match {_VALID_NAME.pattern} "
                "to be callable by the provider"
            )
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")

        args_model = getattr(tool, "Args", None)
        if not (isinstance(args_model, type) and issubclass(args_model, ToolArgs)):
            raise TypeError(
                f"Tool {tool.name!r}: Args must subclass ToolArgs, which forbids "
                "unknown fields. A plain BaseModel silently drops extra keys, so "
                "an attempt to smuggle in a tenant would succeed unnoticed."
            )

        # Field names and their aliases both reach the model, so both are checked.
        declared: set[str] = set()
        for field_name, info in args_model.model_fields.items():
            declared.add(field_name.lower())
            if info.alias:
                declared.add(info.alias.lower())

        reserved = sorted(declared & RESERVED_ARGUMENT_NAMES)
        if reserved:
            raise ValueError(
                f"Tool {tool.name!r} declares reserved argument(s) {reserved}. "
                "Identity and authorization come from ToolContext, which the "
                "server builds; a tool that accepts them as arguments lets the "
                "model choose who it acts as."
            )

        if _has_ref(args_model.model_json_schema()):
            # A nested model produces `$defs` and `$ref`. Whether a given
            # provider resolves those is not something Atlas has verified, and
            # the failure mode if it does not is a 400 at request time. Refusing
            # here keeps that discovery at boot, where the registry puts every
            # other structural check. Flatten the arguments, or inline the
            # nested schema, when a tool genuinely needs one.
            raise ValueError(
                f"Tool {tool.name!r} declares a nested model, which produces a "
                "$ref in its schema. Provider support for $ref is unverified; "
                "use flat arguments instead."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def declarations(self, context: ToolContext | None = None) -> list[dict[str, Any]]:
        """Tool declarations for the provider.

        When a context is supplied, tools the caller lacks permission for are
        omitted entirely rather than advertised and then denied. A model cannot
        be tempted by, or waste a turn on, a tool it was never shown.
        """
        return [
            tool.declaration()
            for _, tool in sorted(self._tools.items())
            if context is None
            or tool.required_permission is None
            or context.has(tool.required_permission)
        ]

    async def invoke(
        self, name: str, arguments: dict[str, Any] | None, context: ToolContext
    ) -> ToolResult:
        """Validate, authorise, time, run and log one tool call.

        Never raises for a bad call. Every failure mode is a `ToolResult` the
        agent loop can hand back to the model.
        """
        raw_args = dict(arguments or {})
        started = time.perf_counter()

        def finish(
            outcome: ToolOutcome,
            *,
            content: Any = None,
            error: str | None = None,
            artifacts: Any = None,
        ):
            duration = (time.perf_counter() - started) * 1000
            result = ToolResult(
                tool=name,
                outcome=outcome,
                duration_ms=duration,
                arguments=raw_args,
                content=content,
                error=error,
                artifacts=artifacts,
            )
            self._log(result, context)
            return result

        tool = self._tools.get(name)
        if tool is None:
            # Models invent tool names. Telling it which names exist turns a
            # dead end into a correctable turn.
            return finish(
                ToolOutcome.UNKNOWN_TOOL,
                error=f"No tool named {name!r}. Available tools: {', '.join(self.names())}.",
            )

        if tool.required_permission is not None and not context.has(tool.required_permission):
            # Deliberately does not name the missing permission: that is an
            # operator-facing detail, and echoing it to the model puts the
            # authorization model into text the model can reason about.
            return finish(
                ToolOutcome.DENIED,
                error=f"Not permitted to use {name!r}.",
            )

        try:
            args = tool.Args.model_validate(raw_args)
        except ValidationError as exc:
            return finish(ToolOutcome.INVALID_ARGUMENTS, error=_format_validation_error(exc))

        try:
            content = await asyncio.wait_for(
                tool.execute(context, args), timeout=tool.timeout_seconds
            )
        except TimeoutError:
            return finish(
                ToolOutcome.TIMEOUT,
                error=f"{name!r} exceeded its {tool.timeout_seconds:g}s budget.",
            )
        except Exception as exc:
            # The tool's fault, not the model's. The message is returned so the
            # model can decide to try something else, and logged with a
            # traceback so an operator can see what actually broke.
            logger.exception("tool %s raised", name)
            return finish(ToolOutcome.ERROR, error=f"{type(exc).__name__}: {exc}")

        if isinstance(content, ToolOutput):
            return finish(ToolOutcome.OK, content=content.content, artifacts=content.artifacts)
        return finish(ToolOutcome.OK, content=content)

    def _log(self, result: ToolResult, context: ToolContext) -> None:
        try:
            rendered = json.dumps(result.arguments, default=str)
        except (TypeError, ValueError):
            rendered = repr(result.arguments)
        if len(rendered) > _MAX_LOGGED_ARGS:
            rendered = rendered[:_MAX_LOGGED_ARGS] + "...(truncated)"

        # One line per call carrying name, arguments, duration and outcome --
        # enough to reconstruct what the agent did without the response bodies,
        # which can contain customer document text.
        log = logger.info if result.ok else logger.warning
        log(
            "tool=%s outcome=%s duration_ms=%.1f tenant=%s request=%s args=%s%s",
            result.tool,
            result.outcome.value,
            result.duration_ms,
            context.tenant_id,
            context.request_id or "-",
            rendered,
            f" error={result.error}" if result.error else "",
        )


def _format_validation_error(exc: ValidationError) -> str:
    """Render pydantic's error into something a model can act on.

    Pydantic's default rendering carries URLs and input echoes that cost tokens
    and add nothing the model can use. "query: Field required" is the actionable
    part.
    """
    parts = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "Invalid arguments -- " + "; ".join(parts)
