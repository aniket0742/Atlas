"""Provider interfaces.

Atlas talks to two kinds of external model: an embedding model (local, CPU) and
a generation model (an API). Both sit behind narrow Protocols so that:

  * the eval harness can swap models and compare, which is the whole point of
    Section 7 of the spec;
  * tests can run offline against deterministic fakes;
  * replacing the LLM vendor is a config change, not a refactor.

The Protocols are intentionally small. Anything a provider cannot express
uniformly (Gemini's safety settings, say) is a constructor argument of the
concrete class, not a parameter on the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from atlas.core.models import TokenUsage


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    Embedding models are asymmetric: BGE-family models expect a short
    instruction prefix on queries but not on passages, and getting this wrong
    measurably degrades retrieval. The interface therefore separates the two
    call sites rather than exposing one embed().
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def max_tokens(self) -> int:
        """Model's input limit. Chunking uses this to avoid silent truncation."""
        ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LLMError(RuntimeError):
    """Generation failed in a way the caller may want to retry or surface."""


class LLMTimeout(LLMError):
    pass


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a structured response.

    Atlas never asks an LLM for free-form prose in the answering path: it asks
    for a JSON object matching a schema (answer text + citations + a refusal
    flag). Requiring structure is what makes citation validation possible at
    all, so it is part of the interface rather than an option.
    """

    @property
    def model_id(self) -> str: ...

    def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: type[Any],
        timeout_seconds: float | None = None,
    ) -> tuple[Any, TokenUsage]:
        """Return (parsed instance of response_schema, token usage).

        Raises LLMTimeout on timeout, LLMError on any other provider failure.
        """
        ...


@runtime_checkable
class RerankProvider(Protocol):
    """Scores (query, passage) pairs directly.

    A cross-encoder differs from the embedding model in kind, not just quality:
    an embedding model encodes the query and the passage independently and
    compares the results, so it never sees the two together. A cross-encoder
    reads both in one forward pass and can therefore judge whether a passage
    answers *this* question rather than whether it is about the same topic.

    The cost is that it cannot be precomputed or indexed -- every pair is a model
    call -- so it only ever runs over a shortlist that cheaper retrieval already
    produced.
    """

    @property
    def model_id(self) -> str: ...

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage, higher being better.

        Scores are comparable within a single call only. Cross-encoder outputs
        are unnormalised logits whose scale varies by query, so they order a
        shortlist but say nothing absolute about relevance.
        """
        ...


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------
#
# A separate Protocol rather than more methods on `LLMProvider`. Tool calling is
# a capability some providers lack and the answering path never needs, and
# widening the existing interface would force every provider -- including the
# offline fake the whole test suite depends on -- to implement a surface it has
# no use for. The two are composed where both are wanted, not merged.
#
# The message types below are the interface's own vocabulary. They exist so the
# agent loop can hold a multi-turn conversation without naming a vendor: the
# loop appends neutral messages, and the provider translates them at its own
# boundary. Storing the SDK's `Content` objects instead would put Gemini types
# in the loop, in the trace, and in every test.


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation the model asked for.

    `arguments` is whatever the model produced, unvalidated. Validation belongs
    to the registry, which is the only thing allowed to decide a call is
    malformed.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Providers that support parallel calls may supply one; Gemini usually does
    # not, so nothing may depend on it being present.
    id: str | None = None
    #: Opaque provider state that must be echoed back with this call when the
    #: conversation continues. Gemini 3.x thinking models reject a follow-up
    #: turn whose function-call parts have lost their `thought_signature`, so
    #: dropping it breaks every loop longer than one iteration.
    #:
    #: The loop carries this without reading it, which is what keeps the neutral
    #: types neutral: the field is deliberately untyped as to meaning, and only
    #: the provider that produced it ever interprets it.
    provider_state: Any = None


@dataclass(frozen=True, slots=True)
class UserMessage:
    text: str


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """What the model said, and what it asked to call. Either may be empty."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """The outcome of one tool call, fed back so the model can continue.

    Carries failures as readily as successes: a model that called a tool wrongly
    should be told so and given the chance to correct itself, which is why
    `ToolRegistry.invoke` returns rather than raises.
    """

    name: str
    response: dict[str, Any]
    call_id: str | None = None


AgentMessage = UserMessage | ModelMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One model response in a tool-calling conversation."""

    text: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage
    finish_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class ToolCallingLLM(Protocol):
    """Generates a turn that may request tool calls.

    Deliberately stateless: the caller owns the conversation and passes the
    whole history each time. That keeps the provider a pure function of its
    inputs, which is what makes a loop reproducible in a test and what stops a
    half-finished conversation from being stranded inside a provider object
    shared across concurrent requests.
    """

    @property
    def model_id(self) -> str: ...

    def generate_with_tools(
        self,
        *,
        system_instruction: str,
        history: list[AgentMessage],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> AgentTurn:
        """Return the model's next turn.

        `tools` are declarations in the shape `Tool.declaration()` produces.
        Raises LLMTimeout on timeout, LLMError on any other provider failure.
        """
        ...
