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
