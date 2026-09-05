"""Deterministic offline providers.

These exist so the full pipeline -- ingest, chunk, embed, index, retrieve,
answer -- is runnable and testable with no API key, no model download and no
network. That matters for CI, and it matters for being able to assert on
retrieval plumbing without asserting on model behaviour.

The fake embedder is not a toy: it is a hashed bag-of-words projection, so
lexically similar texts really do land near each other. That makes integration
tests meaningful rather than vacuous. It is, obviously, useless for measuring
retrieval *quality* -- eval runs use a real model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from atlas.core.models import TokenUsage
from atlas.providers.base import (
    AgentMessage,
    AgentTurn,
    ToolCall,
)

_WORD = re.compile(r"[a-z0-9]+")

FAKE_DIMENSIONS = 384


class FakeEmbeddingProvider:
    """Hashed bag-of-words embeddings. Deterministic across processes and runs."""

    def __init__(self, dimensions: int = FAKE_DIMENSIONS, max_tokens: int = 512) -> None:
        self._dimensions = dimensions
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return f"fake-hashed-bow-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        # Whitespace tokens are a poor proxy for BPE tokens, but the fake only
        # needs to be self-consistent: chunking assertions are about structure.
        return len(text.split())

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            # Sign from an independent byte so unrelated words do not all add
            # positively and collapse every vector toward the same direction.
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # An empty or purely-punctuation string. Return a fixed unit vector
            # rather than zeros, which have undefined cosine distance.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLMProvider:
    """Answers by extractive quoting, never by generation.

    It returns the first sentence of the highest-ranked evidence block and cites
    it. This lets tests assert on the citation-validation and refusal paths
    deterministically. When handed no evidence it refuses, which is the exact
    behaviour the real prompt is meant to produce.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "fake-extractive"

    def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: type[Any],
        timeout_seconds: float | None = None,
    ) -> tuple[Any, TokenUsage]:
        self.calls.append({"system": system_instruction, "prompt": prompt})

        # The prompt embeds evidence as blocks tagged with their chunk id; parse
        # them back out rather than re-deriving them from state. The tag also
        # carries provenance attributes after the id, hence [^>]*.
        blocks = re.findall(
            r"<evidence id=\"([0-9a-f-]{36})\"[^>]*>\n(.*?)\n</evidence>", prompt, re.DOTALL
        )

        if not blocks:
            return (
                response_schema(
                    answer="I could not find evidence for that in the knowledge base.",
                    citations=[],
                    sufficient_evidence=False,
                ),
                TokenUsage(prompt_tokens=len(prompt.split()), output_tokens=12, total_tokens=0),
            )

        chunk_id, body = blocks[0]
        first_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[0][:400]
        return (
            response_schema(
                answer=first_sentence,
                citations=[{"chunk_id": chunk_id, "quote": first_sentence}],
                sufficient_evidence=True,
            ),
            TokenUsage(
                prompt_tokens=len(prompt.split()),
                output_tokens=len(first_sentence.split()),
                total_tokens=len(prompt.split()) + len(first_sentence.split()),
            ),
        )


class ScriptedToolCallingLLM:
    """A tool-calling model whose turns are written by the test.

    The agent loop's job is to enforce bounds, feed results back and degrade
    safely. None of that should be asserted through a real model, whose
    behaviour is the one thing in the system that is not deterministic -- a test
    that depends on Gemini choosing to search twice is a test that fails for
    reasons unrelated to the loop.

    So the script is a list of turns. Each entry is either a list of
    (tool_name, arguments) pairs to request, or a string to finish with. Running
    off the end finishes, which mirrors a model that has decided it is done.
    """

    def __init__(
        self,
        script: list[Any] | None = None,
        *,
        model_id: str = "fake-agent",
        error: Exception | None = None,
        usage_per_turn: TokenUsage | None = None,
    ) -> None:
        self._script = list(script or [])
        self._model_id = model_id
        # Raised on every call, for testing provider failure and the fallback.
        self._error = error
        self._usage = usage_per_turn or TokenUsage(
            prompt_tokens=100, output_tokens=20, thinking_tokens=5, total_tokens=125
        )
        #: Every call's history, so a test can assert what the model was shown.
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_with_tools(
        self,
        *,
        system_instruction: str,
        history: list[AgentMessage],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> AgentTurn:
        self.calls.append(
            {
                "system": system_instruction,
                "history": list(history),
                "tools": [t["name"] for t in tools],
                "timeout_seconds": timeout_seconds,
            }
        )

        if self._error is not None:
            raise self._error

        index = len(self.calls) - 1
        if index >= len(self._script):
            return AgentTurn(text="done", tool_calls=(), usage=self._usage)

        turn = self._script[index]
        if isinstance(turn, str):
            return AgentTurn(text=turn, tool_calls=(), usage=self._usage)

        return AgentTurn(
            text=None,
            tool_calls=tuple(
                ToolCall(name=name, arguments=dict(args)) for name, args in turn
            ),
            usage=self._usage,
        )
