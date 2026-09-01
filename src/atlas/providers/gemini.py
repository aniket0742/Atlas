"""Gemini generation provider.

Verified against google-genai 2.21.0. The SDK exposes two generation surfaces --
`client.models.generate_content` and a newer `client.interactions` API. This uses
the former: it is the stable, widely-documented surface, and it accepts a
`response_schema`, which is the feature the answering path depends on.

Free-tier specifics that shape this code:

  * Rate limits are low and shared, so HTTP 429 is a normal operating condition,
    not an exception. Retrying with backoff is required for the system to work
    at all, which is why it is here in Phase 1 rather than deferred to the
    "reliability" phase.
  * 503 (model overloaded) also occurs and is retryable.
  * 400 (bad request) and 403 (bad key) are not retryable and retrying them
    just burns quota, so they fail fast.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from atlas.core.models import TokenUsage
from atlas.providers.base import LLMError, LLMTimeout

logger = logging.getLogger(__name__)

# Status codes worth retrying: rate limit, and the server-side transients.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        max_attempts: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Get a free-tier key from "
                "https://aistudio.google.com/apikey, or set ATLAS_LLM_PROVIDER=fake "
                "to run without one."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._max_attempts = max_attempts

    @property
    def model_id(self) -> str:
        return self._model

    def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: type[Any],
        timeout_seconds: float | None = None,
    ) -> tuple[Any, TokenUsage]:
        budget = timeout_seconds or self._timeout_seconds
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            # SDK expects milliseconds.
            http_options=types.HttpOptions(timeout=int(budget * 1000)),
        )

        deadline = time.monotonic() + budget
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            if time.monotonic() >= deadline:
                raise LLMTimeout(f"Exceeded {budget}s budget before attempt {attempt}")
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
            except genai_errors.APIError as exc:
                status = getattr(exc, "code", None)
                if status not in _RETRYABLE_STATUS or attempt == self._max_attempts:
                    raise LLMError(f"Gemini call failed (status={status}): {exc}") from exc
                last_error = exc
                # Exponential backoff with full jitter. Jitter matters here
                # because ingestion fans out concurrent calls that would
                # otherwise retry in lockstep and re-trigger the same 429.
                delay = min(2**attempt, 16) * random.random()
                remaining = deadline - time.monotonic()
                if delay >= remaining:
                    raise LLMTimeout(
                        f"Retry backoff would exceed the {budget}s budget"
                    ) from exc
                logger.warning(
                    "gemini retryable error status=%s attempt=%s/%s sleeping=%.2fs",
                    status,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                time.sleep(delay)
                continue
            except Exception as exc:  # transport-level failure
                raise LLMError(f"Gemini transport failure: {exc}") from exc

            parsed = response.parsed
            if parsed is None:
                # Happens when the model hit max_output_tokens mid-JSON, or a
                # safety filter truncated the candidate. Surfacing it as an
                # error is correct: a half-parsed answer must never be treated
                # as grounded.
                finish = None
                if response.candidates:
                    finish = getattr(response.candidates[0], "finish_reason", None)
                raise LLMError(
                    f"Gemini returned no parseable structured output "
                    f"(finish_reason={finish}). Raw text: {(response.text or '')[:200]!r}"
                )

            return parsed, _usage(response)

        raise LLMError(f"Exhausted {self._max_attempts} attempts: {last_error}")


def _usage(response: types.GenerateContentResponse) -> TokenUsage:
    meta = response.usage_metadata
    if meta is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=meta.prompt_token_count or 0,
        output_tokens=meta.candidates_token_count or 0,
        # Thinking tokens are billed and are invisible in the response text, so
        # they are tracked separately rather than folded into output tokens.
        thinking_tokens=meta.thoughts_token_count or 0,
        total_tokens=meta.total_token_count or 0,
    )


def list_available_models(api_key: str) -> list[str]:
    """Models this key can actually reach.

    Free-tier model availability changes over time and is not reliably
    documented per-tier, so `atlas models` resolves it empirically instead of
    the project hard-coding a claim about what is free.
    """
    client = genai.Client(api_key=api_key)
    names: list[str] = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append((model.name or "").removeprefix("models/"))
    return sorted(n for n in names if n)
