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
from atlas.providers.base import (
    AgentMessage,
    AgentTurn,
    LLMError,
    LLMTimeout,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

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

        response = self._call_with_retries(contents=prompt, config=config, budget=budget)

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


    def generate_with_tools(
        self,
        *,
        system_instruction: str,
        history: list[AgentMessage],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> AgentTurn:
        """One turn of a tool-calling conversation.

        Note what is *not* here: automatic function calling. The SDK can execute
        Python callables on the model's behalf and loop internally, and that is
        exactly the wrong shape for Atlas -- it would run tools outside
        `ToolRegistry.invoke`, which is where the timeout, the permission check,
        the argument validation and the audit log live. Passing declarations
        rather than callables means the SDK cannot call anything, and the flag
        is set explicitly so a future SDK default cannot change that quietly.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=tool["parameters"],
                        )
                        for tool in tools
                    ]
                )
            ]
            if tools
            else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(
                timeout=int((timeout_seconds or self._timeout_seconds) * 1000)
            ),
        )
        response = self._call_with_retries(
            contents=[_to_content(message) for message in history],
            config=config,
            budget=timeout_seconds or self._timeout_seconds,
        )

        calls = tuple(
            ToolCall(
                name=call.function_call.name or "",
                arguments=dict(call.function_call.args or {}),
                id=call.function_call.id,
                # Carried back verbatim on the next turn. Read from the *part*
                # rather than from `response.function_calls`, which flattens to
                # FunctionCall objects and drops the signature with it.
                provider_state=call.thought_signature,
            )
            for call in _function_call_parts(response)
        )

        # Assembled from the text parts directly. `response.text` warns loudly
        # on every turn that also carries function calls -- which, in a tool
        # loop, is most of them.
        text = _text_of(response)

        finish = None
        if response.candidates:
            finish = getattr(response.candidates[0], "finish_reason", None)

        return AgentTurn(
            text=text,
            tool_calls=calls,
            usage=_usage(response),
            finish_reason=str(finish) if finish is not None else None,
        )

    def _call_with_retries(self, *, contents: Any, config: Any, budget: float) -> Any:
        """Retry policy shared by both generation surfaces.

        Extracted when tool calling arrived rather than duplicated: 429 is a
        normal operating condition here too, and an agent loop makes several
        calls per request, so it meets the rate limit more often than answering
        does -- not less.
        """
        deadline = time.monotonic() + budget
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            if time.monotonic() >= deadline:
                raise LLMTimeout(f"Exceeded {budget}s budget before attempt {attempt}")
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except genai_errors.APIError as exc:
                status = getattr(exc, "code", None)
                if status not in _RETRYABLE_STATUS or attempt == self._max_attempts:
                    raise LLMError(f"Gemini call failed (status={status}): {exc}") from exc
                last_error = exc
                delay = min(2**attempt, 16) * random.random()
                remaining = deadline - time.monotonic()
                if delay >= remaining:
                    raise LLMTimeout(f"Retry backoff would exceed the {budget}s budget") from exc
                logger.warning(
                    "gemini retryable error status=%s attempt=%s/%s sleeping=%.2fs",
                    status,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                time.sleep(delay)
            except Exception as exc:  # transport-level failure
                raise LLMError(f"Gemini transport failure: {exc}") from exc

        raise LLMError(f"Exhausted {self._max_attempts} attempts: {last_error}")


def _text_of(response: types.GenerateContentResponse) -> str | None:
    """Concatenate the candidate's text parts, ignoring thoughts."""
    if not response.candidates:
        return None
    content = response.candidates[0].content
    if content is None or not content.parts:
        return None
    pieces = [
        part.text for part in content.parts if part.text and not getattr(part, "thought", False)
    ]
    return "".join(pieces) or None


def _function_call_parts(response: types.GenerateContentResponse) -> list[types.Part]:
    """Function-call parts of the first candidate, signatures intact."""
    if not response.candidates:
        return []
    content = response.candidates[0].content
    if content is None or not content.parts:
        return []
    return [part for part in content.parts if part.function_call is not None]


def _to_content(message: AgentMessage) -> types.Content:
    """Translate a neutral message into the SDK's wire shape.

    This function is the only place in Atlas that knows Gemini's conversation
    encoding, which is the point of the neutral types.
    """
    if isinstance(message, UserMessage):
        return types.Content(role="user", parts=[types.Part(text=message.text)])

    if isinstance(message, ModelMessage):
        parts: list[types.Part] = []
        if message.text:
            parts.append(types.Part(text=message.text))
        parts.extend(
            types.Part(
                function_call=types.FunctionCall(
                    name=call.name, args=dict(call.arguments), id=call.id
                ),
                # Required by 3.x thinking models; a follow-up turn without it
                # is rejected with a 400 naming the missing signature.
                thought_signature=call.provider_state,
            )
            for call in message.tool_calls
        )
        # A model turn with neither text nor calls cannot be sent; an empty
        # parts list is rejected by the API.
        return types.Content(role="model", parts=parts or [types.Part(text="")])

    if isinstance(message, ToolResultMessage):
        # Function responses are sent with the "user" role: they are input to
        # the model's next turn, not something the model said.
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=message.name, response=message.response
                )
            ],
        )

    raise TypeError(f"Unsupported message type {type(message).__name__}")


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
