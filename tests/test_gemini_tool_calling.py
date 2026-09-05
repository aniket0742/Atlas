"""Translation between Atlas's neutral messages and the Gemini wire format.

Every test here is a regression test. All of them cover bugs that the 245 tests
written before the first live run passed straight through, because a fake
provider validates the loop's logic and says nothing about whether the real API
will accept what the loop produces. No network: these exercise the translation
functions and the declaration shape, which is where both live failures were.
"""

from __future__ import annotations

import pytest
from pydantic import Field

from atlas.agent.knowledge_base import SearchKnowledgeBaseTool
from atlas.agent.tools import Tool, ToolArgs, ToolRegistry
from atlas.providers.base import ModelMessage, ToolCall, ToolResultMessage, UserMessage

types = pytest.importorskip("google.genai.types")

from atlas.providers.gemini import _to_content  # noqa: E402

# ---------------------------------------------------------------------------
# Declaration shape
# ---------------------------------------------------------------------------


def test_declarations_omit_additional_properties():
    """`extra="forbid"` emits a key the Gemini API rejects with a 400.

    The Step 2 test asserting the SDK accepts a declaration passed throughout,
    because the SDK's `FunctionDeclaration` type is happy to hold the field --
    it is the API that refuses it. Constructing the type is not evidence the
    request will succeed.
    """
    params = SearchKnowledgeBaseTool(None).declaration()["parameters"]

    assert "additionalProperties" not in params
    assert "$schema" not in params


def test_stripping_the_schema_key_does_not_relax_validation():
    """The rule is enforced by pydantic, not by what the provider was told."""
    import asyncio
    import uuid

    from atlas.agent.tools import ToolContext, ToolOutcome

    tool = SearchKnowledgeBaseTool(None)
    registry = ToolRegistry([tool])

    result = asyncio.run(
        registry.invoke(
            "search_knowledge_base",
            {"query": "x", "tenant_id": "someone-else"},
            ToolContext(tenant_id=uuid.uuid4()),
        )
    )

    assert result.outcome is ToolOutcome.INVALID_ARGUMENTS


def test_a_nested_model_argument_is_refused_at_registration():
    """It produces a `$ref`, whose provider support Atlas has not verified.

    Refusing at boot rather than shipping a declaration that may fail with a
    400 at request time, which is where the registry puts every other
    structural check.
    """

    class Inner(ToolArgs):
        name: str

    class OuterArgs(ToolArgs):
        filter: Inner

    class NestedTool(Tool):
        name = "nested"
        description = "d"
        Args = OuterArgs

        async def execute(self, context, args):
            return None

    with pytest.raises(ValueError, match=r"\$ref"):
        ToolRegistry([NestedTool()])


def test_an_argument_named_title_is_not_mistaken_for_a_schema_keyword():
    """`title` is stripped as a keyword, but it is a legal argument name.

    Filtering keys blindly would delete the argument itself and produce a tool
    the model can never call correctly.
    """

    class TitledArgs(ToolArgs):
        title: str = Field(description="The document title to look up.")

    class TitledTool(Tool):
        name = "titled"
        description = "d"
        Args = TitledArgs

        async def execute(self, context, args):
            return None

    params = TitledTool().declaration()["parameters"]

    assert "title" in params["properties"]
    assert params["properties"]["title"]["description"] == "The document title to look up."
    # The keyword form is still gone from the schema level.
    assert "title" not in params


def test_a_declaration_is_accepted_by_the_sdk_type():
    decl = SearchKnowledgeBaseTool(None).declaration()
    fd = types.FunctionDeclaration(
        name=decl["name"], description=decl["description"], parameters=decl["parameters"]
    )
    assert fd.parameters.required == ["query"]


# ---------------------------------------------------------------------------
# Conversation translation
# ---------------------------------------------------------------------------


def test_a_user_message_becomes_a_user_turn():
    content = _to_content(UserMessage(text="what is the refund window?"))
    assert content.role == "user"
    assert content.parts[0].text == "what is the refund window?"


def test_a_function_response_is_sent_with_the_user_role():
    """It is input to the model's next turn, not something the model said."""
    content = _to_content(ToolResultMessage(name="search", response={"ok": True}))

    assert content.role == "user"
    assert content.parts[0].function_response.name == "search"
    assert content.parts[0].function_response.response == {"ok": True}


def test_a_model_turn_carries_its_function_calls():
    call = ToolCall(name="search", arguments={"query": "x"})
    content = _to_content(ModelMessage(text="looking", tool_calls=(call,)))

    assert content.role == "model"
    assert content.parts[0].text == "looking"
    assert content.parts[1].function_call.name == "search"
    assert content.parts[1].function_call.args == {"query": "x"}


def test_the_thought_signature_is_echoed_back_on_the_function_call():
    """Without it a 3.x thinking model rejects the follow-up turn with a 400.

    This is the bug that made every loop longer than one iteration fail: the
    first turn worked, the model asked for a search, and reconstructing that
    turn from the neutral types dropped an opaque field the API requires.
    """
    signature = b"opaque-provider-bytes"
    content = _to_content(
        ModelMessage(tool_calls=(ToolCall(name="search", provider_state=signature),))
    )

    assert content.parts[0].thought_signature == signature


def test_a_call_without_a_signature_is_still_translatable():
    """Not every provider or model emits one, so absence must not crash."""
    content = _to_content(ModelMessage(tool_calls=(ToolCall(name="search"),)))
    assert content.parts[0].thought_signature is None


def test_an_empty_model_turn_still_produces_a_part():
    """The API rejects a content with no parts at all."""
    content = _to_content(ModelMessage())
    assert content.parts


def test_an_unknown_message_type_is_refused_loudly():
    with pytest.raises(TypeError):
        _to_content(object())  # type: ignore[arg-type]
