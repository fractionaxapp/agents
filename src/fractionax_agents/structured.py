"""Structured output via forced tool use.

Claude is given a single tool whose input schema is the target Pydantic model and
is forced to call it (`tool_choice`), so the response is guaranteed to be a tool
call we can validate into that model. This is the documented, model-agnostic way
to get typed output without depending on the SDK's evolving `parse()` surface.
"""

from __future__ import annotations

from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam
from pydantic import BaseModel

from .config import get_settings


def extract[T: BaseModel](
    *,
    system: str,
    user: str,
    model_cls: type[T],
    tool_name: str,
    tool_description: str,
) -> T:
    """Call Claude and coerce the reply into ``model_cls`` via a forced tool call."""
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    tool: dict[str, Any] = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": model_cls.model_json_schema(),
    }
    response = client.messages.create(
        model=settings.agent_model,
        max_tokens=settings.max_tokens,
        system=system,
        tools=cast("list[ToolParam]", [tool]),
        tool_choice={"type": "tool", "name": tool_name},
        messages=cast("list[MessageParam]", [{"role": "user", "content": user}]),
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return model_cls.model_validate(block.input)
    raise RuntimeError(f"Model did not return a {tool_name!r} tool call")
