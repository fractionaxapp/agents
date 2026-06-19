from __future__ import annotations

import json
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam

from .config import get_settings
from .tools import TOOLS, execute_tool

SYSTEM_PROMPT = (
    "You are the FractionAX investing assistant. Help users understand fractional "
    "ownership and quote investments. Use the get_quote tool for any pricing question. "
    "Be concise and accurate; never invent prices."
)


def run_agent(prompt: str, *, max_turns: int = 8) -> str:
    """Run the agentic tool-use loop until Claude produces a final answer.

    Uses a manual loop (rather than the SDK tool runner) for explicit control over
    tool execution and error handling. Adaptive thinking lets Claude decide how much
    to reason per turn.
    """
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=settings.agent_model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=cast("list[ToolParam]", TOOLS),
            thinking={"type": "adaptive"},
            messages=cast("list[MessageParam]", messages),
        )
        # Echo the full assistant turn back (including thinking blocks) for the next request.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return _text_of(response.content)

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = execute_tool(block.name, dict(cast("dict[str, Any]", block.input)))
                content, is_error = json.dumps(result), False
            except Exception as exc:  # surface tool errors back to the model
                content, is_error = str(exc), True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Reached the maximum number of reasoning steps without a final answer."


def _text_of(content: Any) -> str:
    return "".join(b.text for b in content if getattr(b, "type", None) == "text")
