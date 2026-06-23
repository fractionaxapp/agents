"""LLM provider access with automatic failover.

Anthropic (Claude) is the primary provider. When Claude is unavailable —
connection errors, timeouts, rate limits, or 5xx/overloaded responses — calls
transparently fail over to MiniMax via its OpenAI-compatible API. Both the
structured-extraction path (Copilot intent + memo) and the agentic tool loop
support both providers, so the Copilot keeps working when one provider is down.

Set ``MINIMAX_API_KEY`` to enable the fallback. You can also run on MiniMax alone
by leaving ``ANTHROPIC_API_KEY`` unset. Failover only triggers on *availability*
errors — client errors (bad request, auth) re-raise, since they would fail on any
provider too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, cast

import anthropic
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam
from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger("fractionax_agents.llm")

ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]

# HTTP statuses from Anthropic that mean "temporarily unavailable" — worth failing
# over. Excludes 4xx client errors (400/401/403/404) which would fail anywhere.
_UNAVAILABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

_NO_PROVIDER = "No LLM provider configured: set ANTHROPIC_API_KEY and/or MINIMAX_API_KEY."


def is_anthropic_unavailable(exc: Exception) -> bool:
    """True if ``exc`` means Claude is temporarily unavailable (failover-worthy)."""
    if isinstance(
        exc,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in _UNAVAILABLE_STATUS
    return False


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool defs to OpenAI/MiniMax function defs."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# --------------------------------------------------------------------------- #
# Structured extraction (forced tool/function call -> validated Pydantic model)
# --------------------------------------------------------------------------- #


def _extract_anthropic[T: BaseModel](
    settings: Settings,
    *,
    system: str,
    user: str,
    model_cls: type[T],
    tool_name: str,
    tool_description: str,
) -> T:
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
    raise RuntimeError(f"Claude did not return a {tool_name!r} tool call")


def _extract_minimax[T: BaseModel](
    settings: Settings,
    *,
    system: str,
    user: str,
    model_cls: type[T],
    tool_name: str,
    tool_description: str,
) -> T:
    from openai import OpenAI

    client = OpenAI(api_key=settings.minimax_api_key, base_url=settings.minimax_base_url)
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": model_cls.model_json_schema(),
        },
    }
    response = client.chat.completions.create(
        model=settings.minimax_model,
        max_tokens=settings.max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=cast("Any", [tool]),
        tool_choice=cast("Any", {"type": "function", "function": {"name": tool_name}}),
    )
    message = response.choices[0].message
    if message.tool_calls:
        # MiniMax returns function tool calls; narrow the SDK's call union.
        call = cast("Any", message.tool_calls[0])
        return model_cls.model_validate_json(call.function.arguments)
    raise RuntimeError(f"MiniMax did not return a {tool_name!r} tool call")


def extract_structured[T: BaseModel](
    *,
    system: str,
    user: str,
    model_cls: type[T],
    tool_name: str,
    tool_description: str,
) -> T:
    """Coerce a reply into ``model_cls`` via a forced tool call, with failover."""
    settings = get_settings()
    if settings.anthropic_api_key:
        try:
            return _extract_anthropic(
                settings,
                system=system,
                user=user,
                model_cls=model_cls,
                tool_name=tool_name,
                tool_description=tool_description,
            )
        except Exception as exc:
            if not (settings.minimax_api_key and is_anthropic_unavailable(exc)):
                raise
            logger.warning(
                "Claude unavailable (%s); failing over to MiniMax for %s",
                type(exc).__name__,
                tool_name,
            )
    if settings.minimax_api_key:
        return _extract_minimax(
            settings,
            system=system,
            user=user,
            model_cls=model_cls,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    raise RuntimeError(_NO_PROVIDER)


# --------------------------------------------------------------------------- #
# Agentic tool loop
# --------------------------------------------------------------------------- #


def _text_of(content: Any) -> str:
    return "".join(b.text for b in content if getattr(b, "type", None) == "text")


_MAX_TURNS_REACHED = "Reached the maximum number of reasoning steps without a final answer."


def _loop_anthropic(
    settings: Settings,
    *,
    system: str,
    prompt: str,
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_turns: int,
) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=settings.agent_model,
            max_tokens=settings.max_tokens,
            system=system,
            tools=cast("list[ToolParam]", tools),
            thinking={"type": "adaptive"},
            messages=cast("list[MessageParam]", messages),
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return _text_of(response.content)

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = execute(block.name, dict(cast("dict[str, Any]", block.input)))
                content, is_error = json.dumps(result), False
            except Exception as exc:
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
    return _MAX_TURNS_REACHED


def _loop_minimax(
    settings: Settings,
    *,
    system: str,
    prompt: str,
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_turns: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.minimax_api_key, base_url=settings.minimax_base_url)
    oai_tools = _to_openai_tools(tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=settings.minimax_model,
            max_tokens=settings.max_tokens,
            messages=cast("Any", messages),
            tools=cast("Any", oai_tools),
            tool_choice="auto",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or ""
        # MiniMax returns function tool calls; narrow the SDK's call union.
        tool_calls = cast("list[Any]", message.tool_calls)

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                result = execute(tc.function.name, json.loads(tc.function.arguments or "{}"))
                content = json.dumps(result)
            except Exception as exc:
                content = json.dumps({"error": str(exc)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return _MAX_TURNS_REACHED


def run_tool_loop(
    *,
    system: str,
    prompt: str,
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_turns: int = 8,
) -> str:
    """Run an agentic tool-use loop, failing over from Claude to MiniMax."""
    settings = get_settings()
    if settings.anthropic_api_key:
        try:
            return _loop_anthropic(
                settings,
                system=system,
                prompt=prompt,
                tools=tools,
                execute=execute,
                max_turns=max_turns,
            )
        except Exception as exc:
            if not (settings.minimax_api_key and is_anthropic_unavailable(exc)):
                raise
            logger.warning(
                "Claude unavailable (%s); failing over to MiniMax for the chat loop",
                type(exc).__name__,
            )
    if settings.minimax_api_key:
        return _loop_minimax(
            settings,
            system=system,
            prompt=prompt,
            tools=tools,
            execute=execute,
            max_turns=max_turns,
        )
    raise RuntimeError(_NO_PROVIDER)
