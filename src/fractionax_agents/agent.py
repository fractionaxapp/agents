from __future__ import annotations

from .llm import run_tool_loop
from .tools import TOOLS, execute_tool

SYSTEM_PROMPT = (
    "You are the FractionAX investing assistant. Help users understand fractional "
    "ownership and quote investments. Use the get_quote tool for any pricing question. "
    "Be concise and accurate; never invent prices."
)


def run_agent(prompt: str, *, max_turns: int = 8) -> str:
    """Run the agentic tool-use loop until the model produces a final answer.

    Anthropic (Claude) is primary; when Claude is unavailable the whole loop fails
    over to MiniMax. Tool execution and the loop live in
    :mod:`fractionax_agents.llm`.
    """
    return run_tool_loop(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        tools=TOOLS,
        execute=execute_tool,
        max_turns=max_turns,
    )
