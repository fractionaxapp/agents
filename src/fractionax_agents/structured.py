"""Structured output via forced tool/function calls, with provider failover.

The model is given a single tool whose input schema is the target Pydantic model
and is forced to call it, so the reply validates into that model. Anthropic is
primary; when Claude is unavailable the call fails over to MiniMax. The provider
logic lives in :mod:`fractionax_agents.llm`.
"""

from __future__ import annotations

from pydantic import BaseModel

from .llm import extract_structured


def extract[T: BaseModel](
    *,
    system: str,
    user: str,
    model_cls: type[T],
    tool_name: str,
    tool_description: str,
) -> T:
    """Call the LLM and coerce the reply into ``model_cls`` via a forced tool call."""
    return extract_structured(
        system=system,
        user=user,
        model_cls=model_cls,
        tool_name=tool_name,
        tool_description=tool_description,
    )
