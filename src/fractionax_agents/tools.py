from __future__ import annotations

from typing import Any

# Tool schemas exposed to Claude. Descriptions are prescriptive about WHEN to call
# the tool, which improves the model's tool-selection accuracy.
GET_QUOTE_TOOL: dict[str, Any] = {
    "name": "get_quote",
    "description": (
        "Get a fractional investment quote. Call this whenever the user asks how much "
        "an investment costs or wants a price for a given amount."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "integer",
                "description": "Investment amount in minor units (e.g. cents).",
            },
            "currency": {
                "type": "string",
                "description": "ISO 4217 currency code, e.g. USD.",
            },
        },
        "required": ["amount"],
    },
}

TOOLS: list[dict[str, Any]] = [GET_QUOTE_TOOL]


def get_quote(amount: int, currency: str = "USD") -> dict[str, Any]:
    """Compute a fractional investment quote in integer minor units."""
    if amount <= 0:
        raise ValueError("amount must be a positive integer in minor units")
    return {"amount": amount, "currency": currency.upper()}


def execute_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call from Claude to its implementation."""
    if name == "get_quote":
        return get_quote(**payload)
    raise ValueError(f"Unknown tool: {name}")
