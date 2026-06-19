import pytest

from fractionax_agents.tools import execute_tool, get_quote


def test_get_quote_normalizes_currency() -> None:
    assert get_quote(10_000, "usd") == {"amount": 10_000, "currency": "USD"}


def test_get_quote_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        get_quote(0)


def test_execute_tool_dispatches() -> None:
    assert execute_tool("get_quote", {"amount": 500})["amount"] == 500


def test_execute_tool_unknown() -> None:
    with pytest.raises(ValueError):
        execute_tool("nope", {})
