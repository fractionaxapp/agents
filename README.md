# @fractionax/agents

FractionAX AI agents service — [FastAPI](https://fastapi.tiangolo.com) + the
[Anthropic Claude](https://platform.claude.com) Python SDK, managed with
[uv](https://docs.astral.sh/uv/).

> **This repo is a submodule** of the [`ai`](https://github.com/fractionaxapp/ai) umbrella
> repo, which is itself a submodule of the
> [`fractionaxapp`](https://github.com/fractionaxapp/fractionaxapp) meta-monorepo. It is
> developed from the meta-repo, where it is mounted at `ai/agents`.

## What it is

An agentic tool-use loop: Claude answers fractional-ownership questions and calls
the `get_quote` tool for pricing. Built on the Messages API with a manual agent
loop and adaptive thinking. The default model is **Claude Opus 4.8**
(`claude-opus-4-8`), configurable via `AGENT_MODEL`.

```
src/fractionax_agents/
  config.py   # settings (ANTHROPIC_API_KEY, model, port)
  tools.py    # tool schemas + implementations
  agent.py    # the agentic loop (Anthropic SDK, tool use)
  server.py   # FastAPI app: GET /health, POST /chat
```

## Develop (from the meta-repo root)

```bash
moon run agents:dev        # uvicorn (reload) — needs ANTHROPIC_API_KEY
moon run agents:test       # pytest
moon run agents:lint       # ruff check
moon run agents:format     # ruff format --check
moon run agents:typecheck  # mypy
```

`POST /chat` requires `ANTHROPIC_API_KEY`; without it the endpoint returns 503.
Tests do not call the live API.
