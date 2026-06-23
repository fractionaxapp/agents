# @fractionax/agents

FractionAX AI agents service — [FastAPI](https://fastapi.tiangolo.com) + the
[Anthropic Claude](https://platform.claude.com) Python SDK, managed with
[uv](https://docs.astral.sh/uv/).

> **This repo is a submodule** of the [`ai`](https://github.com/fractionaxapp/ai) umbrella
> repo, which is itself a submodule of the
> [`fractionaxapp`](https://github.com/fractionaxapp/fractionaxapp) meta-monorepo. It is
> developed from the meta-repo, where it is mounted at `ai/agents`.

## What it is

The agent tier behind the Copilot UX (natural language → deal discovery → memo).
Built on the Messages API with the Anthropic SDK; structured output is produced
via forced tool use and validated into the shared `fractionax_core` domain models.
Default model is **Claude Opus 4.8** (`claude-opus-4-8`), configurable via
`AGENT_MODEL`.

```
src/fractionax_agents/
  config.py      # settings (ANTHROPIC_API_KEY, model, port)
  tools.py       # legacy get_quote tool schema + impl
  agent.py       # the get_quote agentic loop (Anthropic SDK, tool use)
  structured.py  # structured output via forced tool use -> Pydantic
  deals.py       # Deal Sourcing Agent: seed catalogue + filter/sort
  memo.py        # Underwriting Agent: deal -> InvestmentMemo (structured)
  copilot.py     # User Copilot: NL -> intent -> deals -> memo orchestration
  server.py      # FastAPI app
```

### Endpoints

| Method | Path        | What it does |
| --- | --- | --- |
| `GET`  | `/health`   | Liveness |
| `GET`  | `/deals`    | Filtered deal catalogue (no LLM) — `jurisdiction`, `risk_tier`, `min_yield_pct` |
| `POST` | `/chat`     | The legacy `get_quote` tool-use loop |
| `POST` | `/copilot`  | NL message → `{ intent, deals, memo }` — the M1 Copilot flow |

The Copilot parses the message into a structured `InvestmentIntent`, sources
matching deals, and (for invest/discover intents) generates an `InvestmentMemo`
for the top match.

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
