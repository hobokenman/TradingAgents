# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent LLM trading-research framework. Specialized agents (analysts, bull/bear researchers, trader, risk team, portfolio manager) collaborate via LangGraph to produce a trading decision for a ticker + date. Entry point: `TradingAgentsGraph` in `tradingagents/graph/trading_graph.py`; `ta.propagate(ticker, date)` returns `(state, decision)`.

## Commands

Use **uv** for local development (a `uv.lock` is the source of truth; README/CI/Docker document pip, but prefer uv here).

- Install: `uv sync --extra dev`
- Test: `uv run pytest -q` (config in `pyproject.toml`; `--strict-markers` is on — markers: `unit`, `integration`, `smoke`)
- Single test: `uv run pytest tests/path_test.py::test_name`
- Lint: `uv run ruff check .` (CI expects the whole repo clean under the configured select)
- Format: run `uv run ruff format <file>` on files you create or edit. Do **not** reformat the whole repo — repo-wide `ruff format` adoption is intentionally deferred to avoid mass merge conflicts.

Ruff config: line-length 100, target py310, select `E,W,F,I,B,UP,C4,SIM`, `E501` ignored.

## CI (`.github/workflows/ci.yml`)

Three jobs must pass: `pytest -q` (Python 3.10–3.13 matrix), `ruff check .`, and a clean-install import smoke (`python -c "import tradingagents, cli.main"`) that catches undeclared runtime deps. Run `/ci-check` to mirror all three locally before pushing.

## Conventions

- Commit messages: **Conventional Commits** — `type(scope): summary` (e.g. `fix(agents): ...`, `docs: ...`).
- Python `>=3.10`.

## Gotchas

- **Config**: all defaults live in `tradingagents/default_config.py`, applied at import. Override at runtime with `TRADINGAGENTS_*` env vars (coerced to the default's type; invalid values fail loudly at startup). Setting a provider/model/backend/language var also skips the matching interactive CLI prompt (enables unattended runs).
- **Secrets**: copy `.env.example` → `.env` (loaded via python-dotenv). Azure uses a separate `.env.enterprise`.
- **Persistent state** under `~/.tradingagents/`: decision log at `memory/trading_memory.md`, cache at `cache/`, logs at `logs/`, checkpoint SQLite DBs at `cache/checkpoints/<TICKER>.db`. Checkpoint resume is opt-in (`--checkpoint`); `--clear-checkpoints` resets.
- **Tests** run without real keys/network: autouse fixtures in `tests/conftest.py` stub API keys with `"placeholder"` and reset the global dataflows config between tests. `test.py` and `scripts/smoke_structured_output.py` are NOT part of the pytest suite.
- **Data vendors** are explicit chains (configured per-category `data_vendors` / per-tool `tool_vendors` in `default_config.py`), not silent fallbacks.
- LLM output is non-deterministic and reasoning models ignore `temperature` — not a defect.

## Layout

- `tradingagents/agents/` — the agent roles (`analysts/`, `researchers/`, `managers/`, `risk_mgmt/`, `trader/`)
- `tradingagents/dataflows/` — data vendor clients + `interface.py`, `config.py`
- `tradingagents/graph/` — LangGraph orchestration
- `tradingagents/llm_clients/` — provider clients, `factory.py`, `model_catalog.py`
- `cli/` — Typer + Rich CLI (`tradingagents analyze`)
