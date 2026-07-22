---
name: ci-check
description: Mirror the three CI jobs locally (ruff check, pytest, import smoke) before pushing. Use before committing or opening a PR, or when the user asks to verify changes pass CI.
---

Run the same checks CI runs, in this order, and report the results. Use `uv run` for the Python tooling.

1. **Lint** — `uv run ruff check .`
   The whole repo must be clean under the configured select.

2. **Tests** — `uv run pytest -q`
   Matrix in CI is Python 3.10–3.13; locally, run the default interpreter.

3. **Import smoke** — `uv run python -c "import tradingagents, cli.main"`
   Catches undeclared runtime dependencies (CI does this against a clean `pip install .`).

Report a concise pass/fail summary for each step. If any step fails, show the relevant output and stop — do not proceed to commit. Do not attempt to auto-fix beyond obvious lint issues unless the user asks.
