"""Buffett Researcher: a value-investing read on the company, ahead of the debate.

Runs once between the analyst team and the Bull/Bear debate, so both debaters
and the Research Manager argue against a concrete value thesis rather than
producing one. Unlike the debaters it takes no turn and casts no vote in the
debate loop; it contributes a standing assessment.

Grounding is a distilled principles document, not retrieval: the shareholder
letters are condensed offline into ``buffett_principles.md`` (see
``scripts/distill_buffett_principles.py``) and injected whole. That keeps the
runtime free of an embedding store, which this codebase deliberately does not
have, and keeps the agent's frame of reference identical across runs.
"""

from __future__ import annotations

from pathlib import Path

from tradingagents.agents.schemas import (
    BuffettAssessment,
    BuffettVerdict,
    render_buffett_assessment,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)

# Prefix on the line appended to the shared debate history. Deliberately not
# "Bull"/"Bear": the debate router in graph/conditional_logic.py switches
# speakers on those prefixes.
SPEAKER_PREFIX = "Buffett Researcher"

_BASELINE_PATH = Path(__file__).with_name("buffett_principles.md")


def principles_override_path() -> Path:
    """Return the user-generated principles path, which wins over the baseline.

    Config is read lazily so tests can redirect it, matching the dataflows
    caches. ``scripts/distill_buffett_principles.py`` writes here.
    """
    from tradingagents.dataflows.config import get_config

    configured = get_config().get("buffett_principles_path")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".tradingagents" / "buffett" / "principles.md"


def load_buffett_principles() -> str:
    """Return the principles document, preferring a user-distilled override.

    Read per run rather than cached: it happens once per graph invocation, so
    a regenerated document takes effect without restarting a long-lived
    process.
    """
    override = principles_override_path()
    source = override if override.is_file() else _BASELINE_PATH
    return source.read_text(encoding="utf-8").strip()


def _out_of_circle_report(asset_type: str, target: str) -> str:
    """Return a principled abstention for assets the value frame cannot price.

    Rendered locally rather than asked of the model: the answer is entailed by
    the framework itself, so spending a deep-model call to rediscover it would
    be waste.
    """
    return render_buffett_assessment(
        BuffettAssessment(
            verdict=BuffettVerdict.TOO_HARD,
            moat=(
                f"{target} is {asset_type}, not an operating business. There is no "
                "franchise to assess, no customers, and no competitive position that "
                "could widen or narrow."
            ),
            management=(
                "There is no management team allocating capital on an owner's behalf, "
                "so the candor and capital-allocation tests do not apply."
            ),
            valuation=(
                "The asset produces nothing. Intrinsic value is the discounted cash a "
                "business can be expected to hand its owners over its life, and an "
                "asset with no cash flows offers nothing to discount. Its worth today "
                "is whatever the next buyer will pay, which is a forecast about other "
                "people rather than about a business."
            ),
            thesis=(
                "This sits squarely in the too-hard pile, and passing costs nothing. "
                "A productive asset can be valued because it produces something; a "
                "non-productive one can only be priced by guessing at the next "
                "participant's mood. That is a game worth declining rather than one "
                "worth playing badly.\n\n"
                "Treat this as an abstention, not a bearish call. The value framework "
                "has no opinion here, and the decision should rest on the other "
                "analysis in this run."
            ),
        )
    )


def create_buffett_researcher(llm):
    structured_llm = bind_structured(llm, BuffettAssessment, "Buffett Researcher")

    def buffett_node(state) -> dict:
        asset_type = state.get("asset_type", "stock")
        instrument_context = get_instrument_context_from_state(state)

        if asset_type != "stock":
            report = _out_of_circle_report(asset_type, str(state["company_of_interest"]))
        else:
            prompt = f"""You are a value investor evaluating a business the way Warren Buffett does in his Berkshire Hathaway shareholder letters. You are not a market forecaster and you are not here to argue a side. Judge the business, the people running it, and the price, in that order.

Work strictly from the principles below. They are distilled from the letters and are your entire frame of reference.

---

{load_buffett_principles()}

---

{instrument_context}

---

**Analyst reports on this company:**

Market research report: {state["market_report"]}

Social media sentiment report: {state["sentiment_report"]}

Latest world affairs news: {state["news_report"]}

Company fundamentals report: {state["fundamentals_report"]}

---

Work through the evaluation checklist against this company. A few things matter more than thoroughness:

- Judge the ten-year picture, not the next quarter. Market sentiment and recent price action are close to irrelevant except where an unpopular price creates an opportunity.
- Say plainly when the reports do not support a judgment. An unsupported estimate is worse than an acknowledged gap.
- If the business genuinely cannot be evaluated within this framework, return the Too Hard verdict. Passing is a legitimate answer and carries no penalty; manufacturing a view to seem useful does.
- Separate business quality from price. A wonderful business at too high a price is not a buy, and saying so is the point of the exercise.

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

            report = invoke_structured_or_freetext(
                structured_llm,
                llm,
                prompt,
                render_buffett_assessment,
                "Buffett Researcher",
            )

        # Seeds the shared debate history so Bull and Bear engage with the
        # thesis. `count` and `current_response` are deliberately left alone:
        # the first drives the debate's termination check, and the second is
        # read by the Bull prompt as the last bear argument.
        debate_state = state["investment_debate_state"]
        history = debate_state.get("history", "")
        entry = f"{SPEAKER_PREFIX}: {report}"

        return {
            "buffett_report": report,
            "investment_debate_state": {
                **debate_state,
                "history": f"{history}\n{entry}".strip(),
            },
        }

    return buffett_node
