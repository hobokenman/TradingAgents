from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_earnings_call_transcript,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)

# Standard operating procedure for interpreting an earnings-call transcript.
# Kept at module scope (like ``NO_EXTERNAL_TOOLS`` in agents/utils/structured.py)
# so the long guidance stays out of the node body. Framed as instructions to the
# analyst — the transcript itself arrives from ``get_earnings_call_transcript``.
EARNINGS_CALL_SOP = """When you analyze the earnings-call transcript retrieved via `get_earnings_call_transcript`, act as a senior hedge fund equity analyst. Do not produce a generic summary — focus on information that could change earnings estimates, valuation, or the investment thesis.

Address the following in your report:
- Was the call positive, neutral, mixed, or negative?
- What were the five most important new disclosures?
- Is revenue growth accelerating or slowing?
- What is driving growth: price, volume, customers, products, geography, or acquisitions?
- Are margins improving or weakening, and why?
- Is earnings quality strong or weak?
- Does cash flow support reported earnings?
- Was guidance raised, lowered, or maintained?
- Is guidance conservative, realistic, aggressive, or back-end loaded?
- Which analyst questions were most important?
- Which management answers were direct, vague, evasive, or contradictory?
- Did management's tone or wording change from previous quarters?
- Is management credible?
- Should revenue, margin, EPS, or free-cash-flow estimates rise or fall?
- What evidence supports the bull case?
- What evidence supports the bear case?
- Did the investment thesis strengthen or weaken?
- What are the main catalysts and risks?
- What are the three most important things to monitor next quarter?

Clearly separate reported facts, management claims, and your own interpretation. Do not invent missing figures. Quote management only when the exact wording is important."""


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_earnings_call_transcript,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, and `get_earnings_call_transcript` for the most recent earnings call."
            + " Pull the latest earnings call transcript and reconcile management's commentary, guidance, and Q&A tone (the transcript includes per-segment sentiment scores; weigh them) with the reported statements; let this shape your overall assessment and directional conclusion. If no transcript is available, say so and proceed with the other data.\n\n"
            + EARNINGS_CALL_SOP
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
