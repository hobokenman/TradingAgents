from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_earnings_call_transcript,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
    get_sec_filing,
    read_sec_filing,
    search_sec_filing,
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


# Standard operating procedure for interpreting a 10-K/10-Q through the bounded
# SEC index/search/read tools. Module scope, like ``EARNINGS_CALL_SOP`` above.
SEC_FILINGS_SOP = """When you analyze a SEC filing through `get_sec_filing`, `search_sec_filing`, and `read_sec_filing`, act as a senior hedge fund equity research analyst focused on fundamental investing. `get_sec_filing` returns a bounded metadata/heading index, not the filing body. Use `search_sec_filing` to locate disclosures in the complete cached text and `read_sec_filing` at returned offsets when more exact context is needed. Never claim the filing is unreviewable merely because it is large: these tools are the supported file-reading path. Determine what the filing implies for the company's earnings power, cash generation, balance-sheet strength, competitive position, valuation, and investment thesis. Do not merely summarize the filing — focus on information that could change investor expectations, financial estimates, or the stock's risk/reward. Work through all ten areas:

1. Executive investment conclusion — Is the filing incrementally positive, neutral, mixed, or negative? What are the three most important new disclosures? Should revenue, margin, EPS, or free-cash-flow expectations change? Did the investment thesis strengthen or weaken? What is the largest newly identified risk, the most important potential catalyst, and your confidence level? Do not call a result positive simply because revenue or EPS increased — judge it on quality, sustainability, and trend versus the company's own history.

2. What changed — Identify material changes versus the prior period in: revenue growth, segment performance, pricing/volume/customer/mix, gross and operating margins, operating expenses, cash flow, capital expenditure, working capital, debt and liquidity, share count, customer or supplier concentration, risk factors, legal proceedings, accounting policies, contractual obligations, and management language. Use what management itself flags as changed in the MD&A and risk factors, and reconcile with the multi-period data from `get_income_statement`, `get_balance_sheet`, and `get_cashflow` to catch changes management did NOT highlight. Rank the five most important changes by potential impact on earnings or valuation. Do not treat repeated boilerplate disclosures as new information.

3. Revenue and business quality — Explain how the company makes money: main products, customers, segments, and the key drivers of revenue and margins; whether the business is recurring, transactional, cyclical, or capital-intensive. Break revenue growth into available components (volume, pricing, mix, customer additions, geography, acquisitions, foreign exchange, new products). Determine whether growth is accelerating or slowing, organic or acquisition-driven, broad-based or concentrated, supported by backlog/bookings/billings/usage/cash collection, and sustainable or pulled forward. Highlight any conflict between reported revenue and operating indicators.

4. Segment analysis — For each major segment: revenue and growth, profit or margin, sequential and year-over-year trends, main operating drivers, key risks, and the investment implication. Identify which segment is most likely to drive future estimate revisions. Flag cases where strong consolidated results hide deterioration in an important business.

5. Margins and earnings quality — Explain why margins changed, separating pricing, volume/utilization, mix, input and labor costs, cost reductions, lower growth investment, foreign exchange, acquisitions, accounting changes, and one-time items. Classify margin movement as structural, cyclical, temporary, accounting-driven, or caused by underinvestment. Identify whether EPS was helped by core operating growth versus lower taxes, interest income, buybacks, investment gains, restructuring adjustments, stock-based-compensation exclusions, or other non-operating items. Rate earnings quality as high, average, or low.

6. Cash flow and working capital — Compare net income, operating cash flow, and free cash flow. Analyze receivables, inventory, payables, deferred revenue, customer deposits, capital expenditure, capitalized software or development costs, stock-based compensation, and restructuring payments. Flag receivables or inventory growing faster than revenue, earnings growth without cash-flow growth, temporary working-capital benefits, and aggressive capitalization of expenses. Give greater weight to cash flow than adjusted earnings when they conflict.

7. Balance sheet and hidden obligations — Review cash and investments, gross and net debt, interest expense, maturities, credit facilities and covenants, leases, goodwill and intangibles, restricted cash, pensions, deferred taxes, and liquidity/refinancing risk. Search specifically for guarantees, variable-interest entities, unconsolidated entities, equity-method investments, joint ventures, supplier financing, receivables factoring, securitizations, purchase commitments, minimum-volume agreements, letters of credit, residual-value guarantees, and litigation or regulatory exposure. For each material exposure state its nature, maximum potential exposure, amount recognized, trigger conditions, and potential impact on cash flow or valuation. Distinguish accounting classification from economic risk.

8. Footnotes and accounting judgment — Prioritize notes on revenue recognition, segments, debt, leases, commitments and contingencies, acquisitions, goodwill, stock compensation, taxes, related parties, investments, legal proceedings, and subsequent events. Identify areas requiring significant judgment (revenue recognition, reserves, credit losses, useful lives, capitalized expenses, impairments, fair-value estimates, tax valuation allowances, legal reserves). Highlight changes in assumptions, definitions, estimates, or non-GAAP adjustments that make current results look stronger or weaker.

9. Red flags and positive inflections — Red flags: weak cash conversion, aggressive revenue recognition, rising receivables or inventory, repeated restructuring, increasing stock-based compensation, acquisition-dependent growth, customer concentration, debt-funded buybacks, changing KPI definitions, discontinued disclosure of weak metrics, material weaknesses, auditor changes. Positive inflections: revenue acceleration, improving volume or retention, market-share gains, sustainable pricing, gross-margin expansion, operating leverage, better working-capital management, debt reduction, lower dilution, improved capital allocation. For each important item give the evidence, its severity or durability, and the financial implication.

10. Estimate revisions and thesis update — State whether expectations should show a material increase, modest increase, no meaningful change, modest decrease, or material decrease for: revenue, gross margin, operating margin, EPS, free cash flow, capital expenditure, net debt, and share count. Then state whether the filing strengthens, slightly strengthens, does not change, slightly weakens, or materially weakens the thesis. Provide the strongest evidence for the bull, base, and bear cases.

8-K event filings — When reviewing 8-Ks retrieved via form_type='8-K' (a digest of recent material-event filings, newest first), do not force the ten areas above onto each one. For each event: identify what happened and under which items it was reported (e.g. Item 1.01 material agreements, Item 2.02 results announcements, Item 4.02 non-reliance on previously issued financials, Item 5.02 officer or director changes, Item 8.01 other events); judge whether it changes earnings power, cash flow, balance-sheet risk, or governance; and classify it as routine housekeeping or thesis-relevant. Fold thesis-relevant events into the required output below.

Required output for the filing portion of your report: (1) executive conclusion; (2) five most important changes; (3) financial and segment scorecard; (4) revenue and margin analysis; (5) earnings and cash-flow quality; (6) balance-sheet and off-balance-sheet risks; (7) important footnote findings; (8) red flags and positive inflections; (9) estimate revision direction; (10) bull, base, and bear implications; (11) final investment view; (12) five items to monitor in the next filing.

Clearly separate reported facts, management statements, and analyst interpretation. Do not invent missing data — if a disclosure is absent from the filing, say so rather than guessing. Cite the filing section or exact wording for every material conclusion when available."""


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
            get_sec_filing,
            search_sec_filing,
            read_sec_filing,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, `get_earnings_call_transcript` for the most recent earnings call, and `get_sec_filing`, `search_sec_filing`, and `read_sec_filing` for bounded access to complete 10-K/10-Q filings and recent 8-K events from SEC EDGAR."
            + " Pull the latest earnings call transcript and reconcile management's commentary, guidance, and Q&A tone (the transcript includes per-segment sentiment scores; weigh them) with the reported statements; let this shape your overall assessment and directional conclusion. If no transcript is available, say so and proceed with the other data."
            + " For SEC review, first call `get_sec_filing` with form_type='latest'. Its first line identifies whether the resolved filing is a 10-Q or 10-K; do not request that same form again. Request form_type='10-K' only when the latest filing is a 10-Q and annual-level context is needed. Search the complete filing in focused batches with `search_sec_filing`, covering at minimum MD&A, revenue recognition, segments, legal proceedings and regulatory matters, commitments and contingencies, leases, related parties, goodwill and impairment, taxes and valuation allowances, debt and liquidity, and subsequent events. Use `read_sec_filing` at material search offsets for enough exact context to verify conclusions. Then inspect form_type='8-K' for events not reflected in the periodic report. Every SEC tool response is intentionally bounded; make additional search/read calls instead of treating a bounded response as a missing filing. Cross-check disclosures against the statements and earnings call. If EDGAR explicitly reports that no filing exists or retrieval failed, say so and proceed with the other data.\n\n"
            + EARNINGS_CALL_SOP
            + "\n\n"
            + SEC_FILINGS_SOP
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
