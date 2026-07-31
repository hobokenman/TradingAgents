from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)


@tool
def get_earnings_call_transcript(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve the most recent earnings call transcript for a given ticker symbol.
    Resolves the latest fiscal quarter reported on or before curr_date (no
    look-ahead) and returns the transcript with per-segment sentiment scores, so
    management's own commentary and guidance can be used to explain the reported
    financial numbers. Uses the configured earnings_transcripts vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted earnings call transcript report, or a message noting
        that no transcript is available.
    """
    return route_to_vendor("get_earnings_call_transcript", ticker, curr_date)


@tool
def get_sec_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    form_type: Annotated[
        str,
        "SEC form to retrieve: '10-K', '10-Q', 'latest' (most recent of either), "
        "or '8-K' (digest of recent material-event filings)",
    ] = "latest",
) -> str:
    """
    Retrieve recent SEC filings for a ticker.
    For a 10-K/10-Q (or 'latest'), resolves the newest filing filed on or before
    curr_date (no look-ahead) from SEC EDGAR and returns a bounded metadata and
    heading index. The complete primary document is cached locally; use
    search_sec_filing and read_sec_filing for bounded access to its contents.
    For '8-K', indexes the recent material-event filing digest.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
        form_type (str): '10-K', '10-Q', '8-K', or 'latest' (default)
    Returns:
        str: A bounded filing index, or a message noting that no filing is available.
    """
    return route_to_vendor("get_sec_filing", ticker, curr_date, form_type)


@tool
def search_sec_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    queries: Annotated[
        list[str],
        "One to twelve literal phrases to find in the complete cached filing",
    ],
    form_type: Annotated[
        str,
        "SEC form to search: '10-K', '10-Q', 'latest', or '8-K'",
    ] = "latest",
    max_matches: Annotated[int, "Maximum matches per query (1-3)"] = 2,
    context_chars: Annotated[int, "Context characters on each side of a match (0-800)"] = 500,
) -> str:
    """Search a complete SEC filing without returning the whole document.

    Results contain exact excerpts and character offsets. Use shorter alternate
    phrases when a query has no literal match, then call read_sec_filing around
    material matches for additional context.
    """
    return route_to_vendor(
        "search_sec_filing",
        ticker,
        curr_date,
        queries,
        form_type,
        max_matches,
        context_chars,
    )


@tool
def read_sec_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    offset: Annotated[int, "Zero-based character offset in the cached complete filing"] = 0,
    form_type: Annotated[
        str,
        "SEC form to read: '10-K', '10-Q', 'latest', or '8-K'",
    ] = "latest",
    max_chars: Annotated[int, "Maximum exact characters to return (up to 16000)"] = 12000,
) -> str:
    """Read one exact bounded page from a cached SEC filing.

    The response reports the next offset, allowing lossless pagination while
    keeping every individual tool result below provider output limits.
    """
    return route_to_vendor(
        "read_sec_filing",
        ticker,
        curr_date,
        offset,
        form_type,
        max_chars,
    )
