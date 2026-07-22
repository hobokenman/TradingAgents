from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_short_volume(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 30-day window"
    ] = None,
) -> str:
    """
    Retrieve daily short sale volume and the Short Volume Ratio (SVR) for a
    US-listed equity. SVR = short volume / total volume of trades reported to
    FINRA facilities (off-exchange/TRF), per trading day.
    Uses the configured short_sale_data vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 30-day window
    Returns:
        str: A markdown report with a per-day short volume table, the daily
        SVR, and summary statistics (latest SVR, period baseline, trend).
    """
    return route_to_vendor("get_short_volume", symbol, curr_date, look_back_days)


@tool
def get_short_interest(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 180-day (~12 cycle) window"
    ] = None,
) -> str:
    """
    Retrieve bi-monthly reported short interest for a US-listed equity:
    short positions per settlement cycle, cycle-over-cycle change, and
    days-to-cover. Only cycles already published as of curr_date are included.
    Uses the configured short_sale_data vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 180-day (~12 cycle) window
    Returns:
        str: A markdown report with a per-cycle short interest table and a
        latest-cycle summary (position, change %, days-to-cover, trend).
    """
    return route_to_vendor("get_short_interest", symbol, curr_date, look_back_days)
