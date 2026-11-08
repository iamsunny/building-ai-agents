"""A tiny MCP server that exposes the company calendar.

Another team could own and deploy this. The agent does not import it; it talks to it
over the Model Context Protocol (stdio here, HTTP in production).

    python -m part03_tools.mcp_server
"""
from datetime import date

from mcp.server.fastmcp import FastMCP

from helpdesk import hrms

mcp = FastMCP("company-calendar")


@mcp.tool()
def list_holidays(month: int, year: int = 2026) -> list[dict]:
    """List company holidays in a month. month is 1 to 12."""
    return hrms.get_holidays(month, year)


@mcp.tool()
def is_working_day(day: str) -> dict:
    """Check whether a date (YYYY-MM-DD) is a working day: not a weekend and not a company holiday."""
    d = date.fromisoformat(day)
    holiday = next((h["name"] for h in hrms.get_holidays(d.month, d.year) if h["date"] == day), None)
    return {"date": day, "working_day": d.weekday() < 5 and holiday is None,
            "reason": holiday or ("weekend" if d.weekday() >= 5 else "")}


if __name__ == "__main__":
    mcp.run(transport="stdio")
