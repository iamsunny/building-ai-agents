"""Part 3: tools that don't break your agent.

Six rules, all visible in this file:
1. Say when: the name and docstring are a prompt, so say when to use the tool, not only what it does.
2. Validate arguments with a schema, so bad input fails before it reaches the HRMS.
3. Return errors the model can act on instead of raising stack traces.
4. Make writes safe to retry (idempotent), so a repeated call cannot create a duplicate.
5. Return only the fields the model needs.
6. Give each tool one job.
"""
from __future__ import annotations

import hashlib
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from helpdesk import hrms


def _error(message: str, hint: str) -> dict:
    """A tool error the model can read and recover from."""
    return {"ok": False, "error": message, "hint": hint}


class EmployeeArgs(BaseModel):
    employee_id: str = Field(pattern=r"^E\d{4}$", description="Employee ID like E1042")


@tool(args_schema=EmployeeArgs)
def get_leave_balance(employee_id: str) -> dict:
    """Use this when the employee asks how much leave they have left.
    Returns remaining days per leave type (earned, sick, casual)."""
    try:
        return {"ok": True, **hrms.get_leave_balance(employee_id)}
    except hrms.HRMSError as exc:
        return _error(str(exc), "Check the employee ID. Do not guess another one.")


class PayslipArgs(BaseModel):
    employee_id: str = Field(pattern=r"^E\d{4}$", description="Employee ID like E1042")
    month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="Payslip month as YYYY-MM, e.g. 2026-10")


@tool(args_schema=PayslipArgs)
def get_payslip(employee_id: str, month: str) -> dict:
    """Use this for questions about salary, deductions, tax or net pay for a specific month.
    Amounts are in INR."""
    try:
        slip = hrms.get_payslip(employee_id, month)
    except hrms.HRMSError as exc:
        return _error(str(exc), "Ask the employee which month they mean, as YYYY-MM.")
    keep = ["month", "currency", "gross", "pf", "professional_tax", "income_tax", "net"]
    return {"ok": True, **{k: slip[k] for k in keep}}


class HolidayArgs(BaseModel):
    month: int = Field(ge=1, le=12, description="Month number, 1 to 12")
    year: int = Field(default=2026, ge=2026, le=2027)


@tool(args_schema=HolidayArgs)
def get_holidays(month: int, year: int = 2026) -> dict:
    """Use this when the employee asks about company holidays in a month."""
    return {"ok": True, "holidays": hrms.get_holidays(month, year)}


class TicketArgs(BaseModel):
    employee_id: str = Field(pattern=r"^E\d{4}$")
    category: Literal["payroll", "leave", "policy", "other"]
    summary: str = Field(min_length=10, max_length=300, description="One or two sentences in the employee's words")


@tool(args_schema=TicketArgs)
def raise_ticket(employee_id: str, category: str, summary: str) -> dict:
    """Use this only when the employee asks for HR to follow up, or when you cannot answer
    with the other tools. Opens a ticket and returns its ID. A retry with the same arguments returns the same ticket."""
    # The same employee + category + summary always maps to the same key, so a retry
    # (by the model, the framework or the network) returns the first ticket.
    key = hashlib.sha256(f"{employee_id}|{category}|{summary.strip().lower()}".encode()).hexdigest()[:16]
    ticket = hrms.raise_ticket(employee_id, category, summary, idempotency_key=key)
    return {"ok": True, **ticket}


ALL_TOOLS = [get_leave_balance, get_payslip, get_holidays, raise_ticket]
