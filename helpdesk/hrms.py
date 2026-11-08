"""A fake HRMS (HR management system) that every part of the series uses.

All people, numbers and policies here are made up. The data lives in memory, so
every run starts from the same state. Functions return plain dicts so they are
easy to send to a model as JSON.
"""
from __future__ import annotations

import copy
import itertools
from datetime import date

_SEED = {
    "employees": {
        "E1042": {"name": "Asha Verma", "role": "employee", "manager_id": "E2001", "location": "Noida",
                  "email": "asha.verma@example.com"},
        "E1043": {"name": "Rohan Mehta", "role": "employee", "manager_id": "E2001", "location": "Pune",
                  "email": "rohan.mehta@example.com"},
        "E2001": {"name": "Neha Kapoor", "role": "manager", "manager_id": None, "location": "Noida",
                  "email": "neha.kapoor@example.com"},
    },
    "leave": {
        "E1042": {"earned": 12, "sick": 6, "casual": 3},
        "E1043": {"earned": 4, "sick": 7, "casual": 1},
        "E2001": {"earned": 18, "sick": 8, "casual": 4},
    },
    "payslips": {
        ("E1042", "2026-10"): {"gross": 142000, "basic": 56800, "hra": 28400, "special_allowance": 56800,
                               "pf": 6816, "professional_tax": 200, "income_tax": 14250, "net": 120734},
        ("E1042", "2026-09"): {"gross": 142000, "basic": 56800, "hra": 28400, "special_allowance": 56800,
                               "pf": 6816, "professional_tax": 200, "income_tax": 13900, "net": 121084},
        ("E1043", "2026-10"): {"gross": 98000, "basic": 39200, "hra": 19600, "special_allowance": 39200,
                               "pf": 4704, "professional_tax": 200, "income_tax": 6100, "net": 86996},
    },
    "holidays": [
        {"date": "2026-10-02", "name": "Gandhi Jayanti"},
        {"date": "2026-10-20", "name": "Dussehra"},
        {"date": "2026-11-09", "name": "Diwali"},
        {"date": "2026-11-24", "name": "Guru Nanak Jayanti"},
        {"date": "2026-12-25", "name": "Christmas"},
        {"date": "2027-01-26", "name": "Republic Day"},
    ],
}

_state: dict = {}
_ids = itertools.count(1)


def reset() -> None:
    """Put the fake HRMS back to its starting data (tests call this)."""
    global _ids
    _state.clear()
    _state.update(copy.deepcopy(_SEED))
    _state["leave_requests"] = {}
    _state["tickets"] = {}
    _state["idempotency"] = {}
    _ids = itertools.count(1)


reset()


class HRMSError(Exception):
    """Raised for requests the HRMS rejects, such as an unknown employee."""


def _employee(employee_id: str) -> dict:
    emp = _state["employees"].get(employee_id)
    if emp is None:
        raise HRMSError(f"No employee with id {employee_id!r}.")
    return emp


def get_employee(employee_id: str) -> dict:
    return {"employee_id": employee_id, **_employee(employee_id)}


def get_leave_balance(employee_id: str) -> dict:
    _employee(employee_id)
    return {"employee_id": employee_id, "unit": "days", **_state["leave"][employee_id]}


def get_payslip(employee_id: str, month: str) -> dict:
    """month is 'YYYY-MM'. Amounts are in INR."""
    _employee(employee_id)
    slip = _state["payslips"].get((employee_id, month))
    if slip is None:
        raise HRMSError(f"No payslip for {employee_id} in {month}.")
    return {"employee_id": employee_id, "month": month, "currency": "INR", **slip}


def get_holidays(month: int, year: int = 2026) -> list[dict]:
    if not 1 <= month <= 12:
        raise HRMSError("month must be between 1 and 12.")
    prefix = f"{year}-{month:02d}"
    return [h for h in _state["holidays"] if h["date"].startswith(prefix)]


def apply_leave(employee_id: str, leave_type: str, start: str, end: str, reason: str = "",
                request_id: str | None = None) -> dict:
    """Create a leave request. Dates are 'YYYY-MM-DD'. Repeating a request_id returns the first result."""
    if request_id and request_id in _state["idempotency"]:
        return _state["idempotency"][request_id]
    _employee(employee_id)
    balance = _state["leave"][employee_id]
    if leave_type not in balance:
        raise HRMSError(f"Unknown leave type {leave_type!r}. Use one of {sorted(balance)}.")
    d1, d2 = date.fromisoformat(start), date.fromisoformat(end)
    if d2 < d1:
        raise HRMSError("End date is before start date.")
    days = (d2 - d1).days + 1
    if days > balance[leave_type]:
        raise HRMSError(f"Not enough {leave_type} leave: asked for {days}, have {balance[leave_type]}.")
    balance[leave_type] -= days
    req = {"request_id": f"LR-{next(_ids):04d}", "employee_id": employee_id, "leave_type": leave_type,
           "start": start, "end": end, "days": days, "reason": reason, "status": "submitted"}
    _state["leave_requests"][req["request_id"]] = req
    if request_id:
        _state["idempotency"][request_id] = req
    return req


def raise_ticket(employee_id: str, category: str, summary: str, idempotency_key: str | None = None) -> dict:
    """Open an HR ticket. Repeating an idempotency_key returns the existing ticket instead of a duplicate."""
    if idempotency_key and idempotency_key in _state["idempotency"]:
        return {**_state["idempotency"][idempotency_key], "duplicate": True}
    _employee(employee_id)
    ticket = {"ticket_id": f"HR-{next(_ids):04d}", "employee_id": employee_id, "category": category,
              "summary": summary, "status": "open"}
    _state["tickets"][ticket["ticket_id"]] = ticket
    if idempotency_key:
        _state["idempotency"][idempotency_key] = ticket
    return ticket


def list_tickets(employee_id: str) -> list[dict]:
    return [t for t in _state["tickets"].values() if t["employee_id"] == employee_id]


def direct_reports(manager_id: str) -> list[str]:
    return [eid for eid, e in _state["employees"].items() if e["manager_id"] == manager_id]
