"""Part 6: when the loop isn't enough. A leave application as a LangGraph graph.

Fixed steps (validate dates, check balance, submit) are plain Python. The model is
used only where language is involved: reading the request, and answering other questions.

    python -m part06_graphs.leave_workflow "I want earned leave from 2026-11-16 to 2026-11-18 for a family trip"
"""
from __future__ import annotations

import sys
from datetime import date
from typing import Literal, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from helpdesk import hrms
from helpdesk.llm import get_model


class LeaveRequest(BaseModel):
    """What the employee is asking for, read from their message."""
    intent: Literal["apply_leave", "other"] = Field(description="apply_leave only if they want to book leave")
    leave_type: Optional[Literal["earned", "sick", "casual"]] = None
    start: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    reason: str = ""


class State(TypedDict, total=False):
    employee_id: str
    message: str
    request: dict          # LeaveRequest as a plain dict, so checkpoints stay simple JSON
    problem: str
    result: dict
    reply: str


def build_graph(model=None, helpdesk_agent=None, checkpointer=None, approval_node=None):
    model = model or get_model()
    reader = model.with_structured_output(LeaveRequest)

    def understand(state: State) -> State:
        prompt = ("Read this HR helpdesk message. Today is 2026-11-07. "
                  f"Extract a leave request if there is one.\n\nMessage: {state['message']}")
        return {"request": reader.invoke(prompt).model_dump()}

    def validate(state: State) -> State:
        r = state["request"]
        if not (r["leave_type"] and r["start"] and r["end"]):
            return {"problem": "I need the leave type and both dates (YYYY-MM-DD) to book this."}
        try:
            d1, d2 = date.fromisoformat(r["start"]), date.fromisoformat(r["end"])
        except ValueError:
            return {"problem": "Those dates don't look right. Please use YYYY-MM-DD."}
        if d2 < d1:
            return {"problem": "The end date is before the start date."}
        if d1 < date(2026, 11, 7) and r["leave_type"] != "sick":
            return {"problem": "Only sick leave can be applied for past dates."}
        return {"problem": ""}

    def check_balance(state: State) -> State:
        r = state["request"]
        have = hrms.get_leave_balance(state["employee_id"])[r["leave_type"]]
        days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days + 1
        if days > have:
            return {"problem": f"You asked for {days} days of {r['leave_type']} leave but have {have} left."}
        return {"problem": ""}

    def submit(state: State) -> State:
        r = state["request"]
        try:  # the HR system re-checks the balance; it may have changed since check_balance ran
            result = hrms.apply_leave(state["employee_id"], r["leave_type"], r["start"], r["end"], r["reason"],
                                      request_id=f"{state['employee_id']}:{r['leave_type']}:{r['start']}:{r['end']}")
        except hrms.HRMSError as exc:
            return {"reply": f"Could not submit: {exc}"}
        return {"result": result,
                "reply": f"Done. Request {result['request_id']} for {result['days']} day(s) of "
                         f"{r['leave_type']} leave is submitted."}

    def explain_problem(state: State) -> State:
        return {"reply": state["problem"]}

    def answer_question(state: State) -> State:
        if helpdesk_agent is None:
            return {"reply": "I can only book leave here. Try the helpdesk chat for other questions."}
        out = helpdesk_agent.invoke({"messages": [{"role": "user", "content": state["message"]}]})
        return {"reply": out["messages"][-1].content}

    def route_intent(state: State) -> str:
        return "validate" if state["request"]["intent"] == "apply_leave" else "answer_question"

    def ok_or_problem(next_step: str):
        return lambda state: "explain_problem" if state.get("problem") else next_step

    g = StateGraph(State)
    g.add_node("understand", understand)
    g.add_node("validate", validate)
    g.add_node("check_balance", check_balance)
    g.add_node("submit", submit)
    g.add_node("explain_problem", explain_problem)
    g.add_node("answer_question", answer_question)
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route_intent, ["validate", "answer_question"])
    g.add_conditional_edges("validate", ok_or_problem("check_balance"), ["check_balance", "explain_problem"])
    if approval_node:  # Part 7 plugs a human approval step in here
        g.add_node("approve", approval_node)
        g.add_conditional_edges("check_balance", ok_or_problem("approve"), ["approve", "explain_problem"])
        g.add_conditional_edges("approve", ok_or_problem("submit"), ["submit", "explain_problem"])
    else:
        g.add_conditional_edges("check_balance", ok_or_problem("submit"), ["submit", "explain_problem"])
    g.add_edge("submit", END)
    g.add_edge("explain_problem", END)
    g.add_edge("answer_question", END)
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from part03_tools.agent import SYSTEM_PROMPT
    from part03_tools.tools import ALL_TOOLS
    from langchain.agents import create_agent

    helpdesk = create_agent(get_model(), tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    graph = build_graph(helpdesk_agent=helpdesk)
    print(graph.get_graph().draw_mermaid())
    message = " ".join(sys.argv[1:]) or "I want earned leave from 2026-11-16 to 2026-11-18 for a family trip"
    print(graph.invoke({"employee_id": "E1042", "message": message})["reply"])
