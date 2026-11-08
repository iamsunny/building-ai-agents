"""Part 7: human in the loop, agent version, with HumanInTheLoopMiddleware.

The agent may read freely, but every write tool waits for a person.

    python -m part07_human_in_loop.approval_agent
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from helpdesk import hrms
from helpdesk.llm import get_model
from part03_tools.tools import get_holidays, get_leave_balance, raise_ticket


@tool
def apply_leave(employee_id: str, leave_type: str, start: str, end: str, reason: str = "") -> dict:
    """Submit a leave request. Dates are YYYY-MM-DD. leave_type is earned, sick or casual."""
    try:
        return hrms.apply_leave(employee_id, leave_type, start, end, reason)
    except hrms.HRMSError as exc:
        return {"ok": False, "error": str(exc)}


def build_agent(model=None, checkpointer=None):
    return create_agent(
        model or get_model(),
        tools=[get_leave_balance, get_holidays, apply_leave, raise_ticket],
        system_prompt="You are an HR helpdesk assistant for employee E1042. Today is 2026-11-07.",
        checkpointer=checkpointer or InMemorySaver(),   # pausing needs a checkpointer
        middleware=[HumanInTheLoopMiddleware(interrupt_on={
            "apply_leave": True,                                          # approve, edit, reject or respond
            "raise_ticket": {"allowed_decisions": ["approve", "reject"]}, # no editing tickets
            "get_leave_balance": False,                                   # reads never pause
            "get_holidays": False,
        })],
    )


if __name__ == "__main__":
    agent = build_agent()
    config = {"configurable": {"thread_id": "demo-1"}}
    result = agent.invoke({"messages": [{"role": "user", "content":
                           "Book earned leave from 2026-11-16 to 2026-11-18 for a family trip."}]}, config)

    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        decisions = []
        for action in request["action_requests"]:
            print(f"\nThe agent wants to run {action['name']} with {action['args']}")
            answer = input("approve / reject? ").strip().lower()
            if answer.startswith("a"):
                decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "reject", "message": input("Why? ")})
        result = agent.invoke(Command(resume={"decisions": decisions}), config)

    print("\n" + result["messages"][-1].content)
