"""Part 7: human in the loop, graph version.

The leave workflow from Part 6 now pauses before submitting and waits for the manager.
Checkpoints go to SQLite, so the pause survives a restart of this process.

    python -m part07_human_in_loop.approval_graph start "Earned leave 2026-11-16 to 2026-11-18, family trip"
    python -m part07_human_in_loop.approval_graph review  <thread_id>
    python -m part07_human_in_loop.approval_graph approve <thread_id> --as E2001
    python -m part07_human_in_loop.approval_graph reject  <thread_id> "Team offsite that week" --as E2001

--as stands in for the manager's verified login. In a real app it comes from the session, never the request.
"""
import argparse
import sqlite3
import sys
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

from helpdesk import hrms
from part06_graphs.leave_workflow import build_graph


class ApprovalError(Exception):
    pass


def manager_approval(state):
    r = state["request"]
    decision = interrupt({                       # the graph stops here and saves its state
        "employee_id": state["employee_id"],
        "leave_type": r["leave_type"], "start": r["start"], "end": r["end"], "reason": r["reason"],
        "question": "Approve this leave request?",
    })
    request = {**r, "decided_by": decision["approver_id"]}   # who decided is saved with the request
    if decision["action"] == "approve":         # runs again, from the top, after resume
        return {"problem": "", "request": request}
    return {"problem": "Your manager declined: " + (decision.get("comment") or "no reason given"),
            "request": request}


def decide(graph, thread_id: str, approver_id: str, action: str, comment: str = "") -> str:
    """Resume a paused request. The thread_id only finds it; the HR data decides who may answer."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.next:                          # unknown thread, or already decided
        raise ApprovalError("No decision is waiting on this request.")
    employee = hrms.get_employee(state.values["employee_id"])
    if approver_id != employee["manager_id"]:
        raise ApprovalError(f"{approver_id} cannot decide on {employee['name']}'s leave.")
    resume = {"action": action, "comment": comment, "approver_id": approver_id}
    return graph.invoke(Command(resume=resume), config)["reply"]


def get_graph(conn, model=None):
    return build_graph(model=model, checkpointer=SqliteSaver(conn), approval_node=manager_approval)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Leave approval demo")
    p.add_argument("command", choices=["start", "review", "approve", "reject"])
    p.add_argument("target", nargs="?", help="the leave message for start, otherwise the thread_id")
    p.add_argument("comment", nargs="?", default="", help="the reason, for reject")
    p.add_argument("--as", dest="approver", help="who is deciding (stands in for a verified login)")
    a = p.parse_args()

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    graph = get_graph(conn)

    if a.command == "start":
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        message = a.target or "Earned leave 2026-11-16 to 2026-11-18, family trip"
        graph.invoke({"employee_id": "E1042", "message": message}, config)
        print(f"Paused for approval. thread_id = {thread_id}")
    elif a.command == "review":
        for task in graph.get_state({"configurable": {"thread_id": a.target}}).tasks:
            for pending in task.interrupts:
                print(pending.value)
    else:
        if not a.approver:
            sys.exit("Say who is deciding: --as <employee_id>")
        try:
            print(decide(graph, a.target, a.approver, a.command, a.comment))
        except ApprovalError as exc:
            sys.exit(f"Not allowed: {exc}")
