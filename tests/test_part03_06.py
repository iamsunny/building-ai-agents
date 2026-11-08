import asyncio
import json

import pytest
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda

from helpdesk import hrms
from tests.fakes import ScriptedModel, ai


@pytest.fixture(autouse=True)
def fresh_hrms():
    hrms.reset()


def test_part03_bad_arguments_go_back_to_the_model():
    from part03_tools.tools import ALL_TOOLS
    model = ScriptedModel(script=[ai("", ("get_payslip", {"employee_id": "E1042", "month": "October"})), ai("Which month?")])
    out = create_agent(model, tools=ALL_TOOLS).invoke({"messages": [{"role": "user", "content": "my payslip"}]})
    tool_msg = [m for m in out["messages"] if m.type == "tool"][0]
    assert tool_msg.status == "error" and "should match pattern" in tool_msg.content


def test_part03_ticket_is_idempotent():
    from part03_tools.tools import raise_ticket
    args = {"employee_id": "E1042", "category": "payroll", "summary": "My October payslip shows the wrong HRA."}
    first, second = raise_ticket.invoke(args), raise_ticket.invoke(args)
    assert first["ticket_id"] == second["ticket_id"] and second.get("duplicate")
    assert len(hrms.list_tickets("E1042")) == 1


def test_part03_mcp_tools_load():
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from part03_tools.agent import MCP_SERVERS

    async def run():
        tools = {t.name: t for t in await MultiServerMCPClient(MCP_SERVERS).get_tools()}
        out = await tools["is_working_day"].ainvoke({"day": "2026-10-20"})
        return tools, out

    tools, out = asyncio.run(run())
    assert {"list_holidays", "is_working_day"} <= set(tools)
    assert '"working_day": false' in out[0]["text"]


def test_part04_threads_and_store():
    from langgraph.store.memory import InMemoryStore
    from part04_memory.agent import UserContext, build_agent
    model = ScriptedModel(script=[ai("", ("save_preference", {"key": "language", "value": "Hindi"})), ai("ok"),
                                  ai("", ("recall_preferences", {})), ai("namaste")])
    agent, me = build_agent(model, store=InMemoryStore()), UserContext("E1042")
    agent.invoke({"messages": [{"role": "user", "content": "Reply in Hindi"}]}, {"configurable": {"thread_id": "a"}}, context=me)
    out = agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, {"configurable": {"thread_id": "b"}}, context=me)
    assert len(out["messages"]) == 4  # thread b starts empty
    assert json.loads([m for m in out["messages"] if m.type == "tool"][0].content) == {"language": "Hindi"}


def test_part04_blocks_sensitive_preferences():
    from langgraph.store.memory import InMemoryStore
    from part04_memory.agent import UserContext, build_agent
    model = ScriptedModel(script=[ai("", ("save_preference", {"key": "health", "value": "medical condition"})), ai("ok")])
    out = build_agent(model, store=InMemoryStore()).invoke(
        {"messages": [{"role": "user", "content": "x"}]}, {"configurable": {"thread_id": "a"}}, context=UserContext("E1042"))
    assert "Not saved" in [m for m in out["messages"] if m.type == "tool"][0].content


def test_part04_only_allowed_keys_are_saved():
    from langgraph.store.memory import InMemoryStore
    from part04_memory.agent import UserContext, build_agent
    store = InMemoryStore()
    model = ScriptedModel(script=[ai("", ("save_preference", {"key": "tuesday_plans", "value": "away"})), ai("ok")])
    out = build_agent(model, store=store).invoke(
        {"messages": [{"role": "user", "content": "x"}]}, {"configurable": {"thread_id": "a"}}, context=UserContext("E1042"))
    assert "not a preference I can keep" in [m for m in out["messages"] if m.type == "tool"][0].content
    assert store.search(("preferences", "E1042")) == []


def test_part04_preferences_stay_with_their_owner():
    from langgraph.store.memory import InMemoryStore
    from part04_memory.agent import UserContext, build_agent
    store = InMemoryStore()
    model = ScriptedModel(script=[ai("", ("save_preference", {"key": "language", "value": "Hindi"})), ai("ok"),
                                  ai("", ("recall_preferences", {})), ai("nothing saved"),
                                  ai("", ("save_preference", {"key": "answer_style", "value": "x" * 61})), ai("ok")])
    agent = build_agent(model, store=store)
    run = lambda who, t: agent.invoke({"messages": [{"role": "user", "content": "x"}]},
                                      {"configurable": {"thread_id": t}}, context=UserContext(who))
    run("E1042", "asha")
    rohan = run("E1043", "rohan")
    assert json.loads([m for m in rohan["messages"] if m.type == "tool"][0].content) == {}
    too_long = run("E1042", "asha2")
    assert "Not saved: keep it under 60" in [m for m in too_long["messages"] if m.type == "tool"][0].content
    assert [i.key for i in store.search(("preferences", "E1042"))] == ["language"]


class FakeReaderModel:
    def __init__(self, request):
        self.request = request

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: self.request)


def test_part06_graph_books_leave():
    from part06_graphs.leave_workflow import LeaveRequest, build_graph
    req = LeaveRequest(intent="apply_leave", leave_type="earned", start="2026-11-16", end="2026-11-18")
    out = build_graph(FakeReaderModel(req)).invoke({"employee_id": "E1042", "message": "..."})
    assert out["result"]["days"] == 3 and hrms.get_leave_balance("E1042")["earned"] == 9


def test_part06_graph_stops_on_low_balance():
    from part06_graphs.leave_workflow import LeaveRequest, build_graph
    req = LeaveRequest(intent="apply_leave", leave_type="casual", start="2026-11-16", end="2026-11-20")
    out = build_graph(FakeReaderModel(req)).invoke({"employee_id": "E1042", "message": "..."})
    assert "have 3 left" in out["reply"] and "result" not in out


def test_part06_submit_handles_a_balance_that_changed():
    from part06_graphs.leave_workflow import LeaveRequest, build_graph
    req = LeaveRequest(intent="apply_leave", leave_type="casual", start="2026-11-16", end="2026-11-18")

    def approval_after_balance_changed(state):  # someone else used the casual leave meanwhile
        hrms.apply_leave("E1042", "casual", "2026-11-23", "2026-11-23")
        return {"problem": ""}

    graph = build_graph(FakeReaderModel(req), approval_node=approval_after_balance_changed)
    out = graph.invoke({"employee_id": "E1042", "message": "..."})
    assert out["reply"].startswith("Could not submit: Not enough casual leave") and "result" not in out


def test_part07_pause_survives_restart(tmp_path):
    import sqlite3
    from langgraph.types import Command
    from part06_graphs.leave_workflow import LeaveRequest
    from part07_human_in_loop.approval_graph import decide, get_graph
    req = LeaveRequest(intent="apply_leave", leave_type="earned", start="2026-11-16", end="2026-11-18")
    db, cfg = tmp_path / "cp.sqlite", {"configurable": {"thread_id": "t1"}}
    first = get_graph(sqlite3.connect(db, check_same_thread=False), FakeReaderModel(req))
    assert "__interrupt__" in first.invoke({"employee_id": "E1042", "message": "..."}, cfg)
    assert hrms.get_leave_balance("E1042")["earned"] == 12          # nothing submitted yet
    second = get_graph(sqlite3.connect(db, check_same_thread=False), FakeReaderModel(req))  # "restart"
    assert decide(second, "t1", "E2001", "approve").startswith("Done. Request")
    state = second.get_state(cfg).values
    assert state["result"]["days"] == 3 and state["request"]["decided_by"] == "E2001"


def test_part07_only_the_manager_can_decide(tmp_path):
    import sqlite3
    import pytest
    from part06_graphs.leave_workflow import LeaveRequest
    from part07_human_in_loop.approval_graph import ApprovalError, decide, get_graph
    req = LeaveRequest(intent="apply_leave", leave_type="earned", start="2026-11-16", end="2026-11-18")
    graph = get_graph(sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False), FakeReaderModel(req))
    graph.invoke({"employee_id": "E1042", "message": "..."}, {"configurable": {"thread_id": "t1"}})
    with pytest.raises(ApprovalError, match="E1043 cannot decide"):     # knowing the thread_id is not enough
        decide(graph, "t1", "E1043", "approve")
    with pytest.raises(ApprovalError, match="E1042 cannot decide"):     # nor approving your own leave
        decide(graph, "t1", "E1042", "approve")
    assert hrms.get_leave_balance("E1042")["earned"] == 12
    decide(graph, "t1", "E2001", "approve")
    with pytest.raises(ApprovalError, match="No decision is waiting"):  # a second approval finds nothing
        decide(graph, "t1", "E2001", "approve")
    with pytest.raises(ApprovalError, match="No decision is waiting"):  # nor does an unknown thread
        decide(graph, "no-such-thread", "E2001", "approve")
    assert hrms.get_leave_balance("E1042")["earned"] == 9


def test_part07_middleware_waits_for_approval():
    from langgraph.types import Command
    from part07_human_in_loop.approval_agent import build_agent
    model = ScriptedModel(script=[ai("", ("apply_leave", {"employee_id": "E1042", "leave_type": "earned",
                                                           "start": "2026-11-16", "end": "2026-11-18"})), ai("Not booked.")])
    agent, cfg = build_agent(model), {"configurable": {"thread_id": "d"}}
    out = agent.invoke({"messages": [{"role": "user", "content": "book it"}]}, cfg)
    assert out["__interrupt__"][0].value["action_requests"][0]["name"] == "apply_leave"
    agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "Team offsite"}]}), cfg)
    assert hrms.get_leave_balance("E1042")["earned"] == 12


def test_part08_supervisor_delegates_and_counts_tokens(monkeypatch):
    from part08_multi_agent import compare
    from part08_multi_agent.agents import build_supervisor
    model = ScriptedModel(script=[ai("", ("ask_payroll", {"question": "Net pay for E1042 in 2026-10?"})),
                                  ai("", ("get_payslip", {"employee_id": "E1042", "month": "2026-10"})),
                                  ai("Net pay was INR 120,734."),
                                  ai("Your October net pay was INR 120,734.")])
    rows = compare.run(build_supervisor(model), [{"question": "Oct net pay?", "expected": ["120734"]}])
    assert rows[0]["correct"] and rows[0]["tokens"] == 480  # 4 model calls, nested ones included


def test_part08_median_of_an_even_sample():
    from part08_multi_agent.compare import summary
    rows = [{"correct": True, "seconds": s, "tokens": 0} for s in (1.0, 2.0, 4.0, 9.0)]
    assert "median latency   3.0s" in summary("single", rows)


def test_part09_critical_case_fails_the_gate_on_its_own():
    import pytest
    from part09_evals.evaluate import load_cases
    from part09_evals.test_evals import test_critical_cases
    assert [c["id"] for c in load_cases() if c.get("critical")] == ["other-person"]
    ok = {"answer_ok": True, "tools_ok": True, "critical": False}
    results = [dict(ok, id=f"c{i}") for i in range(15)] + [{"id": "other-person", "answer_ok": True,
                                                              "tools_ok": False, "critical": True}]
    with pytest.raises(AssertionError, match="other-person"):   # 15 of 16 would still clear a 90% bar
        test_critical_cases(results)


def test_part08_questions_file_is_valid():
    from part08_multi_agent.compare import QUESTIONS, is_correct
    qs = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    assert len(qs) == 30 and all(q["expected"] for q in qs)
    assert is_correct("No, alcohol is never reimbursed.", ["never|not|no"])


def test_part09_checks():
    from langchain_core.messages import HumanMessage
    from part09_evals.evaluate import check_case, in_order, load_cases
    assert in_order(["get_payslip", "get_payslip"], ["get_payslip", "search_policy", "get_payslip"])
    assert not in_order(["search_policy", "get_leave_balance"], ["get_leave_balance", "search_policy"])
    case = {"id": "x", "tools": ["get_leave_balance"], "must_include": ["12"], "forbidden_tools": ["raise_ticket"]}
    msgs = [HumanMessage("q"), ai("", ("get_leave_balance", {"employee_id": "E1042"})), ai("You have 12 days.")]
    assert check_case(case, msgs) == {"id": "x", "answer_ok": True, "tools_ok": True, "calls": ["get_leave_balance"]}
    assert len(load_cases()) == 16


def test_part10_identity_comes_from_context(tmp_path, monkeypatch):
    from part10_security import secure_agent as sa
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "audit.jsonl")
    model = ScriptedModel(script=[ai("", ("my_payslip", {"month": "2026-10", "employee_id": "E1043"})),
                                  ai("", ("team_leave_balances", {})), ai("done")])
    out = sa.build_agent(model).invoke({"messages": [{"role": "user", "content": "Rohan's pay?"}]},
                                       context=sa.UserContext("E1042", "employee"))
    tool_out = [m.content for m in out["messages"] if m.type == "tool"]
    assert "120734" in tool_out[0] and "86996" not in tool_out[0]      # still Asha's payslip
    assert "Only managers" in tool_out[1]
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["employee_id"] == "E1042"
    assert [json.loads(line)["status"] for line in lines] == ["success", "refused"]


def test_part10_audit_reads_the_outcome_not_the_wording():
    from langchain_core.messages import ToolMessage
    from part10_security.secure_agent import AuditLog
    status = lambda content, **kw: AuditLog._status(ToolMessage(content=content, tool_call_id="1", **kw))
    assert status(json.dumps({"ok": False, "error": "Only managers can see this."})) == "refused"
    assert status(json.dumps({"ok": False, "error": "Sorry, that is for managers."})) == "refused"  # reworded
    assert status('The handbook says "ok": false is not a valid answer.') == "success"  # text, not an outcome
    assert status(json.dumps({"casual": 3})) == "success"
    assert status("Error: boom", status="error") == "error"


def test_part10_broken_audit_log_does_not_hide_the_tool_error(tmp_path, monkeypatch):
    import pytest
    from part10_security import secure_agent as sa
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "missing-dir" / "audit.jsonl")   # every write fails
    monkeypatch.setattr(sa.hrms, "get_leave_balance", lambda *a: (_ for _ in ()).throw(RuntimeError("HRMS down")))
    model = ScriptedModel(script=[ai("", ("my_leave_balance", {})), ai("done")])
    with pytest.raises(RuntimeError, match="HRMS down"):
        sa.build_agent(model).invoke({"messages": [{"role": "user", "content": "Leave?"}]},
                                     context=sa.UserContext("E1042", "employee"))


def test_part10_broken_audit_log_does_not_fail_a_finished_call(tmp_path, monkeypatch, caplog):
    from part10_security import secure_agent as sa
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "missing-dir" / "audit.jsonl")   # every write fails
    model = ScriptedModel(script=[ai("", ("my_leave_balance", {})), ai("You have 12 earned days.")])
    out = sa.build_agent(model).invoke({"messages": [{"role": "user", "content": "Leave?"}]},
                                       context=sa.UserContext("E1042", "employee"))
    assert '"earned": 12' in [m for m in out["messages"] if m.type == "tool"][0].content
    assert "audit write failed for my_leave_balance" in caplog.text


def test_part10_audit_records_a_tool_that_raises(tmp_path, monkeypatch):
    import pytest
    from part10_security import secure_agent as sa
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(sa.hrms, "get_leave_balance", lambda *a: (_ for _ in ()).throw(RuntimeError("HRMS down")))
    model = ScriptedModel(script=[ai("", ("my_leave_balance", {})), ai("done")])
    with pytest.raises(RuntimeError):
        sa.build_agent(model).invoke({"messages": [{"role": "user", "content": "Leave?"}]},
                                     context=sa.UserContext("E1042", "employee"))
    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert entry["tool"] == "my_leave_balance" and entry["status"] == "error"


def test_part10_manager_sees_only_direct_reports():
    from part10_security.secure_agent import UserContext, team_leave_balances

    class RT:
        context = UserContext("E2001", "manager")
    out = team_leave_balances.func(runtime=RT())
    assert set(out) == {"E1042", "E1043"}


def test_part10_pii_masked_before_model_sees_it(tmp_path, monkeypatch):
    from part10_security import secure_agent as sa
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "audit.jsonl")
    model = ScriptedModel(script=[ai("ok")])
    model.seen = []
    sa.build_agent(model).invoke({"messages": [{"role": "user", "content": "My PAN is ABCDE1234F, mail me at a@b.com"}]},
                                 context=sa.UserContext("E1042", "employee"))
    sent = " ".join(str(m.content) for m in model.seen[0])
    assert "ABCDE1234F" not in sent and "a@b.com" not in sent


def test_part11_streams_sse(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from part10_security import secure_agent as sa
    from part11_service import app as service
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "audit.jsonl")
    model = ScriptedModel(script=[ai("", ("my_leave_balance", {})), ai("You have 12 earned leave days.")])
    monkeypatch.setattr(service, "get_model", lambda **kw: model)
    with TestClient(service.app) as client:
        r = client.post("/chat", json={"thread_id": "t1", "message": "Leave?"},
                        headers={"X-Employee-Id": "E1042", "X-Role": "employee"})
    events = [block.split("\n")[0].removeprefix("event: ") for block in r.text.strip().split("\n\n")]
    assert events[0] == "tool_call" and "tool_result" in events and events[-1] == "done"
    assert '"input_tokens": 200' in r.text


def test_part11_cost_prices_cached_input_separately(monkeypatch):
    from part11_service import app as service
    monkeypatch.setattr(service, "PRICE_IN", 1.0)
    monkeypatch.setattr(service, "PRICE_CACHED", 0.1)
    monkeypatch.setattr(service, "PRICE_OUT", 4.0)
    usage = {"m": {"input_tokens": 1_000_000, "output_tokens": 0, "input_token_details": {"cache_read": 400_000}}}
    assert service.usage_report(usage)["cost_usd"] == 0.64   # 600k at 1.0 plus 400k at 0.1


def test_part11_unexpected_failure_still_ends_with_error(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from part10_security import secure_agent as sa
    from part11_service import app as service
    monkeypatch.setattr(sa, "AUDIT_LOG", tmp_path / "audit.jsonl")

    class BrokenModel(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("provider is down")

    monkeypatch.setattr(service, "get_model", lambda **kw: BrokenModel(script=[]))
    with TestClient(service.app) as client:
        r = client.post("/chat", json={"thread_id": "t1", "message": "Leave?"},
                        headers={"X-Employee-Id": "E1042", "X-Role": "employee"})
    events = [block.split("\n")[0].removeprefix("event: ") for block in r.text.strip().split("\n\n")]
    assert events[-1] == "error" and "provider is down" not in r.text
