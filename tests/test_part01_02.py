import json
from types import SimpleNamespace

import pytest

from helpdesk import hrms
from tests.fakes import ScriptedModel, ai


@pytest.fixture(autouse=True)
def fresh_hrms():
    hrms.reset()


def _chat_reply(content=None, calls=()):
    tool_calls = [SimpleNamespace(id=f"c{i}", type="function",
                                  function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                  for i, (n, a) in enumerate(calls)] or None

    class Msg(SimpleNamespace):
        def model_dump(self, exclude_none=True):
            d = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                d["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                   for c in self.tool_calls]
            return {k: v for k, v in d.items() if v is not None}

    return SimpleNamespace(choices=[SimpleNamespace(message=Msg(content=content, tool_calls=tool_calls))])


def test_part01_loop_runs_tool_then_answers(monkeypatch):
    from part01_raw_loop import agent

    replies = iter([_chat_reply(calls=[("get_leave_balance", {"employee_id": "E1042"})]),
                    _chat_reply(content="You have 12 days of earned leave left.")])
    sent = []

    def fake_create(**kw):
        sent.append([dict(m) for m in kw["messages"]])
        return next(replies)

    monkeypatch.setattr(agent.client.chat.completions, "create", fake_create)
    assert agent.run_agent("How many leave days do I have?") == "You have 12 days of earned leave left."
    tool_msg = sent[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["earned"] == 12


def test_part01_tool_error_goes_back_to_model():
    from part01_raw_loop import agent
    assert "error" in json.loads(agent.run_tool("get_holidays", {"month": 13}))


def test_part02_create_agent_with_structured_output():
    from part02_first_agent.agent import build_agent

    model = ScriptedModel(script=[
        ai("", ("get_leave_balance", {"employee_id": "E1042"})),
        ai("", ("HelpdeskAnswer", {"answer": "You have 12 days of earned leave.",
                                   "sources": ["get_leave_balance"], "needs_human": False})),
    ])
    result = build_agent(model).invoke({"messages": [{"role": "user", "content": "How many leave days do I have?"}]})
    assert result["structured_response"].answer.startswith("You have 12")
    tool_results = [m for m in result["messages"] if m.type == "tool" and m.name == "get_leave_balance"]
    assert json.loads(tool_results[0].content)["earned"] == 12
