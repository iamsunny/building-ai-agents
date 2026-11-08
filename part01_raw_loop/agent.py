"""Part 1: an agent is just a loop.

No framework here, only the OpenAI Python SDK. Run it from the repo root:

    python -m part01_raw_loop.agent "How many leave days do I have?"
"""
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from helpdesk import hrms

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from the environment
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

SYSTEM = ("You are an HR helpdesk assistant. The person chatting with you is employee E1042. "
          "Use the tools for any fact about leave or holidays. Never guess numbers.")

# 1. Describe the tools in JSON Schema so the model knows what it may call.
TOOLS = [
    {"type": "function", "function": {
        "name": "get_leave_balance",
        "description": "Get the remaining leave days for one employee, split by leave type.",
        "parameters": {"type": "object",
                       "properties": {"employee_id": {"type": "string", "description": "For example E1042"}},
                       "required": ["employee_id"]}}},
    {"type": "function", "function": {
        "name": "get_holidays",
        "description": "List company holidays in a month of 2026.",
        "parameters": {"type": "object",
                       "properties": {"month": {"type": "integer", "description": "1 to 12"}},
                       "required": ["month"]}}},
]

# 2. Map each tool name to the real Python function that does the work.
FUNCTIONS = {"get_leave_balance": hrms.get_leave_balance, "get_holidays": hrms.get_holidays}


def run_tool(name: str, args: dict) -> str:
    try:
        return json.dumps(FUNCTIONS[name](**args))
    except Exception as exc:  # send the error back so the model can recover
        return json.dumps({"error": str(exc)})


# 3. The loop: call the model, run what it asks for, feed results back, repeat.
def run_agent(question: str, max_turns: int = 5) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        reply = response.choices[0].message
        messages.append(reply.model_dump(exclude_none=True))
        if not reply.tool_calls:  # no tool requested: this is the final answer
            print(f"turn {turn}: final answer")
            return reply.content
        print(f"turn {turn}: model asked for {len(reply.tool_calls)} tool(s)")
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"  {call.function.name}({args})")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": run_tool(call.function.name, args)})
    return "Sorry, I could not finish within the turn limit."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many leave days do I have?"
    print(run_agent(question))
