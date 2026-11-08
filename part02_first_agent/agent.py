"""Part 2: the same helpdesk agent, rebuilt with LangChain's create_agent.

    python -m part02_first_agent.agent "How many leave days do I have?"
"""
import sys
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from helpdesk import hrms
from helpdesk.llm import get_model

SYSTEM_PROMPT = ("You are an HR helpdesk assistant. The person chatting with you is employee E1042. "
                 "Use the tools for any fact about leave or holidays. Never guess numbers. "
                 "If the question needs a human in HR, say so.")


@tool
def get_leave_balance(employee_id: str) -> dict:
    """Get the remaining leave days for one employee, split by leave type."""
    return hrms.get_leave_balance(employee_id)


@tool
def get_holidays(month: int) -> list[dict]:
    """List company holidays in a month of 2026. month is 1 to 12."""
    return hrms.get_holidays(month)


# Only real tool names are allowed, so the app can match on them (the model once wrote "functions.get_leave_balance").
ToolName = Literal["get_leave_balance", "get_holidays"]


class HelpdeskAnswer(BaseModel):
    """The reply the helpdesk app shows to the employee."""
    answer: str = Field(description="A short, friendly reply for the employee")
    sources: list[ToolName] = Field(description="Names of the tools whose data the answer uses")
    needs_human: bool = Field(description="True if someone in HR should follow up")


def build_agent(model=None):
    return create_agent(
        model=model or get_model(),
        tools=[get_leave_balance, get_holidays],
        system_prompt=SYSTEM_PROMPT,
        response_format=HelpdeskAnswer,
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many leave days do I have?"
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        message.pretty_print()
    print("\nStructured:", result["structured_response"])
