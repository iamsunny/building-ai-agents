"""Part 8: a Deep Agent for a long task that needs planning and a scratchpad.

    python -m part08_multi_agent.deep_agent
"""
from deepagents import create_deep_agent

from helpdesk.llm import get_model
from part03_tools.tools import get_holidays
from part05_rag.agent import search_policy

TASK = ("Prepare a one-page policy summary for new joiners. Cover leave, work from home, "
        "holidays, payroll dates and expenses. Cite handbook sections in square brackets. "
        "Save it as /new_joiner_summary.md.")


def build_agent(model=None):
    return create_deep_agent(
        model=model or get_model(),
        tools=[search_policy, get_holidays],
        system_prompt=("You write clear HR documents for employees. Plan the document with "
                       "write_todos, research each topic with search_policy, then write the file."),
    )


if __name__ == "__main__":
    result = build_agent().invoke({"messages": [{"role": "user", "content": TASK}]})
    summary = result["files"]["/new_joiner_summary.md"]["content"]
    print(summary)
