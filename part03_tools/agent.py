"""Part 3: the helpdesk agent with real tools plus one tool loaded from an MCP server.

    python -m part03_tools.agent "Is 2026-10-20 a working day? If not, how much earned leave do I have?"
"""
import asyncio
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from helpdesk.llm import get_model
from part03_tools.tools import ALL_TOOLS

SYSTEM_PROMPT = ("You are an HR helpdesk assistant for employee E1042. Use tools for every fact. "
                 "If a tool returns ok=false, read the hint and either fix your call or ask the employee. "
                 "Raise a ticket only when the employee asks for a follow-up or you cannot answer.")

MCP_SERVERS = {
    "calendar": {"transport": "stdio", "command": sys.executable, "args": ["-m", "part03_tools.mcp_server"]},
}


async def build_agent(model=None):
    client = MultiServerMCPClient(MCP_SERVERS)
    mcp_tools = await client.get_tools()  # list_holidays, is_working_day
    return create_agent(model or get_model(), tools=[*ALL_TOOLS, *mcp_tools], system_prompt=SYSTEM_PROMPT)


async def main(question: str):
    agent = await build_agent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Is 2026-10-20 a working day?"))
