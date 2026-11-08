"""Part 8: one good agent versus a supervisor with two specialists.

Both setups get exactly the same tools, so the comparison is fair.
"""
from langchain.agents import create_agent
from langchain.tools import tool

from helpdesk.llm import get_model
from part03_tools.tools import get_holidays, get_leave_balance, get_payslip, raise_ticket
from part05_rag.agent import search_policy

EMPLOYEE = "The person chatting with you is employee E1042. Today is 2026-11-07."

SINGLE_PROMPT = (f"You are an HR helpdesk assistant. {EMPLOYEE} Use tools for every fact. "
                 "Use search_policy for rules and the other tools for the employee's own data. "
                 "Cite policy sections in square brackets.")

PAYROLL_PROMPT = (f"You are a payroll specialist. {EMPLOYEE} Answer questions about salary, "
                  "deductions, tax and payslips using get_payslip. Give exact amounts in INR.")

POLICY_PROMPT = (f"You are an HR policy specialist. {EMPLOYEE} Answer from search_policy results only, "
                 "cite sections in square brackets, and use get_leave_balance or get_holidays when the "
                 "question needs the employee's own numbers.")

SUPERVISOR_PROMPT = (f"You coordinate an HR helpdesk. {EMPLOYEE} You cannot see HR data yourself. "
                     "Send payroll questions to ask_payroll and policy, leave or holiday questions to "
                     "ask_policy. If a question needs both, ask both and combine the answers. "
                     "Use raise_ticket only when the employee wants a follow-up.")


def build_single(model=None, system_prompt=SINGLE_PROMPT):
    return create_agent(model or get_model(), name="helpdesk",
                        tools=[search_policy, get_leave_balance, get_holidays, get_payslip, raise_ticket],
                        system_prompt=system_prompt)


def build_supervisor(model=None):
    model = model or get_model()
    payroll = create_agent(model, tools=[get_payslip], system_prompt=PAYROLL_PROMPT, name="payroll")
    policy = create_agent(model, tools=[search_policy, get_leave_balance, get_holidays],
                          system_prompt=POLICY_PROMPT, name="policy")

    @tool
    def ask_payroll(question: str) -> str:
        """Hand a payroll question (salary, deductions, tax, payslips) to the payroll specialist.
        Write the question so it makes sense on its own."""
        return payroll.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].content

    @tool
    def ask_policy(question: str) -> str:
        """Hand a policy, leave or holiday question to the policy specialist.
        Write the question so it makes sense on its own."""
        return policy.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].content

    return create_agent(model, tools=[ask_payroll, ask_policy, raise_ticket],
                        system_prompt=SUPERVISOR_PROMPT, name="supervisor")
