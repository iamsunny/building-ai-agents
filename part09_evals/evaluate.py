"""Part 9: evaluate the helpdesk agent on fixed cases.

Three layers per case:
1. Answer checks: required and forbidden text in the final answer (cheap, exact).
2. Trajectory checks: the tools the agent called, in order (catches lucky guesses).
3. Judge: a second model grades the answer against a rubric (catches tone and reasoning).

    python -m part05_rag.ingest     # once
    python -m part09_evals.evaluate
"""
import json
from pathlib import Path

from pydantic import BaseModel, Field

from helpdesk import hrms
from helpdesk.llm import get_model
from part08_multi_agent.agents import SINGLE_PROMPT, build_single

CASES = Path(__file__).with_name("cases.jsonl")


def load_cases():
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def normalize(answer: str) -> str:
    """Lower-case, drop thousands separators, and turn curly quotes into straight ones,
    so "INR 1,20,734" matches "120734" and "don’t know" matches "don't know"."""
    return (answer.lower().replace(",", "")
            .replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"'))


def _has(text: str, item: str) -> bool:
    return any(alt.lower() in text for alt in item.split("|"))


def tool_calls(messages) -> list[str]:
    return [c["name"] for m in messages if m.type == "ai" for c in m.tool_calls]


def in_order(expected: list[str], actual: list[str]) -> bool:
    """True if the expected tools appear in this order (other calls may sit between them)."""
    it = iter(actual)
    return all(name in it for name in expected)


class Grade(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = fully meets the rubric, 1 = fails it")
    reason: str = Field(description="One sentence")


JUDGE_PROMPT = """You grade answers from an HR helpdesk assistant.
Rubric: {rubric}
Employee question: {question}
Assistant answer: {answer}
Score 1 to 5 against the rubric only. Do not reward length or politeness."""


def check_case(case, messages, judge=None) -> dict:
    answer = messages[-1].content
    text = normalize(answer)
    calls = tool_calls(messages)
    must, must_not = case.get("must_include", []), case.get("must_not_include", [])
    forbidden = case.get("forbidden_tools", [])
    result = {"id": case["id"], "calls": calls,
        "answer_ok": all(_has(text, s) for s in must)
                     and not any(_has(text, s) for s in must_not),
        "tools_ok": in_order(case.get("tools", []), calls)
                    and not any(t in calls for t in forbidden)}
    if judge is not None:
        grade = judge.invoke(JUDGE_PROMPT.format(
            rubric=case["rubric"], question=case["question"], answer=answer))
        result.update(score=grade.score, reason=grade.reason)
    return result


# The first run of the critical gate failed: the agent opened a ticket about Rohan's pay and guessed
# a payslip month. These rules fix that. They extend the Part 8 prompt, so Part 8's comparison stays as it ran.
RULES = (" Raise a ticket only when the employee asks for HR to follow up."
         " If asked about another employee's data, say you can't share it and do nothing else."
         " If the employee does not say which month, ask before calling get_payslip.")


def run_evals(agent=None, judge_model=None, use_judge=True):
    agent = agent or build_single(system_prompt=SINGLE_PROMPT + RULES)
    judge = (judge_model or get_model(temperature=0)).with_structured_output(Grade) if use_judge else None
    results = []
    for case in load_cases():
        hrms.reset()
        out = agent.invoke({"messages": [{"role": "user", "content": case["question"]}]})
        result = check_case(case, out["messages"], judge)
        result["answer"] = out["messages"][-1].content   # kept so failures can be read in full
        result["critical"] = case.get("critical", False)
        results.append(result)
    return results


def report(results) -> dict:
    n = len(results)
    stats = {"cases": n,
             "answer_pass": sum(r["answer_ok"] for r in results) / n,
             "tools_pass": sum(r["tools_ok"] for r in results) / n}
    scored = [r["score"] for r in results if "score" in r]
    if scored:
        stats["judge_avg"] = sum(scored) / len(scored)
    return stats


if __name__ == "__main__":
    results = run_evals()
    for r in results:
        flag = "PASS" if r["answer_ok"] and r["tools_ok"] else "FAIL"
        print(f"{flag}  {r['id']:<16} tools={r['calls']}  judge={r.get('score', '-')}  {r.get('reason', '')}")
        if flag == "FAIL":
            print(f"      answer: {r['answer'][:300]}")
    print(json.dumps(report(results), indent=2))
