"""Part 8: run the same 30 questions through both setups and compare.

    python -m part05_rag.ingest          # once, if you have not built the index yet
    python -m part08_multi_agent.compare
"""
import json
import statistics
import time
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

from helpdesk import hrms
from part08_multi_agent.agents import build_single, build_supervisor

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def is_correct(answer: str, expected: list[str]) -> bool:
    """Every expected item must appear. 'never|not' means either word is fine."""
    text = (answer.lower().replace(",", "")
            .replace("\u2019", "'").replace("\u2018", "'"))   # models often write curly apostrophes
    return all(any(alt.lower() in text for alt in item.split("|")) for item in expected)


def run(agent, questions):
    rows = []
    for q in questions:
        hrms.reset()
        start = time.perf_counter()
        with get_usage_metadata_callback() as usage:   # counts tokens in nested agents too
            out = agent.invoke({"messages": [{"role": "user", "content": q["question"]}]})
        tokens = sum(v.get("total_tokens", 0) for v in usage.usage_metadata.values())
        rows.append({"question": q["question"],
                     "correct": is_correct(out["messages"][-1].content, q["expected"]),
                     "seconds": time.perf_counter() - start, "tokens": tokens})
    return rows


def summary(name, rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    secs = statistics.median(r["seconds"] for r in rows)
    tok = sum(r["tokens"] for r in rows) / n
    return f"{name:<12} accuracy {acc:6.1%}   median latency {secs:5.1f}s   avg tokens {tok:8.0f}"


if __name__ == "__main__":
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    single, supervisor = build_single(), build_supervisor()
    for name, agent in [("single", single), ("supervisor", supervisor)]:
        rows = run(agent, questions)
        print(summary(name, rows))
        for r in rows:
            if not r["correct"]:
                print(f"    missed: {r['question']}")
