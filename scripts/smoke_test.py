"""Run every part of the series against your real model and save what happened.

Usage (from the repo root, with .env filled in):
    python scripts/smoke_test.py            # everything (about 10 minutes with a small model)
    python scripts/smoke_test.py 1 5 10     # only some parts

Writes smoke_results.md (readable) and smoke_results.json next to this README.
Nothing from .env is written to either file.
"""
import json
import os
import platform
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TIMEOUT = 600

PART11 = r'''
from fastapi.testclient import TestClient
from part11_service.app import app
with TestClient(app) as client:
    r = client.post("/chat", headers={"X-Employee-Id": "E1042", "X-Role": "employee"},
                    json={"thread_id": "smoke", "message": "How many earned leave days do I have?"})
    print(r.status_code)
    print(r.text)
'''

STEPS = [
    ("1", "Part 1: raw loop", ["-m", "part01_raw_loop.agent", "How many casual days do I have, and are there holidays in October?"]),
    ("2", "Part 2: create_agent + structured output", ["-m", "part02_first_agent.agent", "My payslip looks wrong, who do I talk to?"]),
    ("3", "Part 3: tools + MCP", ["-m", "part03_tools.agent", "Is 2026-10-20 a working day? If not, how much earned leave do I have?"]),
    ("3b", "Part 3: validation error recovery", ["-m", "part03_tools.agent", "Show me my payslip for October"]),
    ("4", "Part 4: memory", ["-m", "part04_memory.agent"]),
    ("5i", "Part 5: build the index", ["-m", "part05_rag.ingest"]),
    ("5", "Part 5: RAG answer", ["-m", "part05_rag.agent", "Can I carry my unused earned leave into next year?"]),
    ("5b", "Part 5: RAG says I don't know", ["-m", "part05_rag.agent", "What is the policy on bringing pets to the office?"]),
    ("6", "Part 6: leave graph", ["-m", "part06_graphs.leave_workflow"]),
    ("6b", "Part 6: low balance", ["-m", "part06_graphs.leave_workflow", "Casual leave from 2026-11-16 to 2026-11-20"]),
    ("7", "Part 7: approval graph", None),
    ("7b", "Part 7: approval middleware", ["-m", "part07_human_in_loop.approval_agent"]),
    ("8", "Part 8: single vs supervisor", ["-m", "part08_multi_agent.compare"]),
    ("8b", "Part 8: deep agent", ["-m", "part08_multi_agent.deep_agent"]),
    ("9", "Part 9: evals", ["-m", "part09_evals.evaluate"]),
    ("10", "Part 10: secure agent", ["-m", "part10_security.secure_agent"]),
    ("11", "Part 11: streaming service", ["-c", PART11]),
]


def run(args, stdin=""):
    started = time.perf_counter()
    try:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # Windows consoles default to cp1252
        p = subprocess.run([PY, *args], cwd=ROOT, input=stdin, capture_output=True, text=True,
                           timeout=TIMEOUT, encoding="utf-8", errors="replace", env=env)
        ok, out, err = p.returncode == 0, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        ok, out, err = False, exc.stdout or "", f"Timed out after {TIMEOUT}s"
    return {"ok": ok, "seconds": round(time.perf_counter() - started, 1),
            "stdout": out[-6000:], "stderr": err[-4000:]}


def run_part7():
    start = run(["-m", "part07_human_in_loop.approval_graph", "start"])
    m = re.search(r"thread_id = (\S+)", start["stdout"])
    if not start["ok"] or not m:
        return start
    review = run(["-m", "part07_human_in_loop.approval_graph", "review", m.group(1)])
    approve = run(["-m", "part07_human_in_loop.approval_graph", "approve", m.group(1)])
    return {"ok": review["ok"] and approve["ok"],
            "seconds": round(start["seconds"] + review["seconds"] + approve["seconds"], 1),
            "stdout": "\n".join([start["stdout"], review["stdout"], approve["stdout"]]),
            "stderr": "\n".join(s for s in [start["stderr"], review["stderr"], approve["stderr"]] if s)}


def versions():
    names = ["langchain", "langchain-core", "langgraph", "langchain-openai", "langchain-mcp-adapters",
             "deepagents", "openai", "fastapi"]
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "missing"
    return out


def main():
    wanted = set(sys.argv[1:])
    env = {"python": platform.python_version(), "os": platform.platform(),
           "model": os.getenv("MODEL") or "(from .env)", "packages": versions()}
    results = []
    for key, name, args in STEPS:
        if wanted and key.rstrip("bi") not in wanted and key not in wanted:
            continue
        print(f"Running {name} ...", flush=True)
        if key == "7":
            r = run_part7()
        elif key == "7b":
            r = run(args, stdin="a\na\na\n")
        else:
            r = run(args)
        r.update(step=key, name=name)
        results.append(r)
        print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['seconds']}s", flush=True)

    (ROOT / "smoke_results.json").write_text(json.dumps({"env": env, "results": results}, indent=2), encoding="utf-8")
    lines = ["# Smoke test results", "", f"Python {env['python']} on {env['os']}", "",
             "| Step | Result | Seconds |", "| --- | --- | --- |"]
    lines += [f"| {r['name']} | {'OK' if r['ok'] else 'FAIL'} | {r['seconds']} |" for r in results]
    for r in results:
        lines += ["", f"## {r['name']}", "", "```text", r["stdout"].strip(), "```"]
        if r["stderr"].strip():
            lines += ["", "stderr:", "", "```text", r["stderr"].strip()[-2000:], "```"]
    (ROOT / "smoke_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed = [r["name"] for r in results if not r["ok"]]
    print("\nAll steps passed." if not failed else f"\nFailed: {', '.join(failed)}")
    print("Results saved to smoke_results.md")


if __name__ == "__main__":
    main()
