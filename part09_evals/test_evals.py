"""Part 9: the regression gate. CI runs this on every change to prompts, tools or models.

    pytest part09_evals -m evals
"""
import os

import pytest

from part09_evals.evaluate import report, run_evals

# Thresholds. Raise them as the agent improves; never lower them to make a build pass.
MIN_ANSWER_PASS = 0.90
MIN_TOOLS_PASS = 0.90
MIN_JUDGE_AVG = 4.0

pytestmark = [
    pytest.mark.evals,
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs a model API key"),
]


@pytest.fixture(scope="module")
def results():
    return run_evals()


def test_answers(results):
    assert report(results)["answer_pass"] >= MIN_ANSWER_PASS, [r["id"] for r in results if not r["answer_ok"]]


def test_tool_use(results):
    assert report(results)["tools_pass"] >= MIN_TOOLS_PASS, [(r["id"], r["calls"]) for r in results if not r["tools_ok"]]


def test_critical_cases(results):
    failed = [r["id"] for r in results if r["critical"] and not (r["answer_ok"] and r["tools_ok"])]
    assert not failed, f"critical cases must always pass: {failed}"


def test_judge(results):
    assert report(results)["judge_avg"] >= MIN_JUDGE_AVG, [(r["id"], r["reason"]) for r in results if r["score"] < 4]
