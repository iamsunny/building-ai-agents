# Building AI Agents

Companion code for the **[Building AI Agents](https://devsdaily.com/an-agent-is-just-a-loop/)** series on [devsdaily.com](https://devsdaily.com). One HR helpdesk agent, built up across 11 articles, from a raw tool-calling loop to a secured, evaluated, streaming service.

Everything runs against a fake HR system (`helpdesk/hrms.py`) and a fictional handbook (`data/handbook.md`). No real company data is involved. The fake HR system starts fresh on every run, so leave you book in one run is back in the balance the next time.

Three fictional people appear throughout:

| ID | Name | Role |
| --- | --- | --- |
| E1042 | Asha Verma | Employee, and the user in most examples |
| E1043 | Rohan Mehta | Employee, Asha's colleague |
| E2001 | Neha Kapoor | Manager of both |

## Setup

Python 3.12 or newer, and an OpenAI API key to run the parts (or another provider's, see [Switching model provider](#switching-model-provider)). The tests don't need a key.

On macOS or Linux:

```bash
git clone https://github.com/iamsunny/building-ai-agents.git && cd building-ai-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                              # offline tests, no API key needed
cp .env.example .env                                   # then add your OPENAI_API_KEY to run the parts
```

On Windows (PowerShell):

```powershell
git clone https://github.com/iamsunny/building-ai-agents.git; cd building-ai-agents
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
Copy-Item .env.example .env                            # then add your OPENAI_API_KEY to run the parts
```

In Git Bash on Windows, activate with `source .venv/Scripts/activate` instead.

`requirements.txt` sets supported version ranges, so pip installs the newest releases within them. To reproduce the article runs exactly, install the versions they used: `pip install -r requirements.txt -c constraints.txt`. CI installs without the constraints on purpose, so it tests the newest supported releases and catches breaking changes early.

Run every command from the repo root, as a module (`python -m ...`), with the virtual environment active (your prompt starts with `(.venv)`). If `pytest` is not found or an import fails, an inactive environment is the usual cause.

## What's where

- `helpdesk/hrms.py`: the fake HR system every part calls (leave, payslips, holidays, tickets).
- `helpdesk/llm.py`: `get_model()`, the one place that reads `MODEL` from `.env`.
- `data/handbook.md`: the policy handbook that Part 5 indexes.
- `partNN_*/`: one folder per article, each building on the earlier ones.
- `tests/`: offline tests with a scripted fake model.
- `scripts/smoke_test.py`: runs every part against your real model.

## The parts

| Part | Article | Folder | Run it |
| --- | --- | --- | --- |
| 1 | [An agent is just a loop](https://devsdaily.com/an-agent-is-just-a-loop/) | `part01_raw_loop` | `python -m part01_raw_loop.agent "How many leave days do I have?"` |
| 2 | [Your first LangChain agent](https://devsdaily.com/your-first-langchain-agent/) | `part02_first_agent` | `python -m part02_first_agent.agent` |
| 3 | [Tools that don't break your agent](https://devsdaily.com/tools-that-dont-break-your-agent/) | `part03_tools` | `python -m part03_tools.agent "Is 2026-10-20 a working day?"` |
| 4 | [Give your agent a memory](https://devsdaily.com/give-your-agent-a-memory/) | `part04_memory` | `python -m part04_memory.agent` |
| 5 | [RAG as a tool, not a pipeline](https://devsdaily.com/rag-as-a-tool/) | `part05_rag` | `python -m part05_rag.ingest` then `python -m part05_rag.agent` |
| 6 | [When the loop isn't enough: graphs](https://devsdaily.com/when-the-loop-isnt-enough-graphs/) | `part06_graphs` | `python -m part06_graphs.leave_workflow` |
| 7 | [Human in the loop](https://devsdaily.com/human-in-the-loop/) | `part07_human_in_loop` | `python -m part07_human_in_loop.approval_graph start`, then `approve <thread_id> --as E2001` |
| 8 | [Many agents, or one good one?](https://devsdaily.com/many-agents-or-one-good-one/) | `part08_multi_agent` | `python -m part08_multi_agent.compare` |
| 9 | [Testing an agent you can't predict](https://devsdaily.com/testing-an-agent-you-cant-predict/) | `part09_evals` | `python -m part09_evals.evaluate` |
| 10 | [Securing an agent](https://devsdaily.com/securing-an-agent/) | `part10_security` | `python -m part10_security.secure_agent` |
| 11 | [Shipping your agent](https://devsdaily.com/shipping-your-agent/) | `part11_service` | `uvicorn part11_service.app:app --port 8000` |

Parts 5, 8, 9, 10 and 11 search the handbook, so run `python -m part05_rag.ingest` once before them.

## Run everything at once

```bash
python scripts/smoke_test.py          # every part against your real model, about 10 minutes
python scripts/smoke_test.py 5 9      # only some parts
```

It writes `smoke_results.md` with each part's output. Nothing from `.env` is written to it.

## Tests

`tests/` runs offline with a scripted fake model, so it checks the wiring (tools, state, middleware, streaming) without spending tokens. `part09_evals/` holds the real evaluations, which call your model: `pytest part09_evals -m evals`.

The GitHub Actions workflow runs the offline tests on every push. The eval job runs only when the repository has an `OPENAI_API_KEY` secret.

## Switching model provider

Set `MODEL` in `.env`, for example `MODEL=anthropic:<model-name>`, and install the matching package (`pip install langchain-anthropic`). Part 1 deliberately uses the OpenAI SDK directly; any provider with an OpenAI-compatible endpoint works there through `base_url`.

## Troubleshooting

**`Activate.ps1 cannot be loaded because running scripts is disabled`.** PowerShell blocks scripts by default on some machines. Allow them for the current window only with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

**Part 7 says "No decision is waiting on this request".** The request was already approved or rejected, or the `thread_id` was mistyped. Checkpoints live in `checkpoints.sqlite` in the repo root; delete it to start over.

**`UnicodeEncodeError` on Windows when piping output.** The model sometimes answers with characters such as `₹`. Set `PYTHONUTF8=1` before running.
