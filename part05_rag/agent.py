"""Part 5: RAG as a tool. The agent decides when to search the handbook.

    python -m part05_rag.ingest
    python -m part05_rag.agent "Can I carry my unused earned leave into next year?"
"""
import sys

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from helpdesk.llm import get_model
from part03_tools.tools import get_leave_balance, get_payslip
from part05_rag.ingest import INDEX, get_embeddings

SYSTEM_PROMPT = (
    "You are an HR helpdesk assistant for employee E1042.\n"
    "- For policy questions, call search_policy and answer only from what it returns.\n"
    "- Cite the section in square brackets, for example [1.1 Earned leave].\n"
    "- If the search results do not answer the question, say you don't know and offer to raise a ticket.\n"
    "- For the employee's own numbers (leave balance, payslip), use those tools instead."
)

_store = None


def get_store():
    global _store
    if _store is None:
        _store = InMemoryVectorStore.load(str(INDEX), get_embeddings())
    return _store


MIN_SCORE = 0.30  # below this, a chunk is probably not about the question


@tool
def search_policy(query: str) -> dict:
    """Search the employee handbook. Use for any question about rules or policy:
    leave rules, work from home, holidays policy, payroll dates, expenses, notice period."""
    hits = get_store().similarity_search_with_score(query, k=4)
    results = [{"section": doc.metadata["source"], "text": doc.page_content, "score": round(score, 3)}
               for doc, score in hits if score >= MIN_SCORE]
    if not results:
        return {"results": [], "note": "Nothing relevant in the handbook."}
    return {"results": results}


def build_agent(model=None):
    return create_agent(model or get_model(), tools=[search_policy, get_leave_balance, get_payslip],
                        system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I carry my unused earned leave into next year?"
    result = build_agent().invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)
