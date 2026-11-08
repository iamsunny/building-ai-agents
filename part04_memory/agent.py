"""Part 4: memory.

Short-term memory: a checkpointer saves every step of a conversation under a thread_id.
Long-term memory: a store keeps small facts about the employee across conversations.
Summarisation: old messages get folded into a summary so long threads stay affordable.

    python -m part04_memory.agent
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from helpdesk import hrms
from helpdesk.llm import get_model

SYSTEM_PROMPT = ("You are an HR helpdesk assistant. Use tools for facts. "
                 "At the start of a conversation call recall_preferences. "
                 "When the employee states a lasting preference (answer style, language, how to address "
                 "them, preferred contact time), save it with save_preference. Never save health details, "
                 "salary figures, or anything the employee asks you to forget.")


@dataclass
class UserContext:
    employee_id: str  # set by your app after login, never by the model


@tool
def get_leave_balance(runtime: ToolRuntime[UserContext]) -> dict:
    """Get the current employee's remaining leave days by type."""
    return hrms.get_leave_balance(runtime.context.employee_id)


BLOCKED_WORDS = ("salary", "medical", "illness", "diagnosis", "pregnan")
ALLOWED_KEYS = ("answer_style", "language", "preferred_name", "contact_time")
MAX_VALUE_LENGTH = 60   # a preference is a few words, never a story


@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save one lasting preference for this employee. key is one of answer_style, language,
    preferred_name or contact_time, e.g. key='answer_style', value='one short line'."""
    if any(w in f"{key} {value}".lower() for w in BLOCKED_WORDS):
        return "Not saved: this looks like sensitive information."
    if key not in ALLOWED_KEYS:
        return f"Not saved: {key!r} is not a preference I can keep. Allowed: {', '.join(ALLOWED_KEYS)}."
    if len(value) > MAX_VALUE_LENGTH:
        return f"Not saved: keep it under {MAX_VALUE_LENGTH} characters."
    runtime.store.put(("preferences", runtime.context.employee_id), key, {"value": value})
    return f"Saved {key}."


@tool
def recall_preferences(runtime: ToolRuntime[UserContext]) -> dict:
    """Return every saved preference for this employee."""
    items = runtime.store.search(("preferences", runtime.context.employee_id))
    return {item.key: item.value["value"] for item in items}


@tool
def forget_preference(key: str, runtime: ToolRuntime[UserContext]) -> str:
    """Delete one saved preference when the employee asks you to forget it."""
    if key not in ALLOWED_KEYS:
        return f"Nothing to forget: {key!r} is not a preference I keep."
    runtime.store.delete(("preferences", runtime.context.employee_id), key)
    return f"Forgot {key}."


def build_agent(model=None, summary_model=None, checkpointer=None, store=None):
    model = model or get_model()
    return create_agent(
        model,
        tools=[get_leave_balance, save_preference, recall_preferences, forget_preference],
        system_prompt=SYSTEM_PROMPT,
        context_schema=UserContext,
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
        middleware=[SummarizationMiddleware(
            model=summary_model or model,
            trigger=("messages", 40),   # summarise once a thread passes 40 messages
            keep=("messages", 12),      # keep the latest 12 messages word for word
        )],
    )


if __name__ == "__main__":
    agent = build_agent()
    me = UserContext(employee_id="E1042")

    def ask(text, thread):
        config = {"configurable": {"thread_id": thread}}
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=me)
        print(f"[{thread}] you: {text}\n[{thread}] bot: {result['messages'][-1].content}\n")

    ask("Hi! Please keep your answers to one short line from now on.", "chat-1")
    ask("How many earned leave days do I have?", "chat-1")   # same thread: remembers the chat
    ask("And casual leave?", "chat-1")                        # follow-up works because of the checkpointer
    ask("How many sick days do I have?", "chat-2")           # new thread: no chat history, but the store
                                                              # still knows the answer-style preference
