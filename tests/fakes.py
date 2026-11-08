"""A scripted chat model so the tests run offline, with no API key.

Each call to the model returns the next AIMessage from the script. That lets the
tests check our wiring (tools, state, middleware) without a real LLM.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def ai(content: str = "", *calls: tuple[str, dict]) -> AIMessage:
    """Build an AIMessage. ai("text") answers; ai("", ("tool", {...})) calls a tool."""
    return AIMessage(content=content, tool_calls=[
        {"name": n, "args": a, "id": f"call_{n}_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)],
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        response_metadata={"model_name": "scripted"})


class ScriptedModel(BaseChatModel):
    script: list[AIMessage]
    seen: list = []
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if self.i >= len(self.script):
            raise AssertionError("The scripted model ran out of replies")
        msg = self.script[self.i]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])
