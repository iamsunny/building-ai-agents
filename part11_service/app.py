"""Part 11: the helpdesk agent as a streaming HTTP service.

    uvicorn part11_service.app:app --port 8000
    curl -N -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -H "X-Employee-Id: E1042" -H "X-Role: employee" \
         -d '{"thread_id": "t1", "message": "How many earned leave days do I have?"}'

Auth here is a demo header. In production, take employee_id and role from a verified
session or token (for example your SSO's ID token), never from a header the client sets.
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from helpdesk.llm import get_model
from part10_security.secure_agent import UserContext, build_agent

PRICE_IN = float(os.getenv("PRICE_INPUT_PER_1M", "0"))    # your provider's price per 1M input tokens
PRICE_OUT = float(os.getenv("PRICE_OUTPUT_PER_1M", "0"))  # and per 1M output tokens
PRICE_CACHED = float(os.getenv("PRICE_CACHED_INPUT_PER_1M") or PRICE_IN)  # cached input; unset means full price
REQUEST_TIMEOUT_S = 90

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = get_model(timeout=30, max_retries=2)   # per-call timeout and retries
    state["agent"] = build_agent(model, checkpointer=InMemorySaver())  # Postgres saver in production
    yield


app = FastAPI(title="HR helpdesk agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def usage_report(usage: dict) -> dict:
    tokens_in = sum(u.get("input_tokens", 0) for u in usage.values())       # includes cached tokens
    cached = sum(u.get("input_token_details", {}).get("cache_read", 0) for u in usage.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in usage.values())
    cost = ((tokens_in - cached) * PRICE_IN + cached * PRICE_CACHED + tokens_out * PRICE_OUT) / 1e6
    return {"input_tokens": tokens_in, "output_tokens": tokens_out, "cost_usd": round(cost, 6)}


async def stream_answer(req: ChatRequest, ctx: UserContext):
    usage = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": f"{ctx.employee_id}:{req.thread_id}"}, "callbacks": [usage]}
    inputs = {"messages": [{"role": "user", "content": req.message}]}
    started = time.perf_counter()
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            async for mode, chunk in state["agent"].astream(inputs, config, context=ctx,
                                                            stream_mode=["messages", "updates"]):
                if mode == "messages":
                    token, meta = chunk
                    if meta.get("langgraph_node") == "model" and token.text and not getattr(token, "tool_call_chunks", None):
                        yield sse("token", {"text": token.text})
                elif mode == "updates":
                    for node, update in chunk.items():
                        for msg in (update or {}).get("messages", []):
                            if msg.type == "ai" and msg.tool_calls:
                                yield sse("tool_call", {"tools": [c["name"] for c in msg.tool_calls]})
                            elif msg.type == "tool":
                                yield sse("tool_result", {"tool": msg.name, "status": msg.status})
    except TimeoutError:
        yield sse("error", {"message": "The assistant took too long. Please try again."})
        return
    except Exception:  # model, tool or middleware failure: log the details, show the user a safe message
        logging.getLogger(__name__).exception("chat request failed")
        yield sse("error", {"message": "Something went wrong. Please try again."})
        return
    report = usage_report(usage.usage_metadata)
    report["seconds"] = round(time.perf_counter() - started, 2)
    yield sse("done", report)


@app.post("/chat")
async def chat(req: ChatRequest, x_employee_id: str = Header(...), x_role: str = Header("employee")):
    if x_role not in ("employee", "manager"):
        raise HTTPException(400, "Unknown role")
    ctx = UserContext(employee_id=x_employee_id, role=x_role)
    return StreamingResponse(stream_answer(req, ctx), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}
