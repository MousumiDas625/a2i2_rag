#!/usr/bin/env python3
"""
server.py — FastAPI Server for Interactive / Programmatic Access
=================================================================

PURPOSE:
    Exposes HTTP endpoints so external clients (web UIs, notebooks, etc.)
    can interact with the evacuation dialogue system without running
    scripts directly.

ENDPOINTS:
    POST /chat          — Send a message and get the next reply.
    POST /simulate      — Run a full automated conversation and return results.
    GET  /health        — Health check.
    GET  /personas      — List available resident personas.

HOW IT WORKS:
    /chat accepts a JSON body with the conversation history, resident name,
    speaker role, and strategy.  It generates the next turn (operator or
    resident) and returns it.

    /simulate runs a full conversation via conversation_loop and returns
    the final result dict.

USAGE:
    uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload

    # Or via the convenience entry point:
    python api/server.py
"""

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import API_HOST, API_PORT

app = FastAPI(
    title="A2I2 Evacuation Dialogue API",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────
class Turn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    speaker: str  # "operator" or "resident"
    resident: str
    history: List[Turn]
    text: str  # latest utterance from the other side
    strategy: str = "iql_rag"
    temperature: float = 0.7
    max_tokens: int = 128


class ChatResponse(BaseModel):
    text: str
    selected_policy: Optional[str] = None


class SimulateRequest(BaseModel):
    resident: str = "ross"
    strategy: str = "iql_rag"
    seed_text: Optional[str] = None
    max_turns: int = 16


class SimulateResponse(BaseModel):
    status: str
    success: int
    turns: int
    history: List[dict]
    path: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/personas")
def list_personas():
    return {"personas": list(PERSONA.keys())}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [{"role": t.role, "text": t.text} for t in req.history]

    if req.speaker.lower() == "resident":
        from simulation.resident_simulator import generate_resident_reply
        reply = generate_resident_reply(
            history, req.resident,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return ChatResponse(text=reply)

    else:  # operator
        from simulation.operator_generator import generate_operator_reply
        policy_name = None
        rag_examples: list = []

        if req.strategy == "iql_rag":
            from retrieval.policy_selector import IQLPolicySelector
            from retrieval.rag_retrieval import retrieve_topk_pairs
            selector = IQLPolicySelector()
            policy_name, _qvals = selector.select_policy(history)
            last_res = next(
                (h["text"] for h in reversed(history) if h["role"] == "resident"), ""
            )
            rag_examples = retrieve_topk_pairs(policy_name, last_res, k=2)

        elif req.strategy == "rag_successful":
            from retrieval.rag_retrieval import retrieve_from_successful
            last_res = next(
                (h["text"] for h in reversed(history) if h["role"] == "resident"), ""
            )
            rag_examples = retrieve_from_successful(last_res, k=3)

        reply = generate_operator_reply(
            history,
            strategy=req.strategy,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            policy_name=policy_name,
            rag_examples=rag_examples,
        )
        return ChatResponse(text=reply, selected_policy=policy_name)


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    from simulation.conversation_loop import run_conversation
    result = run_conversation(
        resident_name=req.resident,
        strategy=req.strategy,
        seed_text=req.seed_text,
        max_turns=req.max_turns,
    )
    return SimulateResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=True)
