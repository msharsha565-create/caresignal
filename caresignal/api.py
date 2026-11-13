"""FastAPI backend: REST endpoints for the UI plus the agent chat endpoint."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentUnavailable, CareSignalAgent
from .model import ARTIFACT_DIR
from .service import NotFound, get_service

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()  # fail fast if artifacts are missing
    async with CareSignalAgent() as agent:
        app.state.agent = agent
        yield


app = FastAPI(title="CareSignal", version="1.0.0", lifespan=lifespan)

if origins := os.environ.get("CARESIGNAL_CORS_ORIGINS"):  # e.g. http://localhost:5173 in dev
    app.add_middleware(CORSMiddleware, allow_origins=origins.split(","), allow_methods=["*"], allow_headers=["*"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=60)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "tools": app.state.agent.tool_names}


@app.get("/api/metrics")
def metrics() -> dict:
    return json.loads((ARTIFACT_DIR / "metrics.json").read_text())


@app.get("/api/patients")
def patients(
    risk_band: Literal["high", "medium", "low"] | None = None,
    condition: Literal["chf", "copd", "diabetes", "ckd"] | None = None,
    min_age: int | None = Query(None, ge=0, le=120),
    limit: int = Query(50, ge=1, le=200),
) -> list[dict]:
    return get_service().list_patients(risk_band, condition, min_age, limit)


@app.get("/api/patients/{patient_id}")
def patient(patient_id: str) -> dict:
    svc = get_service()
    try:
        return {
            "patient": svc.get_patient(patient_id),
            "risk": svc.readmission_risk(patient_id),
            "interactions": svc.patient_interactions(patient_id)["interactions"],
        }
    except NotFound as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/cohort/summary")
def cohort(
    group_by: Literal["risk_band", "discharge_disposition", "sex", "chf", "copd", "diabetes", "ckd", "lives_alone"]
    | None = None,
) -> dict:
    return get_service().cohort_summary(group_by)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    if req.messages[-1].role != "user":
        raise HTTPException(422, "The last message must be from the user.")
    try:
        reply = await app.state.agent.chat([m.model_dump() for m in req.messages])
    except AgentUnavailable as e:
        raise HTTPException(503, str(e)) from e
    return {"reply": reply.text, "tool_calls": reply.tool_calls, "model": reply.model, "refused": reply.refused}


if WEB_DIST.exists():  # production: serve the built UI from the same container
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
