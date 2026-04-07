"""
server.py — FastAPI server exposing the OT Incident Defender environment.

Endpoints:
  POST /reset   — {"task_id": "task1", "seed": 42} -> Observation
  POST /step    — Action JSON -> StepResult
  GET  /state   — current internal state dict
  GET  /tasks   — list of TaskInfo
  GET  /health  — {"status": "ok"}

Run: uvicorn environment.server:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment.env import OTIncidentEnv
from environment.models import Action, Observation, StepResult, TaskInfo, ResetRequest
from environment.tasks import TASK_REGISTRY

app = FastAPI(
    title="OT Incident Defender",
    description=(
        "OpenEnv environment simulating real-world ICS/OT incident response. "
        "Three scenarios: Oldsmar 2021, Ukraine 2015 grid attack, Stuxnet FDI."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment instance (one session at a time)
_env = OTIncidentEnv()


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    """Reset the environment for a given task and seed. Returns initial observation."""
    if request.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {request.task_id!r}. Valid: {list(TASK_REGISTRY)}"
        )
    obs = _env.reset(task_id=request.task_id, seed=request.seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """Apply an action to the environment. Returns StepResult with obs, reward, done, info."""
    try:
        result = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def state() -> dict:
    """Return current internal state of the environment (including hidden plant state)."""
    return _env.get_state()


@app.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list[TaskInfo]:
    """Return list of all available tasks."""
    return list(TASK_REGISTRY.values())


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
