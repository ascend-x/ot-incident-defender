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

_env = OTIncidentEnv()


@app.get("/")
def read_root():
    return {
        "message": "OT Incident Defender — OpenEnv Submission",
        "scenarios": ["task1_oldsmar", "task2_ukraine", "task3_fdi"],
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    if request.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {request.task_id!r}. Valid: {list(TASK_REGISTRY)}"
        )
    obs = _env.reset(task_id=request.task_id, seed=request.seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        result = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def state() -> dict:
    return _env.get_state()


@app.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list[TaskInfo]:
    return list(TASK_REGISTRY.values())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("environment.server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
