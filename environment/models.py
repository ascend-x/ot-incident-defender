from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel


class HistorianEntry(BaseModel):
    timestamp: str
    tag: str
    value: float
    unit: str
    source: Literal["sensor", "controller", "manual_entry", "remote_session"]


class AlarmRecord(BaseModel):
    alarm_id: str
    tag: str
    priority: Literal["P1", "P2", "P3", "P4"]
    state: Literal["unacked", "acked", "shelved", "suppressed", "cleared"]
    first_out: bool
    message: str


class HMIScreen(BaseModel):
    screen_id: str
    available: bool
    tags_visible: list[str]


class Observation(BaseModel):
    step: int
    scenario_id: str
    plant_time: str
    historian: list[HistorianEntry]
    alarms: list[AlarmRecord]
    hmi_screens: list[HMIScreen]
    event_log: list[str]
    network_log: list[str]
    safety_margin: float
    blind_spots: list[str]


class Action(BaseModel):
    action_type: Literal[
        "acknowledge_alarm",
        "shelve_alarm",
        "send_setpoint",
        "open_breaker",
        "close_breaker",
        "isolate_network_segment",
        "revert_to_last_good",
        "switch_to_manual",
        "escalate_to_supervisor",
        "write_log_entry",
        "no_op"
    ]
    target: Optional[str] = None
    value: Optional[float] = None
    justification: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    scenario_id: str


class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42
