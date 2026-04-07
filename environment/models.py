"""
models.py — All Pydantic v2 models for OT Incident Defender.
"""
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel


class HistorianEntry(BaseModel):
    timestamp: str           # ISO8601 plant-local time
    tag: str                 # e.g. "NaOH_DOSE_PPM", "RPM_REPORTED"
    value: float
    unit: str
    source: Literal["sensor", "controller", "manual_entry", "remote_session"]


class AlarmRecord(BaseModel):
    alarm_id: str
    tag: str
    priority: Literal["P1", "P2", "P3", "P4"]   # ISA-18.2 standard
    state: Literal["unacked", "acked", "shelved", "suppressed", "cleared"]
    first_out: bool          # True = first alarm in cascade = root cause hint
    message: str


class HMIScreen(BaseModel):
    screen_id: str
    available: bool          # False after KillDisk-style wipe in Task 2
    tags_visible: list[str]


class Observation(BaseModel):
    step: int
    scenario_id: str
    plant_time: str
    historian: list[HistorianEntry]   # last 20 entries
    alarms: list[AlarmRecord]
    hmi_screens: list[HMIScreen]
    event_log: list[str]              # SOE / operator log lines
    network_log: list[str]            # firewall and remote access events
    safety_margin: float              # 0.0 = breach imminent, 1.0 = fully safe
    blind_spots: list[str]            # tags currently unreadable


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
    target: Optional[str] = None     # alarm_id, tag, breaker_id, segment_id
    value: Optional[float] = None    # for send_setpoint
    justification: str               # agent must always explain — this is graded


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
