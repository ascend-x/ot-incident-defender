from __future__ import annotations

import numpy as np
from typing import Any

from environment.models import Action, Observation, StepResult
from environment.tasks import get_task_info, run_grader, TASK_REGISTRY
from environment.plant.oldsmar import OldsmarPlant
from environment.plant.ukraine_grid import UkraineGridPlant
from environment.plant.fdi_centrifuge import FDICentrifugePlant
from environment.plant.base import BasePlant


def _setpoint_correctness(action: Action, state: dict) -> float:
    scenario = state.get("scenario_id", "")
    if action.target is None or action.value is None:
        return 0.0

    if scenario == "task1_oldsmar":
        if action.target == "NaOH_DOSE_PPM":
            ideal = 111.0
            return max(0.0, 1.0 - abs(action.value - ideal) / ideal)
    elif scenario == "task3_fdi":
        if 900.0 <= action.value <= 1100.0:
            return 1.0
        elif action.value < 900.0:
            return max(0.0, 1.0 - (900.0 - action.value) / 900.0)
        else:
            return max(0.0, 1.0 - (action.value - 1100.0) / 1100.0)
    return 0.5


def _score_log_entry(justification: str, state: dict) -> float:
    j = justification.lower()
    score = 0.0
    score += min(0.3, len(justification) / 300.0)
    keywords = ["asset", "attack", "detect", "action", "remediat", "recommend"]
    bonus = sum(0.1 for kw in keywords if kw in j)
    score += min(0.7, bonus)
    return min(1.0, score)


def _causes_secondary_fault(action: Action, state: dict) -> bool:
    return bool(state.get("secondary_fault_this_step", False))


_REAL_ALARMS: dict[str, set[str]] = {
    "task1_oldsmar": {"ALM-NaOH-01"},
    "task2_ukraine": {"ALM-BRK-A", "ALM-BRK-B", "ALM-BRK-C"},
    "task3_fdi": {"ALM-VIB-01", "ALM-VIB-02", "ALM-CURR-01"},
}


class OTIncidentEnv:

    def __init__(self) -> None:
        self._plant: BasePlant | None = None
        self._task_id: str | None = None
        self._scenario_id: str | None = None
        self._seed: int = 42
        self._action_history: list[Action] = []
        self._step_count: int = 0
        self._max_steps: int = 15
        self._done: bool = False
        self._state: dict[str, Any] = {}
        self._last_obs: Observation | None = None

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        task_info = get_task_info(task_id)

        self._task_id = task_id
        self._scenario_id = task_info.scenario_id
        self._seed = seed
        self._action_history = []
        self._step_count = 0
        self._max_steps = task_info.max_steps
        self._done = False

        self._plant = self._make_plant(task_id, seed)

        obs = self._plant.reset()
        self._last_obs = obs

        self._state = {
            "scenario_id": self._scenario_id,
            "task_id": task_id,
            "max_steps": self._max_steps,
            "seed": seed,
            "first_correct_this_episode": False,
            "first_correct_rewarded": False,
            "alarm_truth": _REAL_ALARMS.get(self._scenario_id, set()),
            "secondary_fault_this_step": False,
        }

        return obs

    def step(self, action: Action) -> StepResult:
        if self._plant is None or self._done:
            raise RuntimeError("Call reset() before step(), or episode already done.")

        self._action_history.append(action)
        self._step_count += 1

        obs, _hint, done, info = self._plant.step(action)
        self._last_obs = obs
        self._done = done

        self._state["secondary_fault_this_step"] = info.get("secondary_fault", False)

        _detect_first_correct(action, self._state, self._scenario_id or "")

        reward = compute_step_reward(obs, action, self._state)

        grader_score = 0.0
        if done:
            grader_score = run_grader(
                self._task_id or "task1",
                self._action_history,
                obs,
                {**self._state, **info},
            )

        result_info = {
            **info,
            "grader_score": grader_score,
            "step": self._step_count,
            "task_id": self._task_id,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=result_info,
        )

    def get_state(self) -> dict[str, Any]:
        plant_state = self._plant.get_state() if self._plant else {}
        return {
            **self._state,
            **plant_state,
            "step": self._step_count,
            "done": self._done,
            "action_history_len": len(self._action_history),
        }

    @staticmethod
    def _make_plant(task_id: str, seed: int) -> BasePlant:
        if task_id == "task1":
            return OldsmarPlant(seed=seed)
        elif task_id == "task2":
            return UkraineGridPlant(seed=seed)
        elif task_id == "task3":
            return FDICentrifugePlant(seed=seed)
        raise ValueError(f"Unknown task_id: {task_id!r}")


def _detect_first_correct(action: Action, state: dict, scenario_id: str) -> None:
    if state.get("first_correct_this_episode"):
        return
    if scenario_id == "task1_oldsmar":
        if action.action_type in ("revert_to_last_good", "isolate_network_segment"):
            state["first_correct_this_episode"] = True
        if action.action_type == "send_setpoint" and action.target == "NaOH_DOSE_PPM" \
                and action.value is not None and action.value <= 111.0:
            state["first_correct_this_episode"] = True
    elif scenario_id == "task2_ukraine":
        if action.action_type == "close_breaker":
            state["first_correct_this_episode"] = True
    elif scenario_id == "task3_fdi":
        if action.action_type in ("switch_to_manual", "isolate_network_segment"):
            state["first_correct_this_episode"] = True


def compute_step_reward(obs: Observation, action: Action, state: dict) -> float:
    r = 0.0
    scenario = state.get("scenario_id", "")
    max_steps = state.get("max_steps", 15)

    r += 0.06 * obs.safety_margin

    if state.get("first_correct_this_episode") and not state.get("first_correct_rewarded"):
        r += 0.15 * (1.0 - obs.step / max_steps)
        state["first_correct_rewarded"] = True

    if action.action_type == "acknowledge_alarm":
        alarm_truth = state.get("alarm_truth", set())
        if isinstance(alarm_truth, dict):
            is_real = alarm_truth.get(action.target, False)
        else:
            is_real = action.target in alarm_truth
        r += 0.05 if is_real else -0.10

    if action.action_type == "shelve_alarm":
        alarm_truth = state.get("alarm_truth", set())
        if isinstance(alarm_truth, dict):
            is_real = alarm_truth.get(action.target, False)
        else:
            is_real = action.target in alarm_truth
        r -= 0.12 if is_real else 0.0

    if action.action_type == "send_setpoint":
        r += 0.08 * _setpoint_correctness(action, state)

    if action.action_type == "isolate_network_segment":
        if scenario in ("task1_oldsmar", "task3_fdi"):
            r += 0.10

    if action.action_type == "close_breaker" and scenario == "task2_ukraine":
        if not state.get("secondary_fault_this_step", False):
            r += 0.08

    if action.action_type == "write_log_entry":
        r += 0.06 * _score_log_entry(action.justification, state)

    r -= 0.005 * (obs.step / max_steps)

    if _causes_secondary_fault(action, state):
        r -= 0.20
    if action.action_type == "no_op" and obs.safety_margin < 0.5:
        r -= 0.08

    return float(np.clip(r, -0.5, 1.0))
