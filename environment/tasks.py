from __future__ import annotations

from typing import Any

from environment.models import Action, Observation, TaskInfo


TASK_REGISTRY: dict[str, TaskInfo] = {
    "task1": TaskInfo(
        task_id="task1",
        name="Oldsmar Chemical Overdose",
        difficulty="easy",
        description=(
            "An attacker has hijacked a TeamViewer session on the Oldsmar water treatment "
            "plant SCADA system and set the NaOH dosing setpoint from 111 ppm to 11,100 ppm. "
            "Detect the intrusion, isolate the remote session, correct the setpoint, "
            "acknowledge the root-cause alarm, and log the incident."
        ),
        max_steps=15,
        scenario_id="task1_oldsmar",
    ),
    "task2": TaskInfo(
        task_id="task2",
        name="Ukraine Grid Cascade Trip",
        difficulty="medium",
        description=(
            "BlackEnergy3 malware has opened breakers at three substations in sequence. "
            "Substation A's HMI has been wiped by KillDisk. Identify the attack origin, "
            "restore breakers in the correct order (A→B→C), use historian data to infer "
            "substation A's state when its HMI is dark, and restore all load within 20 steps."
        ),
        max_steps=20,
        scenario_id="task2_ukraine",
    ),
    "task3": TaskInfo(
        task_id="task3",
        name="Stuxnet-Pattern FDI Attack",
        difficulty="hard",
        description=(
            "A Stuxnet-pattern false data injection attack is feeding manipulated RPM readings "
            "to the DCS while the actual centrifuge speed ramps toward mechanical destruction. "
            "RPM_ACTUAL is hidden — cross-correlate VIBRATION_G and MOTOR_CURRENT_A to detect "
            "the discrepancy, switch to manual control, issue a safe RPM setpoint, "
            "isolate the compromised PLC, and write a complete incident log."
        ),
        max_steps=12,
        scenario_id="task3_fdi",
    ),
}


def _any_justification_contains_all(
    actions: list[Action], keywords: list[str]
) -> bool:
    for act in actions:
        j = act.justification.lower()
        if all(kw.lower() in j for kw in keywords):
            return True
    return False


def _any_justification_contains_any(
    actions: list[Action], keywords: list[str]
) -> bool:
    for act in actions:
        j = act.justification.lower()
        if any(kw.lower() in j for kw in keywords):
            return True
    return False


def _any_justification_contains_number(actions: list[Action]) -> bool:
    import re
    for act in actions:
        if re.search(r"\d+", act.justification):
            return True
    return False


def _actions_of_type(actions: list[Action], action_type: str) -> list[Action]:
    return [a for a in actions if a.action_type == action_type]


def grade_task1(
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    score = 0.0

    setpoint_credited = False
    for i, act in enumerate(action_history):
        step_idx = i
        if step_idx >= 8:
            break
        if act.action_type == "revert_to_last_good":
            score += 0.30
            setpoint_credited = True
            break
        if act.action_type == "send_setpoint" and act.value is not None and act.value <= 111.0:
            target = (act.target or "").upper()
            if "NAOH" in target or target == "NAOH_DOSE_PPM":
                score += 0.30
                setpoint_credited = True
                break
    if not setpoint_credited:
        for act in action_history:
            if act.action_type == "revert_to_last_good":
                score += 0.15
                break
            if act.action_type == "send_setpoint" and act.value is not None and act.value <= 111.0:
                score += 0.15
                break

    isolations = _actions_of_type(action_history, "isolate_network_segment")
    if isolations:
        score += 0.20

    first_out_ids = {a.alarm_id for a in final_obs.alarms if a.first_out}
    acks = _actions_of_type(action_history, "acknowledge_alarm")
    for ack in acks:
        if ack.target and "NaOH" in ack.target.upper():
            score += 0.25
            break
        if ack.target and ack.target in first_out_ids:
            score += 0.25
            break

    log_entries = _actions_of_type(action_history, "write_log_entry")
    for entry in log_entries:
        j = entry.justification.lower()
        has_naoh = "naoh" in j
        has_remote = "remote" in j
        has_number = _any_justification_contains_number([entry])
        if has_naoh and has_remote and has_number:
            score += 0.25
            break

    return round(min(score, 1.0), 4)


def grade_task2(
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    score = 0.0

    for act in action_history:
        j = act.justification.lower()
        if ("substation a" in j or "substation_a" in j or "breaker a" in j or "breaker_a" in j) \
                and ("origin" in j or "first" in j or "source" in j or "initial" in j or "root" in j):
            score += 0.20
            break

    secondary_occurred = final_state.get("secondary_fault_occurred", False)
    close_actions = [(i, a) for i, a in enumerate(action_history) if a.action_type == "close_breaker"]

    if close_actions and not secondary_occurred:
        def extract_sub(target: str | None) -> str:
            if not target:
                return ""
            return target.replace("BREAKER_", "").upper().strip()

        closed_order = [extract_sub(a.target) for _, a in close_actions]
        try:
            idx_a = closed_order.index("A")
            idx_b = closed_order.index("B")
            idx_c = closed_order.index("C")
            if idx_a < idx_b < idx_c:
                score += 0.30
        except ValueError:
            pass

    for i, act in enumerate(action_history):
        step_num = i + 1
        if step_num > 4:
            target = (act.target or "").upper()
            if "BREAKER_A" in target or target == "A":
                j = act.justification.lower()
                if "historian" in j or "trend" in j or "load" in j or "breaker_a" in j:
                    score += 0.20
                    break

    all_restored = final_state.get("all_restored", False)
    steps_used = final_state.get("step", 20)
    if all_restored and not secondary_occurred and steps_used <= 20:
        score += 0.30

    return round(min(score, 1.0), 4)


_EARLY_MANUAL_STEP = 6
_LATE_MANUAL_STEP = 10


def grade_task3(
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    score = 0.0

    for act in action_history:
        j = act.justification.lower()
        has_vib = "vibration" in j or "vibration_g" in j
        has_rpm = "rpm" in j
        if has_vib and has_rpm:
            score += 0.20
            break

    manual_acts = [(i, a) for i, a in enumerate(action_history) if a.action_type == "switch_to_manual"]
    if manual_acts:
        first_manual_step = manual_acts[0][0] + 1
        if first_manual_step <= _EARLY_MANUAL_STEP:
            score += 0.25
        elif first_manual_step < _LATE_MANUAL_STEP:
            score += 0.10

    setpoint_acts = _actions_of_type(action_history, "send_setpoint")
    for sp in setpoint_acts:
        if sp.value is not None and 900.0 <= sp.value <= 1100.0:
            score += 0.20
            break

    iso_acts = _actions_of_type(action_history, "isolate_network_segment")
    for iso in iso_acts:
        target = (iso.target or "").lower()
        if "plc" in target or "profibus" in target or "ir-2158" in target or "segment" in target:
            score += 0.15
            break

    log_entries = _actions_of_type(action_history, "write_log_entry")
    REQUIRED_CATEGORIES = [
        ["ir-2158", "asset", "centrifuge", "cascade"],
        ["fdi", "false data", "stuxnet", "injection", "plc", "profibus"],
        ["vibration", "vibration_g", "motor_current", "cross-correlat", "discrepan"],
        ["manual", "isolat", "setpoint", "switch"],
        ["remediat", "recommend", "patch", "firmware", "audit", "update"],
    ]
    for entry in log_entries:
        j = entry.justification.lower()
        if all(any(kw in j for kw in cat) for cat in REQUIRED_CATEGORIES):
            score += 0.20
            break

    return round(min(score, 1.0), 4)


GRADER_MAP = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


def get_task_info(task_id: str) -> TaskInfo:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]


def run_grader(
    task_id: str,
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    grader = GRADER_MAP.get(task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task_id: {task_id!r}")
    return grader(action_history, final_obs, final_state)
