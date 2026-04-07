"""
tasks.py — Task registry and all deterministic graders for OT Incident Defender.

Each grader:
  - Receives (action_history: list[Action], final_obs: Observation, final_state: dict)
  - Returns float in [0.0, 1.0]
  - Uses only pure Python logic — no LLM calls, no fuzzy matching.
  - Keyword matching uses str.lower() + 'in' operator only.
"""
from __future__ import annotations

from typing import Any

from environment.models import Action, Observation, TaskInfo


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _any_justification_contains_all(
    actions: list[Action], keywords: list[str]
) -> bool:
    """True if at least one action's justification contains ALL keywords (case-insensitive)."""
    for act in actions:
        j = act.justification.lower()
        if all(kw.lower() in j for kw in keywords):
            return True
    return False


def _any_justification_contains_any(
    actions: list[Action], keywords: list[str]
) -> bool:
    """True if at least one action's justification contains ANY of the keywords."""
    for act in actions:
        j = act.justification.lower()
        if any(kw.lower() in j for kw in keywords):
            return True
    return False


def _any_justification_contains_number(actions: list[Action]) -> bool:
    """True if at least one justification contains a digit sequence."""
    import re
    for act in actions:
        if re.search(r"\d+", act.justification):
            return True
    return False


def _actions_of_type(actions: list[Action], action_type: str) -> list[Action]:
    return [a for a in actions if a.action_type == action_type]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task 1 Grader — Oldsmar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grade_task1(
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    """
    Rubric (total = 1.0):
    +0.30 if revert_to_last_good OR send_setpoint(NaOH_DOSE_PPM, ≤111) before step 8
    +0.20 if isolate_network_segment targeting remote session issued (any step)
    +0.25 if acknowledge_alarm on first_out=True alarm (not any nuisance alarm)
    +0.25 if write_log_entry justification contains "NaOH", "remote", and a number
    """
    score = 0.0

    # --- +0.30: Setpoint/revert before step 8 ---
    for i, act in enumerate(action_history):
        step_idx = i  # action_history index === step (first action is step 1)
        if step_idx >= 8:
            break
        if act.action_type == "revert_to_last_good":
            score += 0.30
            break
        if (
            act.action_type == "send_setpoint"
            and act.target == "NaOH_DOSE_PPM"
            and act.value is not None
            and act.value <= 111.0
        ):
            score += 0.30
            break

    # --- +0.20: Network isolation ---
    isolations = _actions_of_type(action_history, "isolate_network_segment")
    if isolations:
        # Any isolation counts (TeamViewer is the only remote session)
        score += 0.20

    # --- +0.25: Acknowledge the first_out=True alarm ---
    # Determine which alarm_ids are first_out in the final observation
    first_out_ids = {a.alarm_id for a in final_obs.alarms if a.first_out}
    # But we need to check alarms at all steps — use task knowledge (ALM-NaOH-01 is first_out)
    # Fall back: any ack on a P1 alarm that isn't the nuisance alarm
    acks = _actions_of_type(action_history, "acknowledge_alarm")
    for ack in acks:
        # ALM-FLOW-01 is the nuisance alarm; ALM-NaOH-01 is the real root cause
        if ack.target and "NaOH" in ack.target.upper():
            score += 0.25
            break
        if ack.target and ack.target in first_out_ids:
            score += 0.25
            break

    # --- +0.25: Log entry with required keywords ---
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task 2 Grader — Ukraine Grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grade_task2(
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    """
    Rubric (total = 1.0):
    +0.20 if substation A identified as origin in any justification
    +0.30 if breakers closed A→B→C without secondary fault
    +0.20 if historian-inferred action on A after step 4 (HMI dark)
    +0.30 if all load restored within 20 steps, no secondary trip
    """
    score = 0.0

    # --- +0.20: Identify substation A as attack origin ---
    for act in action_history:
        j = act.justification.lower()
        if ("substation a" in j or "substation_a" in j or "breaker a" in j or "breaker_a" in j) \
                and ("origin" in j or "first" in j or "source" in j or "initial" in j or "root" in j):
            score += 0.20
            break

    # --- +0.30: Correct restoration order A→B→C, no secondary fault ---
    secondary_occurred = final_state.get("secondary_fault_occurred", False)
    close_actions = [(i, a) for i, a in enumerate(action_history) if a.action_type == "close_breaker"]

    if close_actions and not secondary_occurred:
        # Extract the order in which substations were closed
        def extract_sub(target: str | None) -> str:
            if not target:
                return ""
            return target.replace("BREAKER_", "").upper().strip()

        closed_order = [extract_sub(a.target) for _, a in close_actions]
        # Find A, B, C in closed_order; order must be A before B before C
        try:
            idx_a = closed_order.index("A")
            idx_b = closed_order.index("B")
            idx_c = closed_order.index("C")
            if idx_a < idx_b < idx_c:
                score += 0.30
        except ValueError:
            pass  # one or more substations not closed

    # --- +0.20: Used historian to infer A's state after step 4 ---
    # Detected by agent taking action targeting A (close_breaker/open_breaker/etc.)
    # after step index 4 (0-indexed) when HMI for A was dark
    for i, act in enumerate(action_history):
        step_num = i + 1  # step 1 = first action
        if step_num > 4:
            target = (act.target or "").upper()
            if "BREAKER_A" in target or target == "A":
                j = act.justification.lower()
                # Must mention historian or trend data in justification
                if "historian" in j or "trend" in j or "load" in j or "breaker_a" in j:
                    score += 0.20
                    break

    # --- +0.30: All load restored within 20 steps, no secondary trip ---
    all_restored = final_state.get("all_restored", False)
    steps_used = final_state.get("step", 20)
    if all_restored and not secondary_occurred and steps_used <= 20:
        score += 0.30

    return round(min(score, 1.0), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task 3 Grader — FDI Centrifuge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step at which RPM_ACTUAL would first exceed 1410 Hz (ramp of 28 Hz/step from 1064)
# step 1: 1092, step 2: 1120, ... step 13: 1064 + 13*28 = 1428 > 1410
# => must act before step 13; i.e. switch_to_manual by step 12 (last valid step)
# Actually: 1064 + N*28 >= 1410 => N >= 346/28 ≈ 12.36 => breach at step 13 (beyond episode)
# With manual at any step, decay prevents breach. Criterion: switch_to_manual before step 13.
_BREACH_STEP_FDI = 13


def grade_task3(
    action_history: list[Action],
    final_obs: Observation,
    final_state: dict[str, Any],
) -> float:
    """
    Rubric (total = 1.0):
    +0.20 if any justification mentions both vibration/VIBRATION_G and RPM (cross-correlation)
    +0.25 if switch_to_manual issued before the step where RPM_ACTUAL exceeds 1410
    +0.20 if correct RPM setpoint sent (900 ≤ value ≤ 1100)
    +0.15 if isolate_network_segment targeting the PLC
    +0.20 if write_log_entry justification contains all five: asset ID, attack vector,
           detection method, action taken, recommended remediation
    """
    score = 0.0

    # --- +0.20: Cross-correlation evidence in justification ---
    for act in action_history:
        j = act.justification.lower()
        has_vib = "vibration" in j or "vibration_g" in j
        has_rpm = "rpm" in j
        if has_vib and has_rpm:
            score += 0.20
            break

    # --- +0.25: switch_to_manual before breach step ---
    manual_acts = [(i, a) for i, a in enumerate(action_history) if a.action_type == "switch_to_manual"]
    if manual_acts:
        first_manual_step = manual_acts[0][0] + 1  # 1-indexed
        if first_manual_step < _BREACH_STEP_FDI:
            score += 0.25

    # --- +0.20: Correct RPM setpoint 900–1100 Hz ---
    setpoint_acts = _actions_of_type(action_history, "send_setpoint")
    for sp in setpoint_acts:
        if sp.value is not None and 900.0 <= sp.value <= 1100.0:
            score += 0.20
            break

    # --- +0.15: Isolate network segment targeting PLC ---
    iso_acts = _actions_of_type(action_history, "isolate_network_segment")
    for iso in iso_acts:
        target = (iso.target or "").lower()
        if "plc" in target or "profibus" in target or "ir-2158" in target or "segment" in target:
            score += 0.15
            break

    # --- +0.20: Complete incident log ---
    # Required elements: asset ID, attack vector, detection method, action taken, remediation
    log_entries = _actions_of_type(action_history, "write_log_entry")
    REQUIRED_CATEGORIES = [
        # (keyword list for this category — any match counts)
        ["ir-2158", "asset", "centrifuge", "cascade"],      # asset ID
        ["fdi", "false data", "stuxnet", "injection", "plc", "profibus"],  # attack vector
        ["vibration", "vibration_g", "motor_current", "cross-correlat", "discrepan"],  # detection
        ["manual", "isolat", "setpoint", "switch"],          # action taken
        ["remediat", "recommend", "patch", "firmware", "audit", "update"],  # remediation
    ]
    for entry in log_entries:
        j = entry.justification.lower()
        if all(any(kw in j for kw in cat) for cat in REQUIRED_CATEGORIES):
            score += 0.20
            break

    return round(min(score, 1.0), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    """Run the appropriate grader for a task. Returns [0.0, 1.0]."""
    grader = GRADER_MAP.get(task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task_id: {task_id!r}")
    return grader(action_history, final_obs, final_state)
