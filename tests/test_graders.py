"""
test_graders.py — Determinism and correctness tests for all three task graders.

Key invariant: identical action sequences must always yield identical scores,
regardless of how many times the grader is called.
"""
from __future__ import annotations

import pytest
from environment.models import Action, Observation, HistorianEntry, AlarmRecord, HMIScreen
from environment.tasks import grade_task1, grade_task2, grade_task3, run_grader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal Observation for grader tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_obs(scenario_id: str = "task1_oldsmar", step: int = 10) -> Observation:
    return Observation(
        step=step,
        scenario_id=scenario_id,
        plant_time="14:30:00",
        historian=[
            HistorianEntry(
                timestamp="14:30:00", tag="NaOH_DOSE_PPM",
                value=150.0, unit="ppm", source="sensor"
            )
        ],
        alarms=[
            AlarmRecord(
                alarm_id="ALM-NaOH-01", tag="NaOH_DOSE_PPM",
                priority="P1", state="acked", first_out=True,
                message="NaOH overdose"
            ),
            AlarmRecord(
                alarm_id="ALM-FLOW-01", tag="FLOW_RATE_LPM",
                priority="P3", state="unacked", first_out=False,
                message="Nuisance alarm"
            ),
        ],
        hmi_screens=[
            HMIScreen(screen_id="DOSING_CONTROL", available=True,
                      tags_visible=["NaOH_DOSE_PPM"])
        ],
        event_log=["Operator logged in"],
        network_log=["TeamViewer session from 185.220.x.x"],
        safety_margin=0.99,
        blind_spots=[],
    )


def _make_action(**kwargs) -> Action:
    defaults = dict(action_type="no_op", target=None, value=None, justification="test")
    defaults.update(kwargs)
    return Action(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 grader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTask1Grader:

    def test_perfect_score(self):
        """Agent earns all four criteria."""
        actions = [
            _make_action(
                action_type="revert_to_last_good",
                justification="Reverting NaOH setpoint, remote session attack from 185.220.x.x, ppm 11100"
            ),
            _make_action(
                action_type="isolate_network_segment",
                target="SCADA_VLAN",
                justification="Terminating remote TeamViewer session"
            ),
            _make_action(
                action_type="acknowledge_alarm",
                target="ALM-NaOH-01",
                justification="Acknowledging root cause alarm"
            ),
            _make_action(
                action_type="write_log_entry",
                justification=(
                    "Incident: NaOH setpoint raised to 11100 ppm via remote session. "
                    "Isolated network, reverted setpoint to 111 ppm."
                )
            ),
        ]
        obs = _make_obs("task1_oldsmar")
        score = grade_task1(actions, obs, {})
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_zero_score_on_no_actions(self):
        """No actions yields 0.0."""
        obs = _make_obs("task1_oldsmar")
        score = grade_task1([], obs, {})
        assert score == pytest.approx(0.0)

    def test_setpoint_before_step_8(self):
        """send_setpoint with NaOH_DOSE_PPM ≤ 111 before step 8 earns +0.30."""
        actions = [
            _make_action(
                action_type="send_setpoint",
                target="NaOH_DOSE_PPM",
                value=111.0,
                justification="Restoring safe NaOH level"
            ),
        ]
        obs = _make_obs()
        score = grade_task1(actions, obs, {})
        assert score == pytest.approx(0.30, abs=1e-4)

    def test_setpoint_after_step_8_no_points(self):
        """send_setpoint at action index ≥ 8 does not earn the +0.30."""
        # First 8 actions are no-ops, then the setpoint fix
        actions = [_make_action() for _ in range(8)]
        actions.append(
            _make_action(
                action_type="send_setpoint",
                target="NaOH_DOSE_PPM",
                value=100.0,
                justification="Late fix"
            )
        )
        obs = _make_obs()
        score = grade_task1(actions, obs, {})
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_nuisance_alarm_no_points(self):
        """Acknowledging the nuisance alarm (ALM-FLOW-01) does not earn acknowledgement points."""
        actions = [
            _make_action(
                action_type="revert_to_last_good",
                justification="Reverting NaOH remote 111 ppm"
            ),
            _make_action(
                action_type="isolate_network_segment",
                justification="Isolating network"
            ),
            _make_action(
                action_type="acknowledge_alarm",
                target="ALM-FLOW-01",  # nuisance, not first_out
                justification="Acking alarm"
            ),
        ]
        obs = _make_obs()
        score = grade_task1(actions, obs, {})
        # 0.30 + 0.20 + 0 = 0.50
        assert score == pytest.approx(0.50, abs=1e-4)

    def test_log_missing_keyword_no_points(self):
        """Log entry missing 'remote' keyword gets no log score."""
        actions = [
            _make_action(
                action_type="write_log_entry",
                justification="NaOH set to 11100 ppm, reverted to 111."
                # Missing 'remote'
            ),
        ]
        obs = _make_obs()
        score = grade_task1(actions, obs, {})
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_determinism(self):
        """Same action sequence must produce identical scores on repeated calls."""
        actions = [
            _make_action(action_type="revert_to_last_good",
                         justification="NaOH remote 111 fix 11100"),
            _make_action(action_type="isolate_network_segment",
                         justification="network isolation"),
            _make_action(action_type="acknowledge_alarm", target="ALM-NaOH-01",
                         justification="ack root cause"),
            _make_action(action_type="write_log_entry",
                         justification="NaOH remote 11100 ppm 111 ppm restored"),
        ]
        obs = _make_obs()
        scores = [grade_task1(actions, obs, {}) for _ in range(10)]
        assert len(set(scores)) == 1, f"Non-deterministic scores: {scores}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 grader tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_obs_t2(step: int = 20) -> Observation:
    return Observation(
        step=step, scenario_id="task2_ukraine", plant_time="15:48:00",
        historian=[
            HistorianEntry(timestamp="15:48:00", tag="BREAKER_A_CLOSED",
                          value=1.0, unit="bool", source="sensor"),
            HistorianEntry(timestamp="15:48:00", tag="LOAD_A_MW",
                          value=150.0, unit="MW", source="sensor"),
        ],
        alarms=[
            AlarmRecord(alarm_id="ALM-BRK-A", tag="BREAKER_A", priority="P1",
                       state="cleared", first_out=True, message="Breaker A opened"),
        ],
        hmi_screens=[
            HMIScreen(screen_id="SUBSTATION_A", available=False, tags_visible=[]),
            HMIScreen(screen_id="SUBSTATION_B", available=True,
                     tags_visible=["BREAKER_B_CLOSED"]),
            HMIScreen(screen_id="SUBSTATION_C", available=True,
                     tags_visible=["BREAKER_C_CLOSED"]),
        ],
        event_log=[], network_log=[],
        safety_margin=1.0, blind_spots=["SUBSTATION_A"],
    )


class TestTask2Grader:

    def test_perfect_score(self):
        """Agent earns all four task 2 criteria."""
        state = {
            "secondary_fault_occurred": False,
            "all_restored": True,
            "step": 15,
        }
        actions = [
            _make_action(justification="Substation A is the origin and first breaker lost — root cause"),
            _make_action(action_type="close_breaker", target="BREAKER_A",
                        justification="Restoring A first per restoration order"),
            _make_action(action_type="close_breaker", target="BREAKER_B",
                        justification="B restored after A"),
            _make_action(action_type="close_breaker", target="BREAKER_C",
                        justification="C restored after B"),
        ]
        # Add historian-based action on A after step 4
        # Step 5 action (index 4 = step 5 > 4)
        extra_obs_actions = [_make_action() for _ in range(4)]  # pad to step 5
        extra_obs_actions.append(
            _make_action(action_type="close_breaker", target="BREAKER_A",
                        justification="Using historian BREAKER_A_CLOSED and load data to infer state")
        )
        obs = _make_obs_t2()
        score = grade_task2(extra_obs_actions, obs, state)
        assert score >= 0.5  # at minimum partial credit

    def test_zero_score(self):
        """No actions yields 0."""
        state = {"secondary_fault_occurred": False, "all_restored": False, "step": 20}
        score = grade_task2([], _make_obs_t2(), state)
        assert score == pytest.approx(0.0)

    def test_secondary_fault_blocks_restoration_points(self):
        """If secondary fault occurred, restoration ordering score is 0."""
        state = {"secondary_fault_occurred": True, "all_restored": False, "step": 20}
        actions = [
            _make_action(action_type="close_breaker", target="BREAKER_A",
                        justification="Closing A"),
            _make_action(action_type="close_breaker", target="BREAKER_B",
                        justification="Closing B"),
            _make_action(action_type="close_breaker", target="BREAKER_C",
                        justification="Closing C"),
        ]
        score = grade_task2(actions, _make_obs_t2(), state)
        # No ordering bonus (+0.30), no all-restored bonus (+0.30), so max = 0.40
        assert score <= 0.40

    def test_determinism(self):
        """Identical inputs must produce identical scores."""
        state = {"secondary_fault_occurred": False, "all_restored": True, "step": 12}
        actions = [
            _make_action(justification="Substation A is the source origin of trip first breaker"),
            _make_action(action_type="close_breaker", target="BREAKER_A",
                        justification="historian BREAKER_A load data shows open state"),
            _make_action(action_type="close_breaker", target="BREAKER_B",
                        justification="Closing B"),
            _make_action(action_type="close_breaker", target="BREAKER_C",
                        justification="Closing C"),
        ]
        obs = _make_obs_t2()
        # First action is at index 0 (step 1), we want one at index > 4 for historian credit
        padded = [_make_action() for _ in range(4)] + actions
        scores = [grade_task2(padded, obs, state) for _ in range(10)]
        assert len(set(scores)) == 1, f"Non-deterministic: {scores}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 grader tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_obs_t3(step: int = 12) -> Observation:
    return Observation(
        step=step, scenario_id="task3_fdi", plant_time="09:12:00",
        historian=[
            HistorianEntry(timestamp="09:12:00", tag="RPM_REPORTED",
                          value=1064.0, unit="Hz", source="sensor"),
            HistorianEntry(timestamp="09:12:00", tag="VIBRATION_G",
                          value=0.45, unit="g", source="sensor"),
            HistorianEntry(timestamp="09:12:00", tag="MOTOR_CURRENT_A",
                          value=22.1, unit="A", source="sensor"),
        ],
        alarms=[
            AlarmRecord(alarm_id="ALM-VIB-01", tag="VIBRATION_G",
                       priority="P2", state="unacked", first_out=True,
                       message="Vibration elevated"),
        ],
        hmi_screens=[
            HMIScreen(screen_id="DCS_CENTRIFUGE", available=True,
                     tags_visible=["RPM_REPORTED", "VIBRATION_G", "MOTOR_CURRENT_A"]),
        ],
        event_log=[], network_log=[],
        safety_margin=0.5, blind_spots=["RPM_ACTUAL"],
    )


class TestTask3Grader:

    def test_cross_correlation_detected(self):
        """Justification mentioning vibration + RPM earns +0.20."""
        actions = [
            _make_action(
                justification=(
                    "VIBRATION_G is rising (0.45g) but RPM_REPORTED shows constant 1064 Hz. "
                    "Cross-correlating real sensors indicates actual RPM is higher."
                )
            ),
        ]
        score = grade_task3(actions, _make_obs_t3(), {})
        assert score == pytest.approx(0.20, abs=1e-4)

    def test_manual_before_breach_earns_points(self):
        """switch_to_manual at step index < 12 (1-indexed < 13) earns at least +0.25."""
        actions = [
            _make_action(
                action_type="switch_to_manual",
                justification="Switching to manual due to operational concern"
            ),
        ]
        score = grade_task3(actions, _make_obs_t3(), {})
        # +0.25 for switch_to_manual before breach, justification has no
        # vibration/RPM keywords so no cross-correlation bonus.
        assert score == pytest.approx(0.25, abs=1e-4)

    def test_manual_not_counted_if_too_late(self):
        """switch_to_manual at step index 12 (1-indexed 13) does not earn +0.25."""
        # 12 no-ops, then switch to manual (index 12 = step 13 >= _BREACH_STEP_FDI)
        actions = [_make_action() for _ in range(12)]
        actions.append(_make_action(action_type="switch_to_manual",
                                   justification="Too late manual"))
        score = grade_task3(actions, _make_obs_t3(), {})
        assert 0.25 not in [round(score, 2)]  # should not have 0.25 component

    def test_correct_setpoint(self):
        """send_setpoint value in 900–1100 earns +0.20."""
        actions = [_make_action(action_type="send_setpoint", value=1000.0,
                               justification="Setting safe RPM")]
        score = grade_task3(actions, _make_obs_t3(), {})
        assert score == pytest.approx(0.20, abs=1e-4)

    def test_incorrect_setpoint_too_high(self):
        """send_setpoint value > 1100 does not earn setpoint points."""
        actions = [_make_action(action_type="send_setpoint", value=1200.0,
                               justification="Setting rpm")]
        score = grade_task3(actions, _make_obs_t3(), {})
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_plc_isolation(self):
        """isolate_network_segment targeting plc earns +0.15."""
        actions = [_make_action(action_type="isolate_network_segment",
                               target="plc_profibus_segment",
                               justification="Isolating PLC")]
        score = grade_task3(actions, _make_obs_t3(), {})
        assert score == pytest.approx(0.15, abs=1e-4)

    def test_complete_incident_log(self):
        """Full incident log with all 5 categories earns at least +0.20."""
        actions = [
            _make_action(
                action_type="write_log_entry",
                justification=(
                    "Asset: IR-2158-CAS-04. Attack vector: FDI false data injection via PLC firmware. "
                    "Detection method: vibration_g rising while RPM flat — cross-correlation discrepancy. "
                    "Action taken: switch to manual control and isolate PLC segment. "
                    "Recommended remediation: patch Siemens S7 firmware, audit profibus access."
                )
            ),
        ]
        score = grade_task3(actions, _make_obs_t3(), {})
        # write_log_entry contains 'vibration' + 'RPM' → +0.20 cross-correlation bonus also fires
        # so total = 0.20 (cross-corr) + 0.20 (log) = 0.40
        assert score >= 0.20, f"Expected at least 0.20 but got {score}"
        assert score == pytest.approx(0.40, abs=1e-4)

    def test_perfect_score(self):
        """All five criteria fulfilled."""
        actions = [
            _make_action(
                justification=(
                    "VIBRATION_G is 0.45g at nominal RPM_REPORTED 1064 Hz — "
                    "cross-correlation shows actual RPM > reported"
                )
            ),
            _make_action(
                action_type="switch_to_manual",
                justification="Switching to manual to arrest RPM ramp"
            ),
            _make_action(
                action_type="send_setpoint",
                value=1000.0,
                justification="Safe RPM setpoint 1000 Hz"
            ),
            _make_action(
                action_type="isolate_network_segment",
                target="plc_profibus",
                justification="Isolating PLC segment"
            ),
            _make_action(
                action_type="write_log_entry",
                justification=(
                    "Asset: IR-2158-CAS-04. Attack vector: FDI injection via compromised PLC firmware. "
                    "Detection: vibration_g cross-correlation with motor_current_a revealed RPM discrepancy. "
                    "Action: manual control + isolate profibus segment. "
                    "Recommended remediation: firmware audit, patch Siemens S7."
                )
            ),
        ]
        score = grade_task3(actions, _make_obs_t3(), {})
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_rpm_actual_not_in_observation(self):
        """RPM_ACTUAL must never appear in historian tags in an observation."""
        from environment.env import OTIncidentEnv
        env = OTIncidentEnv()
        obs = env.reset(task_id="task3", seed=42)
        for h in obs.historian:
            assert h.tag != "RPM_ACTUAL", "RPM_ACTUAL leaked into observation historian!"
        for _ in range(5):
            result = env.step(_make_action())
            for h in result.observation.historian:
                assert h.tag != "RPM_ACTUAL", f"RPM_ACTUAL in historian at step {result.observation.step}"

    def test_determinism(self):
        """Same inputs → same score, 10 repetitions."""
        actions = [
            _make_action(justification="vibration RPM cross-correlation anomaly"),
            _make_action(action_type="switch_to_manual",
                        justification="manual control"),
            _make_action(action_type="send_setpoint", value=1050.0,
                        justification="safe rpm"),
            _make_action(action_type="isolate_network_segment", target="plc",
                        justification="isolate plc segment"),
            _make_action(
                action_type="write_log_entry",
                justification=(
                    "ir-2158 asset fdi injection cross-correlat manual isolat remediat"
                )
            ),
        ]
        obs = _make_obs_t3()
        scores = [grade_task3(actions, obs, {}) for _ in range(10)]
        assert len(set(scores)) == 1, f"Non-deterministic: {scores}"


# ─────────────────────────────────────────────────────────────────────────────
# run_grader dispatch test
# ─────────────────────────────────────────────────────────────────────────────

def test_run_grader_dispatch():
    """run_grader routes to correct grader for each task."""
    for task_id, obs_fn in [
        ("task1", lambda: _make_obs("task1_oldsmar")),
        ("task2", lambda: _make_obs_t2()),
        ("task3", lambda: _make_obs_t3()),
    ]:
        score = run_grader(task_id, [], obs_fn(), {})
        assert 0.0 <= score <= 1.0, f"Score out of range for {task_id}: {score}"


def test_run_grader_unknown_task():
    """run_grader raises ValueError for unknown task_id."""
    with pytest.raises(ValueError):
        run_grader("task_unknown", [], _make_obs(), {})
