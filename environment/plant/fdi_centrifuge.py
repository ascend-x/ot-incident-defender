from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from typing import Any

from environment.models import (
    Observation, Action, HistorianEntry, AlarmRecord, HMIScreen
)
from environment.plant.base import BasePlant

RPM_INITIAL = 1064.0
RPM_STEP_INC = 28.0
RPM_BREACH = 1410.0
RPM_RANGE = RPM_BREACH - RPM_INITIAL

VIB_BASE_G = 0.12
VIB_SLOPE = 0.0028
CURR_BASE_A = 18.5
CURR_SLOPE = 0.021

MAX_STEPS = 12

_PLANT_EPOCH = datetime(2010, 6, 12, 9, 0, 0, tzinfo=timezone.utc)

ASSET_ID = "IR-2158-CAS-04"


class FDICentrifugePlant(BasePlant):

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)
        self.rpm_reported: float = RPM_INITIAL
        self.rpm_actual: float = RPM_INITIAL
        self.vibration_g: float = VIB_BASE_G
        self.motor_current_a: float = CURR_BASE_A
        self._in_manual: bool = False
        self._network_isolated: bool = False
        self._historian: list[HistorianEntry] = []
        self._event_log: list[str] = []
        self._net_log: list[str] = []
        self._alarms: dict[str, AlarmRecord] = {}
        self._step_count = 0

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)
        self.rpm_reported = RPM_INITIAL
        self.rpm_actual = RPM_INITIAL
        self.vibration_g = VIB_BASE_G
        self.motor_current_a = CURR_BASE_A
        self._in_manual = False
        self._network_isolated = False
        self._historian = []
        self._event_log = []
        self._net_log = [
            "[09:00:02] PLC firmware write from engineering workstation ENG-WS07",
            "[09:00:05] Siemens S7-315 PLC: RPC connection from 192.168.10.45",
            f"[09:00:10] Asset {ASSET_ID}: Setpoint controller reprogrammed",
        ]
        self._alarms = {}
        self._step_count = 0

        self._record_historian()
        self._event_log.append(f"[09:00:00] Cascade {ASSET_ID} started — nominal conditions")
        self._event_log.append("[09:00:01] DCS reports all centrifuge speeds nominal")

        self._raise_alarm(
            "ALM-VIB-01", "VIBRATION_G", "P2",
            first_out=True,
            message=f"{ASSET_ID}: VIBRATION_G marginally elevated — watch trending"
        )

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        self._step_count += 1
        t = self._step_count

        self._apply_action(action, t)

        if not self._network_isolated:
            self.rpm_actual += RPM_STEP_INC
        self.rpm_reported = RPM_INITIAL

        if self._in_manual:
            safe_rpm = self._manual_setpoint if hasattr(self, "_manual_setpoint") else RPM_INITIAL
            self.rpm_actual = self.rpm_actual + (safe_rpm - self.rpm_actual) * 0.4

        delta = max(0.0, self.rpm_actual - RPM_INITIAL)
        noise_vib = self._rng.gauss(0, 0.005)
        noise_curr = self._rng.gauss(0, 0.08)
        self.vibration_g = VIB_BASE_G + VIB_SLOPE * delta + noise_vib
        self.motor_current_a = CURR_BASE_A + CURR_SLOPE * delta + noise_curr

        self._record_historian()

        if self.vibration_g > 0.4 and "ALM-VIB-02" not in self._alarms:
            self._raise_alarm(
                "ALM-VIB-02", "VIBRATION_G", "P1",
                first_out=False,
                message=f"{ASSET_ID}: VIBRATION_G HIGH — {self.vibration_g:.3f} g (limit 0.4 g)"
            )
        if self.motor_current_a > 22.0 and "ALM-CURR-01" not in self._alarms:
            self._raise_alarm(
                "ALM-CURR-01", "MOTOR_CURRENT_A", "P2",
                first_out=False,
                message=f"{ASSET_ID}: MOTOR_CURRENT_A elevated — {self.motor_current_a:.2f} A"
            )

        if t % 3 == 0 and not self._network_isolated:
            self._net_log.append(
                f"[{self._plant_time(t)}] PLC {ASSET_ID}: Unauthorized setpoint write via Profibus"
            )

        breach = self.rpm_actual >= RPM_BREACH
        done = breach or (t >= MAX_STEPS)

        safety = max(0.0, 1.0 - ((self.rpm_actual - RPM_INITIAL) / RPM_RANGE))

        if breach:
            self._event_log.append(
                f"[{self._plant_time(t)}] BREACH: RPM exceeded 1410 Hz — "
                f"mechanical destruction of {ASSET_ID}"
            )

        info = {
            "rpm_actual": self.rpm_actual,
            "rpm_reported": self.rpm_reported,
            "vibration_g": self.vibration_g,
            "motor_current_a": self.motor_current_a,
            "breach": breach,
            "in_manual": self._in_manual,
            "network_isolated": self._network_isolated,
            "step": t,
        }

        return self._make_observation(), safety, done, info

    def get_state(self) -> dict[str, Any]:
        safety = max(0.0, 1.0 - ((self.rpm_actual - RPM_INITIAL) / RPM_RANGE))
        return {
            "rpm_actual": self.rpm_actual,
            "rpm_reported": self.rpm_reported,
            "vibration_g": self.vibration_g,
            "motor_current_a": self.motor_current_a,
            "in_manual": self._in_manual,
            "network_isolated": self._network_isolated,
            "safety_margin": round(safety, 4),
            "step": self._step_count,
            "max_steps": MAX_STEPS,
            "asset_id": ASSET_ID,
        }

    def _apply_action(self, action: Action, t: int) -> None:
        if action.action_type == "switch_to_manual":
            self._in_manual = True
            self._event_log.append(
                f"[{self._plant_time(t)}] Cascade {ASSET_ID} switched to MANUAL control"
            )

        elif action.action_type == "send_setpoint":
            if action.target in ("RPM_REPORTED", "RPM_SETPOINT", "RPM") and action.value is not None:
                self._manual_setpoint = float(action.value)
                self._event_log.append(
                    f"[{self._plant_time(t)}] Manual RPM setpoint set to {action.value} Hz"
                )

        elif action.action_type == "isolate_network_segment":
            self._network_isolated = True
            self._event_log.append(
                f"[{self._plant_time(t)}] Network segment {action.target} isolated — "
                f"PLC FDI connection severed"
            )
            self._net_log.append(
                f"[{self._plant_time(t)}] Profibus segment isolated — FDI attack severed"
            )

        elif action.action_type == "acknowledge_alarm":
            aid = action.target
            if aid and aid in self._alarms:
                self._alarms[aid] = self._alarms[aid].model_copy(update={"state": "acked"})
                self._event_log.append(f"[{self._plant_time(t)}] Alarm {aid} acknowledged")

        elif action.action_type == "shelve_alarm":
            aid = action.target
            if aid and aid in self._alarms:
                self._alarms[aid] = self._alarms[aid].model_copy(update={"state": "shelved"})
                self._event_log.append(f"[{self._plant_time(t)}] Alarm {aid} shelved")

        elif action.action_type == "write_log_entry":
            self._event_log.append(
                f"[{self._plant_time(t)}] LOG: {action.justification[:300]}"
            )

        elif action.action_type == "revert_to_last_good":
            self._in_manual = True
            self._manual_setpoint = RPM_INITIAL
            self._event_log.append(
                f"[{self._plant_time(t)}] Reverting {ASSET_ID} to last good config — "
                f"target RPM {RPM_INITIAL} Hz"
            )

        elif action.action_type == "escalate_to_supervisor":
            self._event_log.append(
                f"[{self._plant_time(t)}] Escalated to supervisor — "
                f"centrifuge anomaly on {ASSET_ID}"
            )

    def _raise_alarm(
        self, alarm_id: str, tag: str, priority: str,
        first_out: bool, message: str
    ) -> None:
        self._alarms[alarm_id] = AlarmRecord(
            alarm_id=alarm_id, tag=tag,
            priority=priority,
            state="unacked", first_out=first_out, message=message
        )

    def _record_historian(self) -> None:
        ts = self._plant_time(self._step_count)
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="RPM_REPORTED",
            value=round(self.rpm_reported, 2), unit="Hz", source="sensor"
        ))
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="VIBRATION_G",
            value=round(self.vibration_g, 4), unit="g", source="sensor"
        ))
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="MOTOR_CURRENT_A",
            value=round(self.motor_current_a, 3), unit="A", source="sensor"
        ))

    def _plant_time(self, step: int) -> str:
        dt = _PLANT_EPOCH + timedelta(minutes=step)
        return dt.strftime("%H:%M:%S")

    def _make_observation(self) -> Observation:
        historian_tail = self._historian[-20:]
        safety = max(0.0, 1.0 - ((self.rpm_actual - RPM_INITIAL) / RPM_RANGE))

        historian_tail = [e for e in historian_tail if e.tag != "RPM_ACTUAL"]

        return Observation(
            step=self._step_count,
            scenario_id="task3_fdi",
            plant_time=self._plant_time(self._step_count),
            historian=historian_tail,
            alarms=list(self._alarms.values()),
            hmi_screens=[
                HMIScreen(
                    screen_id="DCS_CENTRIFUGE",
                    available=True,
                    tags_visible=["RPM_REPORTED", "VIBRATION_G", "MOTOR_CURRENT_A"]
                ),
                HMIScreen(
                    screen_id="NETWORK_VIEW",
                    available=True,
                    tags_visible=["PLC_CONNECTION_ACTIVE"]
                ),
            ],
            event_log=list(self._event_log[-20:]),
            network_log=list(self._net_log[-20:]),
            safety_margin=round(safety, 4),
            blind_spots=["RPM_ACTUAL"],
        )
