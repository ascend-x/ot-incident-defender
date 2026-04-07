"""
oldsmar.py — Oldsmar Water Treatment Plant physics (Scenario 1).

Real-world incident: February 2021, Oldsmar FL — attacker used TeamViewer to
remotely access the SCADA system and changed the sodium hydroxide (NaOH)
setpoint from 111 ppm to 11,100 ppm. An alert operator caught the change.

Simulation:
  - NaOH_DOSE_PPM: first-order dosing system. setpoint hijacked to 11100.
  - REMOTE_SESSION_ACTIVE: True from step 0; cleared only by isolate_network_segment.
  - FLOW_RATE_LPM: steady-state, slight noise.
  - safety_margin = max(0.0, 1.0 - (naoh_ppm / 11100))
  - Episode ends at step 15 or when naoh_ppm > 10000 (breach).
"""
from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from typing import Any

from environment.models import (
    Observation, Action, HistorianEntry, AlarmRecord, HMIScreen
)
from environment.plant.base import BasePlant

# Plant constants
SAFE_NAOH_PPM    = 111.0
ATTACK_SETPOINT  = 11_100.0
FLOW_NOMINAL_LPM = 3_785.0   # ~1 MGD
MAX_STEPS        = 15
BREACH_THRESHOLD = 10_000.0
FIRST_ORDER_TAU  = 0.3       # fraction per step

# Epoch anchor – Oldsmar incident date in plant-local EST
_PLANT_EPOCH = datetime(2021, 2, 5, 14, 23, 0, tzinfo=timezone(timedelta(hours=-5)))


class OldsmarPlant(BasePlant):
    """Oldsmar chemical dosing plant physics."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)
        # Mutable state
        self.naoh_ppm: float = SAFE_NAOH_PPM
        self.setpoint: float = SAFE_NAOH_PPM
        self.remote_active: bool = True
        self.network_isolated: bool = False
        self.flow_rate: float = FLOW_NOMINAL_LPM
        self._historian: list[HistorianEntry] = []
        self._event_log: list[str] = []
        self._alarms: dict[str, AlarmRecord] = {}
        self._step_count: int = 0
        self._manual_setpoint: float | None = None  # agent override

    # ------------------------------------------------------------------ #
    #  BasePlant interface                                                 #
    # ------------------------------------------------------------------ #

    def reset(self) -> Observation:
        """Fully deterministic reset."""
        self._rng = random.Random(self.seed)
        self.naoh_ppm = SAFE_NAOH_PPM
        self.setpoint = SAFE_NAOH_PPM
        self.remote_active = True
        self.network_isolated = False
        self.flow_rate = FLOW_NOMINAL_LPM
        self._historian = []
        self._event_log = []
        self._alarms = {}
        self._step_count = 0
        self._manual_setpoint = None

        # Inject attack at step 0 — attacker sets the setpoint via remote session
        self.setpoint = ATTACK_SETPOINT

        # Record the attack in the network log
        self._net_log: list[str] = [
            "[14:23:01] TeamViewer session opened from 185.220.101.34 -> SCADA_WS01",
        ]

        # Create initial historian entries (pre-step baseline)
        self._record_historian()

        # Create alarm: NaOH setpoint deviation
        self._raise_alarm(
            alarm_id="ALM-NaOH-01",
            tag="NaOH_DOSE_PPM",
            priority="P1",
            first_out=True,
            message="NaOH setpoint deviation: commanded 11100 ppm (safe limit 111 ppm)"
        )
        self._raise_alarm(
            alarm_id="ALM-FLOW-01",
            tag="FLOW_RATE_LPM",
            priority="P3",
            first_out=False,
            message="Flow rate within normal band — nuisance alarm during shift change"
        )

        self._event_log.append("[14:23:00] Operator workstation login: ops_user1")
        self._event_log.append("[14:23:01] Remote session detected on SCADA_WS01")
        self._event_log.append("[14:23:05] NaOH setpoint modified via HMI tag write")

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        self._step_count += 1

        # --- Apply action side-effects ---
        info: dict[str, Any] = {
            "naoh_ppm": self.naoh_ppm,
            "setpoint": self.setpoint,
            "remote_active": self.remote_active,
            "network_isolated": self.network_isolated,
        }
        self._apply_action(action)

        # --- Physics step ---
        # If attacker still active (network not isolated), setpoint stays at attack value.
        effective_setpoint = self.setpoint
        if self.remote_active and not self.network_isolated:
            effective_setpoint = ATTACK_SETPOINT

        # First-order system: x(t+1) = x(t) + (setpoint - x(t)) * tau
        noise = self._rng.gauss(0.0, 0.5)
        self.naoh_ppm += (effective_setpoint - self.naoh_ppm) * FIRST_ORDER_TAU + noise
        self.naoh_ppm = max(0.0, self.naoh_ppm)

        # Flow rate slight variation
        self.flow_rate = FLOW_NOMINAL_LPM + self._rng.gauss(0.0, 10.0)

        # Update historian
        self._record_historian()

        # Update network log
        if self._step_count % 3 == 0 and self.remote_active and not self.network_isolated:
            self._net_log.append(
                f"[{self._plant_time(self._step_count)}] TeamViewer keepalive from 185.220.101.34"
            )

        # Check breach / done
        breach = self.naoh_ppm > BREACH_THRESHOLD
        done = breach or (self._step_count >= MAX_STEPS)

        safety_margin = max(0.0, 1.0 - (self.naoh_ppm / ATTACK_SETPOINT))
        if breach:
            self._event_log.append(
                f"[{self._plant_time(self._step_count)}] BREACH: NaOH level exceeded 10000 ppm — SAFETY SHUTDOWN"
            )

        info.update({
            "naoh_ppm_after": self.naoh_ppm,
            "breach": breach,
            "effective_setpoint": effective_setpoint,
        })

        obs = self._make_observation()
        return obs, safety_margin, done, info

    def get_state(self) -> dict[str, Any]:
        return {
            "naoh_ppm": self.naoh_ppm,
            "setpoint": self.setpoint,
            "remote_active": self.remote_active,
            "network_isolated": self.network_isolated,
            "flow_rate": self.flow_rate,
            "step": self._step_count,
            "max_steps": MAX_STEPS,
            "safety_margin": max(0.0, 1.0 - (self.naoh_ppm / ATTACK_SETPOINT)),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _apply_action(self, action: Action) -> None:
        t = self._step_count

        if action.action_type == "isolate_network_segment":
            # Any segment isolation terminates remote session
            self.network_isolated = True
            self.remote_active = False
            # Once isolated, attacker can't push setpoint changes — revert if not already done
            if self._manual_setpoint is None:
                self.setpoint = SAFE_NAOH_PPM
            self._event_log.append(
                f"[{self._plant_time(t)}] Network segment isolated by operator"
            )
            self._net_log.append(
                f"[{self._plant_time(t)}] TeamViewer session terminated"
            )

        elif action.action_type == "send_setpoint":
            if action.target == "NaOH_DOSE_PPM" and action.value is not None:
                self._manual_setpoint = float(action.value)
                self.setpoint = float(action.value)
                self._event_log.append(
                    f"[{self._plant_time(t)}] Setpoint NaOH_DOSE_PPM <- {action.value} ppm"
                )

        elif action.action_type == "revert_to_last_good":
            self._manual_setpoint = SAFE_NAOH_PPM
            self.setpoint = SAFE_NAOH_PPM
            self._event_log.append(
                f"[{self._plant_time(t)}] Reverted NaOH setpoint to last good value ({SAFE_NAOH_PPM} ppm)"
            )

        elif action.action_type == "acknowledge_alarm":
            alarm_id = action.target
            if alarm_id in self._alarms:
                self._alarms[alarm_id] = self._alarms[alarm_id].model_copy(
                    update={"state": "acked"}
                )
                self._event_log.append(
                    f"[{self._plant_time(t)}] Alarm {alarm_id} acknowledged by operator"
                )

        elif action.action_type == "shelve_alarm":
            alarm_id = action.target
            if alarm_id in self._alarms:
                self._alarms[alarm_id] = self._alarms[alarm_id].model_copy(
                    update={"state": "shelved"}
                )
                self._event_log.append(
                    f"[{self._plant_time(t)}] Alarm {alarm_id} shelved by operator"
                )

        elif action.action_type == "write_log_entry":
            self._event_log.append(
                f"[{self._plant_time(t)}] LOG: {action.justification[:200]}"
            )

        elif action.action_type == "switch_to_manual":
            self._event_log.append(
                f"[{self._plant_time(t)}] Switched to manual control"
            )

        elif action.action_type == "escalate_to_supervisor":
            self._event_log.append(
                f"[{self._plant_time(t)}] Escalated to supervisor"
            )

    def _raise_alarm(
        self, alarm_id: str, tag: str, priority: str,
        first_out: bool, message: str
    ) -> None:
        self._alarms[alarm_id] = AlarmRecord(
            alarm_id=alarm_id,
            tag=tag,
            priority=priority,  # type: ignore[arg-type]
            state="unacked",
            first_out=first_out,
            message=message,
        )

    def _record_historian(self) -> None:
        ts = self._plant_time(self._step_count)
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="NaOH_DOSE_PPM",
            value=round(self.naoh_ppm, 2), unit="ppm", source="sensor"
        ))
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="NaOH_SETPOINT_PPM",
            value=round(self.setpoint, 2), unit="ppm", source="controller"
        ))
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="FLOW_RATE_LPM",
            value=round(self.flow_rate, 2), unit="L/min", source="sensor"
        ))
        self._historian.append(HistorianEntry(
            timestamp=ts, tag="REMOTE_SESSION_ACTIVE",
            value=1.0 if self.remote_active else 0.0,
            unit="bool", source="sensor"
        ))

    def _plant_time(self, step: int) -> str:
        dt = _PLANT_EPOCH + timedelta(minutes=step * 2)
        return dt.strftime("%H:%M:%S")

    def _make_observation(self) -> Observation:
        historian_tail = self._historian[-20:]
        safety = max(0.0, 1.0 - (self.naoh_ppm / ATTACK_SETPOINT))

        return Observation(
            step=self._step_count,
            scenario_id="task1_oldsmar",
            plant_time=self._plant_time(self._step_count),
            historian=historian_tail,
            alarms=list(self._alarms.values()),
            hmi_screens=[
                HMIScreen(
                    screen_id="DOSING_CONTROL",
                    available=True,
                    tags_visible=["NaOH_DOSE_PPM", "NaOH_SETPOINT_PPM", "FLOW_RATE_LPM"]
                ),
                HMIScreen(
                    screen_id="NETWORK_STATUS",
                    available=True,
                    tags_visible=["REMOTE_SESSION_ACTIVE"]
                ),
            ],
            event_log=list(self._event_log[-20:]),
            network_log=list(self._net_log[-20:]),
            safety_margin=round(safety, 4),
            blind_spots=[],
        )
