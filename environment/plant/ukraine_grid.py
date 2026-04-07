"""
ukraine_grid.py — Ukraine Power Grid attack physics (Scenario 2).

Real-world incident: December 2015 — BlackEnergy3 malware + Industroyer
operators opened each substation's breakers remotely, then wiped HMI systems
with KillDisk, denying operators visibility.

Simulation:
  - 3 substations: A, B, C — each with a breaker (open/closed) and MW load.
  - Attacker opens breaker A at step 2, B at step 4, C at step 6.
  - At step 4, HMI for substation A is wiped (available=False).
  - Closing the wrong breaker under load can cause a secondary fault.
  - safety_margin = closed_breakers / 3.
  - Episode ends at step 20.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from typing import Any

from environment.models import (
    Observation, Action, HistorianEntry, AlarmRecord, HMIScreen
)
from environment.plant.base import BasePlant

# Substations
SUBSTATIONS = ["A", "B", "C"]
LOADS_MW = {"A": 150.0, "B": 120.0, "C": 90.0}
MAX_STEPS = 20

# Attacker schedule: breaker open at step N
ATTACKER_SCHEDULE = {2: "A", 4: "B", 6: "C"}

_PLANT_EPOCH = datetime(2015, 12, 23, 15, 30, 0, tzinfo=timezone(timedelta(hours=2)))


class UkraineGridPlant(BasePlant):
    """Ukraine 2015 power grid substation attack physics."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)
        self.breakers: dict[str, bool] = {}      # True = closed
        self.loads_mw: dict[str, float] = {}
        self.hmi_available: dict[str, bool] = {}
        self._historian: list[HistorianEntry] = []
        self._event_log: list[str] = []
        self._net_log: list[str] = []
        self._alarms: dict[str, AlarmRecord] = {}
        self._step_count = 0
        self._secondary_fault_occurred = False
        self._sequence_errors: list[str] = []

    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)
        self.breakers = {s: True for s in SUBSTATIONS}
        self.loads_mw = {k: v for k, v in LOADS_MW.items()}
        self.hmi_available = {s: True for s in SUBSTATIONS}
        self._historian = []
        self._event_log = []
        self._net_log = [
            "[15:28:00] VPN connection from 91.108.4.x -> ENERGY_CORP_VPNGW",
            "[15:29:30] Workstation SCADA-WS03 login: dispatch_user",
            "[15:29:55] BlackEnergy3 lateral movement detected on SCADA-WS03",
        ]
        self._alarms = {}
        self._step_count = 0
        self._secondary_fault_occurred = False
        self._sequence_errors = []

        self._record_historian()
        self._event_log.append("[15:30:00] Control room shift started")
        self._event_log.append("[15:30:01] All substations nominal — loads normal")

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        self._step_count += 1
        t = self._step_count

        # Attacker action at scheduled step
        attacker_target = ATTACKER_SCHEDULE.get(t)
        if attacker_target and self.breakers.get(attacker_target, False):
            self.breakers[attacker_target] = False
            self.loads_mw[attacker_target] = 0.0
            self._event_log.append(
                f"[{self._plant_time(t)}] BREAKER {attacker_target} OPENED by remote command"
            )
            self._raise_alarm(
                f"ALM-BRK-{attacker_target}",
                f"BREAKER_{attacker_target}",
                "P1",
                first_out=(attacker_target == "A"),
                message=f"Substation {attacker_target}: breaker opened unexpectedly — loss of {LOADS_MW[attacker_target]:.0f} MW"
            )

        # KillDisk wipe at step 4
        if t == 4:
            self.hmi_available["A"] = False
            self._event_log.append(
                f"[{self._plant_time(t)}] HMI workstation for SUBSTATION_A wiped — KillDisk executed"
            )
            self._raise_alarm(
                "ALM-HMI-A", "SUBSTATION_A_HMI", "P2",
                first_out=False,
                message="Substation A HMI screen unavailable — disk wipe detected"
            )
            self._net_log.append(
                f"[{self._plant_time(t)}] KillDisk process started on SCADA-WS03"
            )

        # Apply agent action
        secondary_fault = self._apply_action(action, t)

        # Update historian
        self._record_historian()

        # Update net log occasionally
        if t % 2 == 0:
            self._net_log.append(
                f"[{self._plant_time(t)}] Firewall: anomalous traffic on SCADA VLAN"
            )

        closed_count = sum(1 for v in self.breakers.values() if v)
        all_restored = closed_count == 3
        safety_margin = round(closed_count / 3.0, 4)

        done = (t >= MAX_STEPS)
        info = {
            "breakers": dict(self.breakers),
            "loads_mw": dict(self.loads_mw),
            "secondary_fault": secondary_fault,
            "secondary_fault_ever": self._secondary_fault_occurred,
            "secondary_fault_occurred": self._secondary_fault_occurred,
            "all_restored": all_restored,
            "sequence_errors": list(self._sequence_errors),
            "step": t,
        }

        return self._make_observation(), safety_margin, done, info

    def get_state(self) -> dict[str, Any]:
        closed = sum(1 for v in self.breakers.values() if v)
        return {
            "breakers": dict(self.breakers),
            "loads_mw": dict(self.loads_mw),
            "hmi_available": dict(self.hmi_available),
            "safety_margin": closed / 3.0,
            "secondary_fault_occurred": self._secondary_fault_occurred,
            "step": self._step_count,
            "max_steps": MAX_STEPS,
        }

    # ------------------------------------------------------------------ #
    def _apply_action(self, action: Action, t: int) -> bool:
        """Returns True if action caused a secondary fault."""
        secondary = False

        if action.action_type == "close_breaker":
            target = (action.target or "").replace("BREAKER_", "").upper()
            if target not in SUBSTATIONS:
                self._event_log.append(f"[{self._plant_time(t)}] Invalid breaker target: {action.target}")
                return False

            if not self.breakers[target]:
                # Check if closing into dead section (incorrect sequence)
                # Safe order is A → B → C; closing B before A, or C before B, causes fault
                wrong_order = self._is_wrong_order(target)
                if wrong_order:
                    self._secondary_fault_occurred = True
                    self._sequence_errors.append(f"step={t} closed {target} out of order")
                    secondary = True
                    # Secondary fault: the breaker closes but trips immediately
                    self._event_log.append(
                        f"[{self._plant_time(t)}] SECONDARY TRIP: Breaker {target} fault — "
                        f"closed into dead section"
                    )
                    self._raise_alarm(
                        f"ALM-SEC-{target}", f"BREAKER_{target}", "P1",
                        first_out=False,
                        message=f"Substation {target}: Secondary fault — breaker closed out of sequence"
                    )
                    # Breaker trips immediately back open
                else:
                    self.breakers[target] = True
                    self.loads_mw[target] = LOADS_MW[target] + self._rng.gauss(0, 2.0)
                    self._event_log.append(
                        f"[{self._plant_time(t)}] Breaker {target} CLOSED — "
                        f"restoring {self.loads_mw[target]:.1f} MW"
                    )
                    # Clear the breaker alarm
                    alm_id = f"ALM-BRK-{target}"
                    if alm_id in self._alarms:
                        self._alarms[alm_id] = self._alarms[alm_id].model_copy(
                            update={"state": "cleared"}
                        )
            else:
                self._event_log.append(
                    f"[{self._plant_time(t)}] Breaker {target} already closed — no action"
                )

        elif action.action_type == "open_breaker":
            target = (action.target or "").replace("BREAKER_", "").upper()
            if target in SUBSTATIONS and self.breakers.get(target, False):
                self.breakers[target] = False
                self.loads_mw[target] = 0.0
                self._event_log.append(f"[{self._plant_time(t)}] Operator opened breaker {target}")

        elif action.action_type == "acknowledge_alarm":
            aid = action.target
            if aid in self._alarms:
                self._alarms[aid] = self._alarms[aid].model_copy(update={"state": "acked"})
                self._event_log.append(f"[{self._plant_time(t)}] Alarm {aid} acknowledged")

        elif action.action_type == "isolate_network_segment":
            self._net_log.append(
                f"[{self._plant_time(t)}] Network segment {action.target} isolated by operator"
            )
            self._event_log.append(
                f"[{self._plant_time(t)}] Attacker VPN connection terminated"
            )

        elif action.action_type == "write_log_entry":
            self._event_log.append(
                f"[{self._plant_time(t)}] LOG: {action.justification[:200]}"
            )

        elif action.action_type == "escalate_to_supervisor":
            self._event_log.append(f"[{self._plant_time(t)}] Escalated to supervisor")

        elif action.action_type == "revert_to_last_good":
            self._event_log.append(
                f"[{self._plant_time(t)}] Attempting revert to last known good configuration"
            )

        elif action.action_type == "switch_to_manual":
            self._event_log.append(
                f"[{self._plant_time(t)}] Switched substation {action.target} to manual control"
            )

        return secondary

    def _is_wrong_order(self, target: str) -> bool:
        """
        Correct restoration order is A → B → C.
        A can always be closed (it's the main feeder).
        B can only be closed if A is closed (energised).
        C can only be closed if B is closed (energised).
        Closing into an open/dead upstream breaker causes a secondary fault.
        """
        if target == "A":
            return False   # always safe to restore A first
        if target == "B":
            # B wrong if A is currently open (dead section above B)
            return not self.breakers["A"]
        if target == "C":
            # C wrong if B is currently open
            return not self.breakers["B"]
        return False

    def _raise_alarm(
        self, alarm_id: str, tag: str, priority: str,
        first_out: bool, message: str
    ) -> None:
        self._alarms[alarm_id] = AlarmRecord(
            alarm_id=alarm_id, tag=tag,
            priority=priority,  # type: ignore[arg-type]
            state="unacked", first_out=first_out, message=message
        )

    def _record_historian(self) -> None:
        ts = self._plant_time(self._step_count)
        for sub in SUBSTATIONS:
            self._historian.append(HistorianEntry(
                timestamp=ts,
                tag=f"BREAKER_{sub}_CLOSED",
                value=1.0 if self.breakers[sub] else 0.0,
                unit="bool", source="sensor"
            ))
            self._historian.append(HistorianEntry(
                timestamp=ts,
                tag=f"LOAD_{sub}_MW",
                value=round(self.loads_mw[sub], 2),
                unit="MW", source="sensor"
            ))

    def _plant_time(self, step: int) -> str:
        dt = _PLANT_EPOCH + timedelta(minutes=step * 3)
        return dt.strftime("%H:%M:%S")

    def _make_observation(self) -> Observation:
        historian_tail = self._historian[-20:]
        closed_count = sum(1 for v in self.breakers.values() if v)
        safety_margin = round(closed_count / 3.0, 4)

        hmi_screens = [
            HMIScreen(
                screen_id=f"SUBSTATION_{s}",
                available=self.hmi_available[s],
                tags_visible=[f"BREAKER_{s}_CLOSED", f"LOAD_{s}_MW"]
                if self.hmi_available[s] else []
            )
            for s in SUBSTATIONS
        ]

        return Observation(
            step=self._step_count,
            scenario_id="task2_ukraine",
            plant_time=self._plant_time(self._step_count),
            historian=historian_tail,
            alarms=list(self._alarms.values()),
            hmi_screens=hmi_screens,
            event_log=list(self._event_log[-20:]),
            network_log=list(self._net_log[-20:]),
            safety_margin=safety_margin,
            blind_spots=["SUBSTATION_A"] if not self.hmi_available["A"] else [],
        )
