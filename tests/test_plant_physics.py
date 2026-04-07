from __future__ import annotations

import json
import pytest

from environment.models import Action
from environment.plant.oldsmar import OldsmarPlant, SAFE_NAOH_PPM, ATTACK_SETPOINT, BREACH_THRESHOLD
from environment.plant.ukraine_grid import UkraineGridPlant
from environment.plant.fdi_centrifuge import FDICentrifugePlant, RPM_INITIAL, RPM_BREACH


def _noop(justification: str = "test") -> Action:
    return Action(action_type="no_op", justification=justification)


class TestOldsmarPlant:

    def test_initial_naoh_is_safe(self):
        plant = OldsmarPlant(seed=42)
        obs = plant.reset()
        naoh_entries = [e for e in obs.historian if e.tag == "NaOH_DOSE_PPM"]
        assert naoh_entries, "NaOH_DOSE_PPM missing from historian"
        assert naoh_entries[-1].value == pytest.approx(SAFE_NAOH_PPM, abs=1.0)

    def test_network_log_at_step_0(self):
        plant = OldsmarPlant(seed=42)
        obs = plant.reset()
        assert any("TeamViewer" in line for line in obs.network_log), \
            "TeamViewer session not in network_log at step 0"
        assert any("185.220" in line for line in obs.network_log)

    def test_naoh_rises_without_action(self):
        plant = OldsmarPlant(seed=42)
        plant.reset()
        prev_naoh = SAFE_NAOH_PPM
        for _ in range(5):
            obs, _, done, _ = plant.step(_noop())
            curr = [e for e in obs.historian if e.tag == "NaOH_DOSE_PPM"][-1].value
            assert curr > prev_naoh or done, "NaOH should rise toward attacker setpoint"
            prev_naoh = curr

    def test_isolation_stops_attack(self):
        plant = OldsmarPlant(seed=42)
        plant.reset()
        isolate = Action(
            action_type="isolate_network_segment",
            target="SCADA_VLAN",
            justification="Blocking attacker"
        )
        plant.step(isolate)
        obs_after, _, _, _ = plant.step(_noop())
        sp = [e for e in obs_after.historian if e.tag == "NaOH_SETPOINT_PPM"]
        assert sp, "NaOH_SETPOINT_PPM not in historian"
        assert sp[-1].value <= 200.0, f"Setpoint still high after isolation: {sp[-1].value}"

    def test_breach_at_high_naoh(self):
        plant = OldsmarPlant(seed=42)
        plant.reset()
        done = False
        for _ in range(15):
            if done:
                break
            _, _, done, _ = plant.step(_noop())
        assert done, "Episode should be done by max step 15"

    def test_safety_margin_continuous(self):
        plant = OldsmarPlant(seed=42)
        plant.reset()
        margins = []
        for _ in range(8):
            obs, _, done, _ = plant.step(_noop())
            margins.append(obs.safety_margin)
            if done:
                break
        assert all(0.0 <= m <= 1.0 for m in margins), f"Margin out of range: {margins}"
        assert len(set(margins)) > 1, "Safety margin is not changing (not continuous)"

    def test_determinism_full_episode(self):
        def run_episode():
            p = OldsmarPlant(seed=42)
            p.reset()
            obs_list = []
            for _ in range(5):
                obs, _, done, _ = p.step(_noop())
                obs_list.append(obs.model_dump_json())
                if done:
                    break
            return obs_list

        run1 = run_episode()
        run2 = run_episode()
        assert run1 == run2, "Oldsmar plant is not deterministic!"

    def test_alarms_contain_first_out(self):
        plant = OldsmarPlant(seed=42)
        obs = plant.reset()
        assert any(a.first_out for a in obs.alarms), "No first_out=True alarm at reset"


class TestUkraineGridPlant:

    def test_all_breakers_closed_at_reset(self):
        plant = UkraineGridPlant(seed=42)
        obs = plant.reset()
        for sub in ["A", "B", "C"]:
            entries = [e for e in obs.historian if e.tag == f"BREAKER_{sub}_CLOSED"]
            assert entries, f"BREAKER_{sub}_CLOSED not in historian"
            assert entries[-1].value == 1.0, f"Breaker {sub} not closed at reset"

    def test_attacker_opens_breaker_a_at_step_2(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        for step in range(1, 3):
            obs, _, _, _ = plant.step(_noop())
        a_entries = [e for e in obs.historian if e.tag == "BREAKER_A_CLOSED"]
        assert a_entries[-1].value == 0.0, "Breaker A should be open at step 2"

    def test_hmi_wipe_at_step_4(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        for _ in range(4):
            obs, _, _, _ = plant.step(_noop())
        hmi_a = next((s for s in obs.hmi_screens if s.screen_id == "SUBSTATION_A"), None)
        assert hmi_a is not None, "SUBSTATION_A screen not in observation"
        assert not hmi_a.available, "SUBSTATION_A HMI should be unavailable after step 4"

    def test_hmi_stays_dark_permanently(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        for _ in range(10):
            obs, _, done, _ = plant.step(_noop())
            hmi_a = next(s for s in obs.hmi_screens if s.screen_id == "SUBSTATION_A")
            if obs.step >= 4:
                assert not hmi_a.available, f"SUBSTATION_A came back online at step {obs.step}"
            if done:
                break

    def test_secondary_fault_out_of_order(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        plant.step(_noop())
        plant.step(_noop())
        plant.step(_noop())
        plant.step(_noop())
        close_b = Action(
            action_type="close_breaker",
            target="BREAKER_B",
            justification="Closing B before A — wrong order"
        )
        _, _, _, info = plant.step(close_b)
        assert info.get("secondary_fault"), \
            f"Expected secondary fault when closing B (open) before A (open). Info: {info}"

    def test_correct_order_no_secondary_fault(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        for _ in range(7):
            plant.step(_noop())

        for sub in ["A", "B", "C"]:
            act = Action(action_type="close_breaker", target=f"BREAKER_{sub}",
                        justification=f"Closing {sub}")
            _, _, _, info = plant.step(act)
            assert not info.get("secondary_fault"), f"Secondary fault when closing {sub}"

    def test_safety_margin_range(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        for _ in range(20):
            obs, _, done, _ = plant.step(_noop())
            assert 0.0 <= obs.safety_margin <= 1.0
            if done:
                break

    def test_determinism(self):
        def run(steps=7):
            p = UkraineGridPlant(seed=42)
            p.reset()
            return [p.step(_noop())[0].model_dump_json() for _ in range(steps)]

        assert run() == run(), "UkraineGrid plant is not deterministic!"

    def test_max_steps_is_20(self):
        plant = UkraineGridPlant(seed=42)
        plant.reset()
        done = False
        for i in range(21):
            _, _, done, info = plant.step(_noop())
            if done:
                assert info["step"] <= 20
                break
        assert done, "Episode did not terminate at/before step 20"


class TestFDICentrifugePlant:

    def test_rpm_reported_frozen(self):
        plant = FDICentrifugePlant(seed=42)
        plant.reset()
        for _ in range(10):
            obs, _, done, _ = plant.step(_noop())
            rpm = [e for e in obs.historian if e.tag == "RPM_REPORTED"]
            assert rpm, "RPM_REPORTED not in historian"
            assert rpm[-1].value == pytest.approx(RPM_INITIAL, abs=1e-3), \
                f"RPM_REPORTED != 1064 at step {obs.step}"
            if done:
                break

    def test_rpm_actual_never_in_observation(self):
        plant = FDICentrifugePlant(seed=42)
        obs = plant.reset()
        all_tags = {e.tag for e in obs.historian}
        assert "RPM_ACTUAL" not in all_tags, "RPM_ACTUAL in initial observation!"

        for _ in range(12):
            obs, _, done, _ = plant.step(_noop())
            all_tags = {e.tag for e in obs.historian}
            assert "RPM_ACTUAL" not in all_tags, f"RPM_ACTUAL leaked at step {obs.step}"
            if done:
                break

    def test_vibration_rises_with_steps(self):
        plant = FDICentrifugePlant(seed=42)
        plant.reset()
        prev_vib = None
        for _ in range(6):
            obs, _, done, _ = plant.step(_noop())
            vib = [e for e in obs.historian if e.tag == "VIBRATION_G"][-1].value
            if prev_vib is not None:
                assert vib >= prev_vib - 0.02, "Vibration not rising monotonically"
            prev_vib = vib
            if done:
                break

    def test_breach_within_episode(self):
        plant = FDICentrifugePlant(seed=42)
        plant.reset()
        done = False
        for i in range(15):
            _, _, done, info = plant.step(_noop())
            if done and info.get("breach"):
                break
        assert done, "Plant should terminate due to breach within 13 steps"

    def test_manual_control_prevents_breach(self):
        plant = FDICentrifugePlant(seed=42)
        plant.reset()
        manual = Action(action_type="switch_to_manual",
                       justification="Preventing breach")
        setpt = Action(action_type="send_setpoint", target="RPM_SETPOINT",
                      value=1000.0, justification="Safe setpoint")
        plant.step(manual)
        plant.step(setpt)
        for _ in range(10):
            _, _, done, info = plant.step(_noop())
            if done:
                assert not info.get("breach"), "Should not breach with manual+safe setpoint"
                break

    def test_safety_margin_1_at_reset(self):
        plant = FDICentrifugePlant(seed=42)
        obs = plant.reset()
        assert obs.safety_margin == pytest.approx(1.0, abs=0.05)

    def test_safety_margin_decreases(self):
        plant = FDICentrifugePlant(seed=42)
        plant.reset()
        margins = []
        for _ in range(6):
            obs, _, done, _ = plant.step(_noop())
            margins.append(obs.safety_margin)
            if done:
                break
        assert margins[-1] < margins[0], "Safety margin should decrease over time without intervention"

    def test_determinism(self):
        def run():
            p = FDICentrifugePlant(seed=42)
            p.reset()
            return [p.step(_noop())[0].model_dump_json() for _ in range(6)]

        assert run() == run(), "FDI centrifuge plant is not deterministic!"

    def test_blind_spots_contains_rpm_actual(self):
        plant = FDICentrifugePlant(seed=42)
        obs = plant.reset()
        assert "RPM_ACTUAL" in obs.blind_spots

        for _ in range(5):
            obs, _, done, _ = plant.step(_noop())
            assert "RPM_ACTUAL" in obs.blind_spots
            if done:
                break

    def test_max_steps_is_12(self):
        plant = FDICentrifugePlant(seed=42)
        plant.reset()
        done = False
        for i in range(15):
            _, _, done, info = plant.step(_noop())
            if done:
                assert info["step"] <= 12
                break
        assert done


@pytest.mark.parametrize("PlantClass", [OldsmarPlant, UkraineGridPlant, FDICentrifugePlant])
def test_reset_determinism(PlantClass):
    def get_first_obs():
        p = PlantClass(seed=42)
        obs = p.reset()
        return obs.model_dump_json()

    results = [get_first_obs() for _ in range(5)]
    assert len(set(results)) == 1, f"{PlantClass.__name__} reset not deterministic!"
