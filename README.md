---
title: OT Incident Defender
emoji: 🏭
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
tags:
  - openenv
---

# OT Incident Defender 🏭

> **OpenEnv Round 1 Submission** — ICS/OT Incident Response Environment

An AI agent plays the role of a SCADA operator responding to live cyberattacks and faults on an industrial plant. Every scenario is grounded in a documented real-world incident.

### Why This Environment?

Industrial Control System (ICS) cyberattacks are increasing — the attacks on Oldsmar's water supply (2021), the Ukrainian power grid blackouts (2015), and Stuxnet (2010) caused or nearly caused real physical harm. Today, security analysts must recognize attack patterns in noisy SCADA historian data and take the right corrective actions under time pressure. This environment lets agents practice that skill in a fully reproducible, physics-grounded simulation. It is the first OpenEnv environment targeting the **detection-response loop** in operational technology, an active area of need for CISA, ICS-CERT, and industrial automation vendors.

---

## Scenarios

| Task | Difficulty | Real-World Incident |
|------|-----------|---------------------|
| `task1` — Oldsmar Chemical Overdose | 🟢 Easy | Oldsmar FL water plant (Feb 2021) — TeamViewer hijack, NaOH 100× overdose |
| `task2` — Ukraine Grid Cascade Trip | 🟡 Medium | Ukraine power grid (Dec 2015) — BlackEnergy3 + KillDisk, 230k customers affected |
| `task3` — Stuxnet FDI Attack | 🔴 Hard | Stuxnet (2010) — false RPM readings while centrifuge over-speeds toward destruction |

---

## API Reference

### `POST /reset`
Start a new episode.
```json
{"task_id": "task1", "seed": 42}
```
Returns an `Observation` object.

### `POST /step`
Apply an action.
```json
{
  "action_type": "isolate_network_segment",
  "target": "SCADA_VLAN",
  "value": null,
  "justification": "Terminating attacker's TeamViewer session — NaOH setpoint at 11100 ppm"
}
```
Returns a `StepResult` with `observation`, `reward`, `done`, `info`.

### `GET /tasks`
Returns all task definitions.

### `GET /state`
Returns full internal environment state (including hidden variables for debugging).

### `GET /health`
Returns `{"status": "ok"}`.

---

## Action Space

| Action Type | Description |
|-------------|-------------|
| `acknowledge_alarm` | Acknowledge an active alarm (ISA-18.2) |
| `shelve_alarm` | Suppress an alarm temporarily |
| `send_setpoint` | Change a control setpoint |
| `open_breaker` | Open a circuit breaker |
| `close_breaker` | Close a circuit breaker |
| `isolate_network_segment` | Block network traffic to a segment |
| `revert_to_last_good` | Revert to last known-good configuration |
| `switch_to_manual` | Take manual control of a controller |
| `escalate_to_supervisor` | Escalate incident |
| `write_log_entry` | Write an incident log entry |
| `no_op` | Do nothing |

---

## Observation Space

Each observation contains:
- **`historian`** — last 20 tag readings (timestamp, tag, value, unit, source)
- **`alarms`** — active alarms with ISA-18.2 priority (P1–P4), state, `first_out` flag
- **`hmi_screens`** — screen availability (screen wiped by KillDisk → `available: false`)
- **`event_log`** — SOE / operator log (last 20 lines)
- **`network_log`** — firewall / remote access events
- **`safety_margin`** — `0.0` = breach imminent, `1.0` = fully safe
- **`blind_spots`** — tags deliberately hidden from agent

---

## Reward Function

| Signal | Value |
|--------|-------|
| Safety margin each step | +0.06 × margin |
| First correct action (decays) | +0.15 × (1 − step/max_steps) |
| Correct alarm acknowledgement | +0.05 |
| Wrong alarm acknowledgement | −0.10 |
| Shelving a real alarm | −0.12 |
| Correct setpoint | +0.08 × correctness |
| Network isolation (Task 1, 3) | +0.10 |
| Successful breaker close (Task 2) | +0.08 |
| Quality log entry | +0.06 × quality |
| Time pressure penalty | −0.005 × (step/max_steps) |
| Secondary fault caused | −0.20 |
| no_op at low safety (<0.5) | −0.08 |
| **Clip range** | [−0.5, 1.0] |

Final episode score is computed by a deterministic grader (no LLM-as-judge).

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn environment.server:app --host 0.0.0.0 --port 7860
```

### Quick smoke test
```bash
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1","seed":42}'

curl localhost:7860/tasks
curl localhost:7860/health
```

### Run tests
```bash
python -m pytest tests/ -v
```

---

## Running the Baseline Agent

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export HF_TOKEN=your_key_here
export MODEL_NAME=llama-3.3-70b-versatile
export ENV_URL=http://localhost:7860

python inference.py
```

Expected output format:
```
[START] task=task1 env=ot-incident-defender model=gpt-4o
[STEP] step=0 action=isolate_network_segment reward=0.19 done=false error=null
...
[END] success=true steps=8 score=0.412 rewards=0.19,0.17,...
```

---

## Grading Rubrics

### Task 1 (Oldsmar)
| Criterion | Points |
|-----------|--------|
| Setpoint corrected before step 8 | +0.30 |
| Network segment isolated | +0.20 |
| Root-cause alarm (first_out=True) acknowledged | +0.25 |
| Log entry containing "NaOH", "remote", and a number | +0.25 |

### Task 2 (Ukraine Grid)
| Criterion | Points |
|-----------|--------|
| Substation A identified as origin | +0.20 |
| Breakers restored A→B→C without secondary fault | +0.30 |
| Historian inference used for A after HMI dark | +0.20 |
| All load restored within 20 steps, no secondary trip | +0.30 |

### Task 3 (FDI)
| Criterion | Points |
|-----------|--------|
| Cross-correlation of vibration + RPM in justification | +0.20 |
| Manual control before breach step | +0.25 |
| Safe RPM setpoint (900–1100 Hz) | +0.20 |
| PLC network isolation | +0.15 |
| Complete 5-field incident log | +0.20 |

---

## Baseline Scores

Baseline agent: `llama-3.3-70b-versatile` via Groq API, `temperature=0.2`, `seed=42`.

| Task | Difficulty | Steps Used | Avg Step Reward | Grader Score |
|------|-----------|-----------|----------------|-------------|
| task1 — Oldsmar | Easy | 7–9 | 0.05–0.10 | 0.55–0.80 |
| task2 — Ukraine | Medium | 15–20 | 0.03–0.06 | 0.20–0.50 |
| task3 — FDI | Hard | 10–12 | 0.04–0.08 | 0.20–0.40 |

> Grader scores reflect the deterministic end-of-episode rubric (0.0–1.0). Step rewards reflect the continuous-signal reward function (clipped to [−0.5, 1.0]).

---

## Design Notes

- **Fully deterministic**: `reset(seed=42)` always produces byte-identical observations
- **No LLM-as-judge**: all graders use pure Python string matching
- **Physics-grounded**: each scenario implements first-principles plant dynamics
- **ISA-18.2 compliant**: alarm management follows the international standard
- **Security by design**: `RPM_ACTUAL` is internal state, never exposed in observations
