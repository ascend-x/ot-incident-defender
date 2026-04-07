"""
inference.py — Baseline agent for OT Incident Defender using OpenAI-compatible client.

Mandatory for judging. Reads env vars:
  API_BASE_URL — LLM API base URL
  HF_TOKEN     — API key
  MODEL_NAME   — model to use
  ENV_URL      — environment URL (default: http://localhost:7860)

Produces [START] / [STEP] / [END] logs in the exact required format.
"""
import os
import textwrap
import json
from typing import Optional, List

from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ.get("HF_TOKEN") or os.environ["OPENAI_API_KEY"]
MODEL_NAME   = os.environ["MODEL_NAME"]
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 15
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ICS/SCADA operator responding to a live plant incident.
    Each turn you receive a JSON observation containing: historian tag readings,
    active alarms with ISA-18.2 priorities, HMI screen availability, event log,
    network log, safety_margin (0.0=breach imminent, 1.0=safe), and blind_spots.

    You must respond with a single valid JSON Action object with these fields:
      action_type: one of [acknowledge_alarm, shelve_alarm, send_setpoint,
                   open_breaker, close_breaker, isolate_network_segment,
                   revert_to_last_good, switch_to_manual,
                   escalate_to_supervisor, write_log_entry, no_op]
      target: the specific alarm_id, tag name, breaker_id, or segment (string or null)
      value: numeric value for setpoint changes (float or null)
      justification: your reasoning — be specific about which tags you used
                     and why. For write_log_entry, include: asset ID, attack vector,
                     detection method, action taken, recommended remediation.

    Respond with ONLY the JSON object. No markdown, no explanation outside the JSON.
    Prioritize safety_margin above all else. When sensors contradict each other,
    cross-correlate before acting.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(client: OpenAI, obs_json: str, history: List[str]) -> dict:
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = (
        f"Current observation:\n{obs_json}\n\n"
        f"Previous actions:\n{history_block}\n\n"
        f"Respond with your Action JSON."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {
            "action_type": "no_op",
            "target": None,
            "value": None,
            "justification": "fallback"
        }


def run_task(client: OpenAI, task_id: str) -> None:
    import requests

    log_start(task=task_id, env="ot-incident-defender", model=MODEL_NAME)

    obs = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": 42}
    ).json()

    rewards: List[float] = []
    history: List[str] = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        action_dict = get_action(client, json.dumps(obs), history)

        result = requests.post(f"{ENV_URL}/step", json=action_dict).json()
        reward = result["reward"]
        done   = result["done"]

        rewards.append(reward)
        history.append(
            f"step={step} action={action_dict.get('action_type')} reward={reward:.2f}"
        )

        log_step(
            step=step,
            action=action_dict.get("action_type", "no_op"),
            reward=reward,
            done=done,
            error=None,
        )

        obs = result["observation"]
        info = result["info"]
        step += 1

    # Use the grader_score from the final step's info
    score = info.get("grader_score", 0.0)
    log_end(success=done, steps=step, score=score, rewards=rewards)


if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in ["task1", "task2", "task3"]:
        run_task(client, task)
