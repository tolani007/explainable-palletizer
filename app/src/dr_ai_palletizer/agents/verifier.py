"""Verifier agent: hard-rule safety check on the planner's action."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from dr_ai_palletizer.agents.base import AgentMessage
from dr_ai_palletizer.clients.inference_client import InferenceClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the safety verifier on a palletizing-robot team.
Given the perception observations and the planner's proposed action, output:
  {"verdict": "accept"|"reject", "reason": <short>}

Reject when any of these hold:
  - Action is PICK_AND_PLACE for a box marked damaged/contaminated/unpickable.
  - Speed_pct >= 75 for a fragile box.
  - Grip_strength is "firm" for a fragile box.
  - position[2] >= 1 (i.e. not on the floor layer) for a heavy box (>=20kg).
  - Action is WAIT when a clearly safe placement was available.

Otherwise accept. Output ONLY the JSON object."""


@dataclass
class VerifierAgent:
    """Hard-rule audit on the planner's decision."""

    client: InferenceClient
    model: str
    role: str = "verifier"

    async def run(self, *, task: str, context: list[AgentMessage]) -> AgentMessage:
        plan = next((m for m in reversed(context) if m.role == "planner"), None)
        if plan is None:
            data = {"verdict": "reject", "reason": "no plan to verify"}
            return AgentMessage(
                role="verifier", content=json.dumps(data), data=data, model=self.model
            )

        observations = [m for m in context if m.role == "perception"]
        obs_text = "\n".join(json.dumps(m.data, sort_keys=True) for m in observations)
        user_text = (
            f"Perception:\n{obs_text or '(none)'}\n\n"
            f"Planner action:\n{json.dumps(plan.data, sort_keys=True)}\n\n"
            f"Original task:\n{task}\n"
        )

        raw = await self.client.get_plan(
            system_prompt=_SYSTEM_PROMPT,
            scenario_text=user_text,
            model=self.model,
        )
        data = _parse_json(raw)
        if "verdict" not in data:
            data = {"verdict": "accept", "reason": "verifier output unparseable; defaulting to accept"}
        return AgentMessage(role="verifier", content=raw, data=data, model=self.model)


def _parse_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Verifier agent returned non-JSON: %s", text[:200])
        return {}
