"""Planner agent: turns perception output + pallet state into an action."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from dr_ai_palletizer.agents.base import AgentMessage
from dr_ai_palletizer.clients.inference_client import InferenceClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the placement planner on a palletizing-robot team.
You receive structured perception observations for each box, the current
pallet state, and the pre-computed valid grid positions.

Output a single JSON action. Three legal shapes:

  PICK_AND_PLACE:
    {"action": "PICK_AND_PLACE", "box": <BOX_ID>, "target_pallet": 1|2,
     "position": [x, y, z], "speed_pct": <int 30-100>,
     "grip_strength": "gentle"|"standard"|"firm",
     "reason": <short>}

  CALL_A_HUMAN:
    {"action": "CALL_A_HUMAN", "boxes": [<BOX_ID>...], "reason": <short>}

  WAIT:
    {"action": "WAIT", "reason": <short>}

Rules: damaged/contaminated boxes go to CALL_A_HUMAN; heavy items (>=20kg)
go to the lowest free z; fragile items get gentle grip and lower speed_pct;
prefer the more-filled pallet. Output ONLY the JSON object."""


@dataclass
class PlannerAgent:
    """Text-only reasoner that picks the next action."""

    client: InferenceClient
    model: str
    role: str = "planner"

    async def run(self, *, task: str, context: list[AgentMessage]) -> AgentMessage:
        # Bake recent perception observations into the user turn so the
        # planner sees what the vision agent decided.
        observations = [msg for msg in context if msg.role == "perception"]
        obs_text = "\n".join(
            f"- box observation: {json.dumps(m.data, sort_keys=True)}" for m in observations
        )
        user_text = f"{task}\n\nPerception observations:\n{obs_text or '(none)'}\n"

        raw = await self.client.get_plan(
            system_prompt=_SYSTEM_PROMPT,
            scenario_text=user_text,
            model=self.model,
        )
        data = _parse_json(raw)
        return AgentMessage(role="planner", content=raw, data=data, model=self.model)


def _parse_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Planner agent returned non-JSON: %s", text[:200])
        return {}
