"""Perception agent: reads a box image, returns structured observations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from dr_ai_palletizer.agents.base import AgentMessage
from dr_ai_palletizer.clients.inference_client import InferenceClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the perception specialist on a palletizing-robot team.
You receive ONE box image plus its measured weight and dimensions.
Output a JSON object with fields:
  - condition: "ok" | "damaged" | "contaminated" | "unpickable"
  - fragility: "robust" | "standard" | "fragile"
  - contents_guess: short string, your best guess at what's inside
  - notes: short string, anything else relevant for safe handling
Output ONLY the JSON object. No markdown fence, no commentary."""


@dataclass
class PerceptionAgent:
    """Vision-language agent that classifies a box for the planner."""

    client: InferenceClient
    model: str
    role: str = "perception"

    async def run(self, *, task: str, context: list[AgentMessage]) -> AgentMessage:
        """Classify a single box.

        `task` is a short text string with measured weight/dims; `context`
        carries the image bytes via `data["image_bytes"]` on the most recent
        non-empty entry, or empty for a text-only fallback.
        """
        image_bytes: list[bytes] = []
        for msg in reversed(context):
            if msg.data.get("image_bytes"):
                image_bytes = [msg.data["image_bytes"]]
                break

        if image_bytes:
            raw = await self.client.get_action(
                system_prompt=_SYSTEM_PROMPT,
                images=image_bytes,
                scenario_text=task,
                model=self.model,
                max_tokens=256,
            )
        else:
            raw = await self.client.get_plan(
                system_prompt=_SYSTEM_PROMPT,
                scenario_text=task,
                model=self.model,
            )

        data = _parse_json(raw)
        return AgentMessage(role="perception", content=raw, data=data, model=self.model)


def _parse_json(raw: str) -> dict:
    """Strip <think> tags and pull the first JSON object out of the text."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Perception agent returned non-JSON: %s", text[:200])
        return {}
