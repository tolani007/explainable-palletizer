"""Tests for the Conductor multi-agent team."""

from __future__ import annotations

import pytest

from dr_ai_palletizer.agents import (
    Conductor,
    PerceptionAgent,
    PlannerAgent,
    VerifierAgent,
)
from dr_ai_palletizer.agents.base import AgentMessage


class _StubClient:
    """Minimal stand-in for InferenceClient. Returns scripted text per model."""

    def __init__(self, by_model: dict[str, list[str]]) -> None:
        self._scripts = {k: list(v) for k, v in by_model.items()}

    async def get_plan(self, system_prompt, scenario_text, *, model=None):
        return self._scripts[model].pop(0)

    async def get_action(self, system_prompt, images, scenario_text, *, max_tokens=2048, model=None):
        return self._scripts[model].pop(0)


@pytest.mark.asyncio
async def test_perception_parses_json() -> None:
    client = _StubClient({"vision-model": ['{"condition":"ok","fragility":"fragile"}']})
    agent = PerceptionAgent(client=client, model="vision-model")
    msg = await agent.run(task="Box A: 4kg, 1d x 1d x 1d", context=[])
    assert msg.role == "perception"
    assert msg.data == {"condition": "ok", "fragility": "fragile"}


@pytest.mark.asyncio
async def test_planner_consumes_perception() -> None:
    client = _StubClient(
        {
            "planner-model": [
                '{"action":"PICK_AND_PLACE","box":"A","target_pallet":1,'
                '"position":[0,0,0],"speed_pct":80,"grip_strength":"standard","reason":"x"}'
            ]
        }
    )
    agent = PlannerAgent(client=client, model="planner-model")
    perception_msg = AgentMessage(
        role="perception", content="", data={"condition": "ok", "fragility": "robust"}
    )
    msg = await agent.run(task="scenario", context=[perception_msg])
    assert msg.data["action"] == "PICK_AND_PLACE"


@pytest.mark.asyncio
async def test_verifier_rejects_unsafe_plan() -> None:
    client = _StubClient(
        {
            "verifier-model": [
                '{"verdict":"reject","reason":"fragile box at high speed"}'
            ]
        }
    )
    agent = VerifierAgent(client=client, model="verifier-model")
    plan = AgentMessage(
        role="planner",
        content="",
        data={
            "action": "PICK_AND_PLACE", "box": "A", "target_pallet": 1,
            "position": [0, 0, 0], "speed_pct": 95, "grip_strength": "firm",
            "reason": "x",
        },
    )
    perception = AgentMessage(role="perception", content="", data={"fragility": "fragile"})
    msg = await agent.run(task="t", context=[perception, plan])
    assert msg.data["verdict"] == "reject"


@pytest.mark.asyncio
async def test_conductor_recurses_on_reject() -> None:
    client = _StubClient(
        {
            "vision-model": ['{"condition":"ok","fragility":"fragile"}'],
            "planner-model": [
                # First plan: rejected
                '{"action":"PICK_AND_PLACE","box":"A","target_pallet":1,'
                '"position":[0,0,0],"speed_pct":95,"grip_strength":"firm","reason":"first"}',
                # Second plan: safer
                '{"action":"PICK_AND_PLACE","box":"A","target_pallet":1,'
                '"position":[0,0,2],"speed_pct":35,"grip_strength":"gentle","reason":"second"}',
            ],
            "verifier-model": [
                '{"verdict":"reject","reason":"firm grip on fragile"}',
                '{"verdict":"accept","reason":"safe"}',
            ],
        }
    )

    perception = PerceptionAgent(client=client, model="vision-model")
    planner = PlannerAgent(client=client, model="planner-model")
    verifier = VerifierAgent(client=client, model="verifier-model")
    conductor = Conductor(perception=perception, planner=planner, verifier=verifier, max_retries=1)

    result = await conductor.run(
        scenario_text="state",
        boxes=[("A", b"\x89PNG", "4kg, 1d x 1d x 1d")],
    )
    assert result.action["reason"] == "second"
    roles = [m.role for m in result.transcript]
    assert roles == ["perception", "planner", "verifier", "planner", "verifier"]


@pytest.mark.asyncio
async def test_conductor_emits_events() -> None:
    client = _StubClient(
        {
            "v": ['{"condition":"ok"}'],
            "p": ['{"action":"WAIT","reason":"wait"}'],
            "x": ['{"verdict":"accept","reason":"ok"}'],
        }
    )
    perception = PerceptionAgent(client=client, model="v")
    planner = PlannerAgent(client=client, model="p")
    verifier = VerifierAgent(client=client, model="x")

    emitted: list[str] = []

    async def emit(msg: AgentMessage) -> None:
        emitted.append(msg.role)

    conductor = Conductor(
        perception=perception, planner=planner, verifier=verifier, max_retries=0, emit=emit
    )
    await conductor.run(scenario_text="s", boxes=[("A", b"x", "")])
    assert emitted == ["perception", "planner", "verifier"]
