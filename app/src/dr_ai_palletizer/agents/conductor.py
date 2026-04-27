"""Conductor: orchestrates the perception/planner/verifier team.

Mirrors the orchestration pattern from "Learning to Orchestrate Agents in
Natural Language with the Conductor" (Sakana, ICLR 2026, arXiv:2512.04388).
The original Conductor is a 7B model trained end-to-end with RL; here we
implement the same dispatch pattern with a strong instruction-tuned model
acting as the meta-prompt engineer. It decides which sub-agent runs next,
what subtask each receives, and -- crucially -- can recurse when the
verifier rejects, re-issuing the planner with the verifier's feedback as
new constraint.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from dr_ai_palletizer.agents.base import AgentMessage
from dr_ai_palletizer.agents.perception import PerceptionAgent
from dr_ai_palletizer.agents.planner import PlannerAgent
from dr_ai_palletizer.agents.verifier import VerifierAgent

logger = logging.getLogger(__name__)

EventEmitter = Callable[[AgentMessage], Awaitable[None]]


@dataclass
class ConductorResult:
    """Final output of one Conductor turn."""

    action: dict  # the planner's accepted action JSON
    transcript: list[AgentMessage] = field(default_factory=list)


@dataclass
class Conductor:
    """Top-level orchestrator for one decision step.

    On each call to `run()`:
      1. Run perception once per box.
      2. Run planner with all perception output.
      3. Run verifier on the planner's action.
      4. If verifier rejects, re-run planner with the rejection as a
         constraint. Up to `max_retries` times.
    """

    perception: PerceptionAgent
    planner: PlannerAgent
    verifier: VerifierAgent
    max_retries: int = 1
    emit: EventEmitter | None = None

    async def _emit(self, msg: AgentMessage) -> None:
        if self.emit:
            await self.emit(msg)

    async def run(
        self,
        *,
        scenario_text: str,
        boxes: list[tuple[str, bytes, str]],
    ) -> ConductorResult:
        """Drive one decision.

        Parameters
        ----------
        scenario_text:
            Pallet state, valid-position table, and last action -- everything
            the planner needs.
        boxes:
            Up to three (box_id, image_bytes, weight_dims_text) tuples for
            the boxes currently visible at the front of the conveyor.
        """
        transcript: list[AgentMessage] = []

        # 1) Perception, one call per box.
        for box_id, image_bytes, meta in boxes:
            seed = AgentMessage(
                role="conductor",
                content=f"Inspect box {box_id}.",
                data={"image_bytes": image_bytes},
            )
            obs = await self.perception.run(
                task=f"Box {box_id}: {meta}",
                context=[seed],
            )
            transcript.append(obs)
            await self._emit(obs)

        # 2) Planner.
        plan = await self.planner.run(task=scenario_text, context=transcript)
        transcript.append(plan)
        await self._emit(plan)

        # 3) Verifier (with recursive re-plan on reject).
        for attempt in range(self.max_retries + 1):
            verdict = await self.verifier.run(task=scenario_text, context=transcript)
            transcript.append(verdict)
            await self._emit(verdict)

            if verdict.data.get("verdict") == "accept":
                break
            if attempt >= self.max_retries:
                logger.info("Conductor: verifier rejected and retries exhausted")
                break

            constraint = (
                f"Your previous plan was rejected by the safety verifier: "
                f"{verdict.data.get('reason', 'no reason given')}. "
                f"Produce a different action that addresses this concern."
            )
            replan = await self.planner.run(
                task=f"{scenario_text}\n\nADDITIONAL CONSTRAINT: {constraint}",
                context=transcript,
            )
            transcript.append(replan)
            await self._emit(replan)
            plan = replan

        return ConductorResult(action=plan.data, transcript=transcript)
