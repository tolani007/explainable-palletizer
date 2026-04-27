"""Conductor multi-agent team.

Implements the orchestration pattern described in
"Learning to Orchestrate Agents in Natural Language with the Conductor"
(Sakana, ICLR 2026, arXiv:2512.04388). A single `Conductor` model decides,
in natural language, which specialist agent runs next, what subtask each
gets, and what context they see. The specialist agents (perception,
planner, verifier) each call a different NVIDIA-hosted endpoint model
matched to their role.
"""

from dr_ai_palletizer.agents.base import Agent, AgentMessage
from dr_ai_palletizer.agents.conductor import Conductor
from dr_ai_palletizer.agents.perception import PerceptionAgent
from dr_ai_palletizer.agents.planner import PlannerAgent
from dr_ai_palletizer.agents.verifier import VerifierAgent

__all__ = [
    "Agent",
    "AgentMessage",
    "Conductor",
    "PerceptionAgent",
    "PlannerAgent",
    "VerifierAgent",
]
