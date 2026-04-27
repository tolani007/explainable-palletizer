"""Shared types for the Conductor multi-agent team."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

AgentRole = Literal["perception", "planner", "verifier", "conductor"]


@dataclass
class AgentMessage:
    """A single utterance from one agent in the team.

    The `content` is the agent's natural-language output; `data` carries
    structured fields the next agent (or the control loop) consumes.
    """

    role: AgentRole
    content: str
    data: dict = field(default_factory=dict)
    model: str = ""


class Agent(Protocol):
    """Interface every specialist agent in the team implements."""

    role: AgentRole
    model: str

    async def run(self, *, task: str, context: list[AgentMessage]) -> AgentMessage: ...
