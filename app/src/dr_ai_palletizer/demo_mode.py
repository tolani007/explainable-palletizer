"""Synthetic event generator that mirrors the real control-loop event shape.

When the full NVIDIA stack (Isaac Sim + Cosmos-Reason2 + cuRobo) is not
available -- for example on a CPU-only cloud host or in the GitHub Pages
preview build -- the server can be started with `DEMO_MODE=true`. In that
mode this module replaces the live control-loop and emits a deterministic,
realistic-looking sequence of `reasoning`, `action`, `status`, and
`box_images` events over the same `/api/events` WebSocket the production
build uses, so the UI behaves identically.

The generator is intentionally small and dependency-free: it only uses the
standard library plus an `asyncio.Queue` so it can be driven by FastAPI's
event loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_ai_palletizer.agents import Conductor

logger = logging.getLogger(__name__)

EventBroadcaster = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class _DemoBox:
    box_id: str
    label: str
    weight_kg: float
    shape: tuple[int, int, int]  # (w, l, h) in d-units
    fragile: bool = False
    damaged: bool = False
    color: tuple[int, int, int] = (140, 140, 150)


_DEMO_BOXES: tuple[_DemoBox, ...] = (
    _DemoBox("BOX-001", "Sealed carton, dry", 12.4, (2, 2, 1), color=(170, 130, 90)),
    _DemoBox("BOX-002", "Fragile glassware", 4.2, (1, 1, 1), fragile=True, color=(120, 200, 220)),
    _DemoBox("BOX-003", "Heavy metal parts", 27.8, (2, 1, 1), color=(110, 110, 130)),
    _DemoBox("BOX-004", "Crushed corner, suspect", 9.1, (2, 1, 1), damaged=True, color=(180, 90, 90)),
    _DemoBox("BOX-005", "Standard dry goods", 7.6, (1, 2, 1), color=(160, 150, 110)),
    _DemoBox("BOX-006", "Light packaging", 2.1, (1, 1, 1), color=(200, 200, 210)),
    _DemoBox("BOX-007", "Mid-weight retail", 14.5, (2, 2, 1), color=(150, 110, 170)),
    _DemoBox("BOX-008", "Wet patch on top, fragile", 5.3, (1, 1, 1), fragile=True, damaged=True, color=(80, 130, 180)),
)


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Return a tiny solid-color PNG.

    Uses a hand-rolled minimal encoder so the demo build has zero image
    dependencies (Pillow is not available in the static demo bundle).
    """
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    row = bytes((rgb[0], rgb[1], rgb[2])) * width
    for _ in range(height):
        raw.append(0)  # filter: none
        raw.extend(row)
    idat = zlib.compress(bytes(raw), 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _render_box_thumbnail(box: _DemoBox) -> str:
    """Return a data: URL for a synthetic 96x72 box thumbnail."""
    png = _solid_png(96, 72, box.color)
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _build_reasoning(box: _DemoBox, pallet_idx: int, position: tuple[int, int, int]) -> list[str]:
    """Produce a chain-of-thought trace tailored to the box."""
    lines: list[str] = []
    lines.append(f"Observing {box.box_id}: {box.label}.")
    lines.append(
        f"Measured weight {box.weight_kg:.1f} kg, shape {box.shape[0]}d x {box.shape[1]}d x {box.shape[2]}d."
    )

    if box.damaged:
        lines.append(
            "I see surface damage / contamination on the upper face. Per safety policy, "
            "this is unstackable and must be routed to human inspection before it touches a pallet."
        )
        return lines

    if box.fragile:
        lines.append(
            "Surface and labeling look fragile. I will reduce the placement speed and switch grip "
            "to gentle so I do not crush internal contents."
        )
    if box.weight_kg >= 20:
        lines.append(
            "This is a heavy item. Heavy items should sit at the lowest free z-layer to keep the "
            "pallet's centre of mass low and avoid crushing lighter boxes underneath."
        )
    elif box.weight_kg <= 5:
        lines.append(
            "Light item. Safe to place on top of heavier inventory provided support is adequate."
        )

    lines.append(
        f"Pallet {pallet_idx} is closer to completion -- I will continue filling it before opening "
        f"a new layer on the alternate pallet."
    )
    lines.append(
        f"Selected grid cell ({position[0]}, {position[1]}, {position[2]}). Lateral neighbours "
        f"provide support; placement passes the stacking-stability check."
    )
    return lines


def _build_action(box: _DemoBox, pallet_idx: int, position: tuple[int, int, int]) -> dict:
    """Produce the JSON action the LLM would have emitted."""
    if box.damaged:
        return {
            "action": "CALL_A_HUMAN",
            "boxes": [box.box_id],
            "reason": "Surface damage / contamination detected -- not safe to stack.",
        }
    grip = "gentle" if box.fragile else ("firm" if box.weight_kg >= 20 else "standard")
    speed = 35 if box.fragile else (60 if box.weight_kg >= 20 else 80)
    return {
        "action": "PICK_AND_PLACE",
        "box": box.box_id,
        "target_pallet": pallet_idx,
        "position": list(position),
        "speed_pct": speed,
        "grip_strength": grip,
        "reason": (
            f"{'Fragile' if box.fragile else 'Heavy' if box.weight_kg >= 20 else 'Standard'} "
            f"item placed on pallet {pallet_idx} for stable stacking."
        ),
    }


class DemoControlLoop:
    """Drop-in replacement for `ControlLoop` that emits synthetic events.

    The interface intentionally mirrors `ControlLoop` so the FastAPI server
    can swap implementations based on `Settings.demo_mode`.
    """

    def __init__(
        self,
        broadcast: EventBroadcaster,
        *,
        step_period: float = 4.0,
        conductor: "Conductor | None" = None,
    ) -> None:
        self._broadcast = broadcast
        self._step_period = step_period
        self._conductor = conductor
        self._state = "idle"
        self._stop = asyncio.Event()
        self._pause = asyncio.Event()
        self._pause.set()  # not paused == set
        self._task: asyncio.Task | None = None
        self._step_count = 0

    @property
    def state(self) -> str:
        return self._state

    async def _emit_status(self, state: str) -> None:
        self._state = state
        await self._broadcast(
            {
                "type": "status",
                "state": state,
                "model": "demo://cosmos-reason2-8b (synthetic)",
            }
        )

    async def _emit_box_images(self, boxes: list[_DemoBox]) -> None:
        await self._broadcast(
            {
                "type": "box_images",
                "images": [
                    {"data": _render_box_thumbnail(b), "label": b.box_id} for b in boxes
                ],
            }
        )

    async def _step(self, rng: random.Random) -> None:
        # Pick 1-3 boxes from the demo pool, mirroring how the real loop sees
        # the front of the conveyor.
        n = rng.choice([1, 2, 2, 3])
        boxes = rng.sample(_DEMO_BOXES, n)
        await self._emit_box_images(list(boxes))

        await self._broadcast({"type": "status", "state": "thinking"})
        await asyncio.sleep(0.3)

        if self._conductor is not None:
            await self._step_with_conductor(boxes)
        else:
            await self._step_synthetic(boxes, rng)
        await self._broadcast({"type": "status", "state": "running"})
        self._step_count += 1

    async def _step_synthetic(self, boxes: list[_DemoBox], rng: random.Random) -> None:
        primary = next((b for b in boxes if b.damaged), None) or max(boxes, key=lambda b: b.weight_kg)
        pallet_idx = rng.choice([1, 2])
        position = (rng.randrange(0, 4), rng.randrange(0, 4), rng.randrange(0, 3))
        for line in _build_reasoning(primary, pallet_idx, position):
            await self._broadcast({"type": "reasoning", "content": line})
            await asyncio.sleep(0.35)
        action = _build_action(primary, pallet_idx, position)
        await self._broadcast({"type": "action", "content": _format_action(action)})

    async def _step_with_conductor(self, boxes: list[_DemoBox]) -> None:
        """Run a real Conductor turn over NVIDIA hosted endpoints."""
        scenario_text = _scenario_text(boxes, step=self._step_count)
        # Emit each agent message as a typed event so the UI can show the team.
        async def emit(msg) -> None:  # AgentMessage; avoid import cycle
            await self._broadcast(
                {
                    "type": "agent_message",
                    "agent": msg.role,
                    "model": msg.model,
                    "content": _strip_think(msg.content),
                    "data": _safe_data(msg.data),
                }
            )

        # Wire the emit on the conductor for this turn.
        prev_emit = self._conductor.emit
        self._conductor.emit = emit
        try:
            box_inputs = [
                (b.box_id, _solid_png(64, 48, b.color), f"{b.weight_kg}kg, {b.shape[0]}d x {b.shape[1]}d x {b.shape[2]}d, {b.label}")
                for b in boxes
            ]
            try:
                result = await self._conductor.run(scenario_text=scenario_text, boxes=box_inputs)
            except Exception as exc:
                logger.exception("Conductor step failed")
                await self._broadcast({"type": "reasoning", "content": f"Conductor error: {exc}"})
                return
        finally:
            self._conductor.emit = prev_emit

        await self._broadcast(
            {"type": "action", "content": _format_action(result.action) if result.action else "WAIT"}
        )

    async def _run(self) -> None:
        rng = random.Random(7)
        try:
            await self._emit_status("running")
            while not self._stop.is_set():
                await self._pause.wait()
                if self._stop.is_set():
                    break
                await self._step(rng)
                # Idle gap between decisions
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._step_period)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Demo control loop crashed")
        finally:
            await self._emit_status("idle")

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._pause.set()
        self._task = asyncio.create_task(self._run())

    def pause(self) -> None:
        self._pause.clear()
        self._state = "paused"

    def resume(self) -> None:
        self._pause.set()
        self._state = "running"

    def reset(self) -> None:
        self._stop.set()
        self._pause.set()
        self._state = "idle"

    async def aclose(self) -> None:
        self.reset()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def _scenario_text(boxes: list[_DemoBox], *, step: int) -> str:
    """Plain-text pallet state + valid-position summary for the planner."""
    lines = [f"Step {step}.", "Boxes visible at the conveyor head:"]
    for b in boxes:
        lines.append(
            f"  - {b.box_id}: {b.weight_kg}kg, "
            f"{b.shape[0]}d x {b.shape[1]}d x {b.shape[2]}d, label='{b.label}'"
        )
    lines.append("")
    lines.append("Pallet 1: empty floor layer (z=0). Pallet 2: empty.")
    lines.append("Valid positions: any (x in 0..3, y in 0..3, z in 0..2).")
    lines.append("Last action: PICK_AND_PLACE.")
    return "\n".join(lines)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks for UI display."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _safe_data(data: dict) -> dict:
    """Drop non-JSON-serializable fields (e.g. raw image bytes)."""
    return {k: v for k, v in data.items() if not isinstance(v, (bytes, bytearray))}


def _format_action(action: dict) -> str:
    """Render an action dict as a single-line human-readable summary."""
    name = action.get("action", "?")
    if name == "PICK_AND_PLACE":
        pos = action.get("position", [0, 0, 0])
        return (
            f"PICK_AND_PLACE  {action.get('box')}  -> pallet "
            f"{action.get('target_pallet')} @ ({pos[0]},{pos[1]},{pos[2]})  "
            f"speed={action.get('speed_pct')}%  grip={action.get('grip_strength')}"
        )
    if name == "CALL_A_HUMAN":
        boxes = ",".join(action.get("boxes", []))
        return f"CALL_A_HUMAN  {boxes}  reason={action.get('reason')}"
    return f"{name}  reason={action.get('reason', '')}"
