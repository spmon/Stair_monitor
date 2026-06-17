from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnalyzerState:
    hip_motion_history: dict[int, list[int]] = field(default_factory=dict)
    shoulder_motion_history: dict[int, list[int]] = field(default_factory=dict)
    lane_history: dict[int, list[bool | None]] = field(default_factory=dict)
    lane_last_state: dict[int, dict[str, object]] = field(default_factory=dict)
    lane_last_seen: dict[int, int] = field(default_factory=dict)
    hold_status_history: dict[int, list[str]] = field(default_factory=dict)
    last_valid_direction: dict[int, str] = field(default_factory=dict)
    hand_claim_state: dict[int, dict[str, dict[str, int | str | None]]] = field(
        default_factory=dict
    )
    front_carry_history: dict[int, list[bool]] = field(default_factory=dict)
    front_carry_one_arm_history: dict[int, list[bool]] = field(default_factory=dict)
    backward_history: dict[int, list[bool]] = field(default_factory=dict)
    standing_history: dict[int, list[bool]] = field(default_factory=dict)
    standing_motion_history: dict[int, list[tuple[int, int]]] = field(
        default_factory=dict
    )
    frame_index: int = -1
