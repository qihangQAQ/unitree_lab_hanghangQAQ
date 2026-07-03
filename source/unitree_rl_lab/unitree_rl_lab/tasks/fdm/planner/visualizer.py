"""FDM-MPPI planner visualisation — simplified.

Only draws the best trajectory (green) and goal sphere (red=not reached, green=reached).
"""

from __future__ import annotations

import math

import torch

try:
    from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
except ImportError:
    omni_debug_draw = None


def _is_gui_available() -> bool:
    return omni_debug_draw is not None


class FDMVisualizer:
    """Minimal debug-draw for MPPI demo."""

    def __init__(self):
        self._draw = omni_debug_draw.acquire_debug_draw_interface() if _is_gui_available() else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        if self._draw is not None:
            self._draw.clear_lines()

    def draw_best_trajectory(
        self,
        world_states: torch.Tensor,
        robot_base_z: float = 0.8,
    ) -> None:
        """Draw the best (selected) trajectory as a thick green line.

        Args:
            world_states:  (1, horizon, 3) world-frame [x, y, yaw].
            robot_base_z:  z height for the line.
        """
        if self._draw is None or world_states is None:
            return

        states = world_states[0]  # (horizon, 3) = [x, y, yaw]
        if states.shape[0] < 2:
            return

        h = states.shape[0]
        z = robot_base_z + 0.02

        starts = torch.stack([states[:-1, 0], states[:-1, 1], torch.full((h - 1,), z, device=states.device)], dim=-1).cpu()
        ends   = torch.stack([states[1:, 0],  states[1:, 1],  torch.full((h - 1,), z, device=states.device)], dim=-1).cpu()

        colour = (0.2, 0.9, 0.2, 1.0)   # green
        width = 8.0
        n = starts.shape[0]

        self._draw.draw_lines(
            starts.tolist(),
            ends.tolist(),
            [colour] * n,
            [width] * n,
        )

    def draw_goal_sphere(
        self, x: float, y: float, z: float = 0.0, reached: bool = False
    ) -> None:
        """Draw a wireframe sphere at the goal position.

        Args:
            x, y, z:  world position.
            reached:  if True, green; if False, red.
        """
        if self._draw is None:
            return

        colour = (0.2, 0.9, 0.2, 1.0) if reached else (0.9, 0.2, 0.2, 1.0)
        width = 5.0
        r = 0.3
        n = 16

        # Three orthogonal rings to form a wireframe sphere
        for axis in range(3):
            pts = []
            for i in range(n + 1):
                angle = 2 * math.pi * i / n
                if axis == 0:
                    pts.append((x + r * math.cos(angle), y + r * math.sin(angle), z))
                elif axis == 1:
                    pts.append((x + r * math.cos(angle), y, z + r * math.sin(angle)))
                else:
                    pts.append((x, y + r * math.cos(angle), z + r * math.sin(angle)))
            for i in range(n):
                self._draw.draw_lines([pts[i]], [pts[i + 1]], [colour], [width])
