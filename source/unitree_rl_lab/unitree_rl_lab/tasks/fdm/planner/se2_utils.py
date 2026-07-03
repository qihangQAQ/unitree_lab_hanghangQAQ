"""Hand-written SE(2) math utilities (no pypose dependency)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def get_se2(state: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, yaw) to 3x3 SE(2) homogeneous transform matrix.

    Exact replica of reference FDM ``get_se2`` (utils.py:68-80).
    The last dimension indexes the *row* (0=c1, 1=c2, 2=c3).

    Args:
        state: (..., 3) where last dim is (x, y, yaw).

    Returns:
        (..., 3, 3) where the last dim is the row index:
            row 0: [cos(yaw), sin(yaw), 0      ]
            row 1: [-sin(yaw), cos(yaw), 0     ]
            row 2: [x,        y,        1      ]
    """
    dim = state.dim()
    cos_yaw = torch.cos(state[..., 2])
    sin_yaw = torch.sin(state[..., 2])
    x = state[..., 0]
    y = state[..., 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    c_vec1 = torch.stack([cos_yaw, sin_yaw, zeros], dim=dim - 1)
    c_vec2 = torch.stack([-sin_yaw, cos_yaw, zeros], dim=dim - 1)
    c_vec3 = torch.stack([x, y, ones], dim=dim - 1)
    return torch.stack([c_vec1, c_vec2, c_vec3], dim=dim)


def get_x_y_yaw(se2: torch.Tensor) -> torch.Tensor:
    """Extract (x, y, yaw) from 3x3 SE(2) matrix.

    Exact replica of reference FDM ``get_x_y_yaw`` (utils.py:83-88).

    Args:
        se2: (..., 3, 3) from ``get_se2``.

    Returns:
        (..., 3) where last dim is (x, y, yaw).
    """
    dim = se2.dim()
    return torch.stack(
        [
            se2[..., 0, 2],  # x = row2 col0
            se2[..., 1, 2],  # y = row2 col1
            torch.atan2(se2[..., 1, 0], se2[..., 0, 0]),  # yaw = atan2(sin, cos)
        ],
        dim=dim - 2,
    )


def invert_se2(se2: torch.Tensor) -> torch.Tensor:
    """Invert an SE(2) matrix."""
    R_t = se2[..., :2, :2].transpose(-2, -1)
    t = se2[..., :2, 2:3]
    new_t = -R_t @ t
    result = se2.clone()
    result[..., :2, :2] = R_t
    result[..., :2, 2] = new_t.squeeze(-1)
    return result


def cosine_distance(yaw1: torch.Tensor, yaw2: torch.Tensor) -> torch.Tensor:
    """Cosine distance between two headings.  0 = same, 2 = opposite.

    Args:
        yaw1, yaw2: (...,) radians.

    Returns:
        (...,) in [0, 2].
    """
    x1 = torch.stack([torch.cos(yaw1), torch.sin(yaw1)], dim=-1)
    x2 = torch.stack([torch.cos(yaw2), torch.sin(yaw2)], dim=-1)
    return 1.0 - F.cosine_similarity(x1, x2, dim=-1)


def smallest_angle(yaw1: torch.Tensor, yaw2: torch.Tensor) -> torch.Tensor:
    """Smallest absolute angular difference in [0, pi].

    Args:
        yaw1, yaw2: (...,) radians.

    Returns:
        (...,) angular difference in [0, pi].
    """
    diff = torch.abs((yaw1 % (2 * math.pi)) - (yaw2 % (2 * math.pi)))
    return torch.min(diff, 2 * math.pi - diff)
