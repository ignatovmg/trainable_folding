from typing import Tuple, Any, Sequence, Callable, Optional

import numpy as np
import torch

def apply_rot_to_vec(
    rot: torch.Tensor,
    vec: torch.Tensor
) -> torch.Tensor:
    x = vec[..., 0]
    y = vec[..., 1]
    z = vec[..., 2]
    return torch.stack(
        [
            rot[..., 0, 0] * x + rot[..., 0, 1] * y + rot[..., 0, 2] * z,
            rot[..., 1, 0] * x + rot[..., 1, 1] * y + rot[..., 1, 2] * z,
            rot[..., 2, 0] * x + rot[..., 2, 1] * y + rot[..., 2, 2] * z,
        ],
        dim=-1,
    )


def _multiply(a, b):
    row_1 = torch.stack(
        [
            a[..., 0, 0] * b[..., 0, 0]
            + a[..., 0, 1] * b[..., 1, 0]
            + a[..., 0, 2] * b[..., 2, 0],
            a[..., 0, 0] * b[..., 0, 1]
            + a[..., 0, 1] * b[..., 1, 1]
            + a[..., 0, 2] * b[..., 2, 1],
            a[..., 0, 0] * b[..., 0, 2]
            + a[..., 0, 1] * b[..., 1, 2]
            + a[..., 0, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[..., 1, 0] * b[..., 0, 0]
            + a[..., 1, 1] * b[..., 1, 0]
            + a[..., 1, 2] * b[..., 2, 0],
            a[..., 1, 0] * b[..., 0, 1]
            + a[..., 1, 1] * b[..., 1, 1]
            + a[..., 1, 2] * b[..., 2, 1],
            a[..., 1, 0] * b[..., 0, 2]
            + a[..., 1, 1] * b[..., 1, 2]
            + a[..., 1, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[..., 2, 0] * b[..., 0, 0]
            + a[..., 2, 1] * b[..., 1, 0]
            + a[..., 2, 2] * b[..., 2, 0],
            a[..., 2, 0] * b[..., 0, 1]
            + a[..., 2, 1] * b[..., 1, 1]
            + a[..., 2, 2] * b[..., 2, 1],
            a[..., 2, 0] * b[..., 0, 2]
            + a[..., 2, 1] * b[..., 1, 2]
            + a[..., 2, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)

def invert_point(points, trans, rots):
    points = points - trans
    return apply_rot_to_vec(rots, points)


def make_canonical_transform(
        n_xyz: torch.tensor,
        ca_xyz: torch.tensor,
        c_xyz: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """Returns translation and rotation matrices to canonicalize residue atoms.

    Note that this method does not take care of symmetries. If you provide the
    atom positions in the non-standard way, the N atom will end up not at
    [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
    need to take care of such cases in your code.

    Args:
      n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
      ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
      c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

    Returns:
      A tuple (translation, rotation) where:
        translation is an array of shape [batch, 3] defining the translation.
        rotation is an array of shape [batch, 3, 3] defining the rotation.
      After applying the translation and rotation to all atoms in a residue:
        * All atoms will be shifted so that CA is at the origin,
        * All atoms will be rotated so that C is at the x-axis,
        * All atoms will be shifted so that N is in the xy plane.
    """

    # Place CA at the origin.
    translation = -1*ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation

    # Place C on the x-axis.
    c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
    # Rotate by angle c1 in the x-y plane (around the z-axis).
    sin_c1 = -c_y / torch.sqrt(1e-20 + c_x**2 + c_y**2)
    cos_c1 = c_x / torch.sqrt(1e-20 + c_x**2 + c_y**2)
    zeros = torch.zeros_like(sin_c1)
    ones = torch.ones_like(sin_c1)
    # pylint: disable=bad-whitespace
    c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
    c1_rots[..., 0, 0] = cos_c1
    c1_rots[..., 0, 1] = -1 * sin_c1
    c1_rots[..., 1, 0] = sin_c1
    c1_rots[..., 1, 1] = cos_c1
    c1_rots[..., 2, 2] = 1

    # Rotate by angle c2 in the x-z plane (around the y-axis).
    sin_c2 = c_z / torch.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
    cos_c2 = torch.sqrt(c_x**2 + c_y**2) / torch.sqrt(
        1e-20 + c_x**2 + c_y**2 + c_z**2)

    c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
    c2_rots[..., 0, 0] = cos_c2
    c2_rots[..., 0, 2] = sin_c2
    c2_rots[..., 1, 1] = 1
    c2_rots[..., 2, 0] = -1 * sin_c2
    c2_rots[..., 2, 2] = cos_c2

    c_rot_matrix = _multiply(c2_rots, c1_rots)
    n_xyz = apply_rot_to_vec(c_rot_matrix, n_xyz)

    # Place N in the x-y plane.
    _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
    # Rotate by angle alpha in the y-z plane (around the x-axis).
    sin_n = -n_z / torch.sqrt(1e-20 + n_y**2 + n_z**2)
    cos_n = n_y / torch.sqrt(1e-20 + n_y**2 + n_z**2)
    n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
    n_rots[..., 0, 0] = 1
    n_rots[..., 1, 1] = cos_n
    n_rots[..., 1, 2] = -1 * sin_n
    n_rots[..., 2, 1] = sin_n
    n_rots[..., 2, 2] = cos_n

    rots = _multiply(n_rots, c_rot_matrix)
    rots = rots.to(dtype=torch.float32)
    translation = -1 * translation

    return rots, translation

