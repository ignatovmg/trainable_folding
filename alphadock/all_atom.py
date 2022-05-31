# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Modifications Copyright © 2022 Applied BioComputation Group, Stony Brook University

"""Ops for all atom representations.

Generally we employ two different representations for all atom coordinates,
one is atom37 where each heavy atom corresponds to a given position in a 37
dimensional array, This mapping is non amino acid specific, but each slot
corresponds to an atom of a given name, for example slot 12 always corresponds
to 'C delta 1', positions that are not present for a given amino acid are
zeroed out and denoted by a mask.
The other representation we employ is called atom14, this is a more dense way
of representing atoms with 14 slots. Here a given slot will correspond to a
different kind of atom depending on amino acid type, for example slot 5
corresponds to 'N delta 2' for Aspargine, but to 'C delta 1' for Isoleucine.
14 is chosen because it is the maximum number of heavy atoms for any standard
amino acid.
The order of slots can be found in 'residue_constants.residue_atoms'.
Internally the model uses the atom14 representation because it is
computationally more efficient.
The internal atom14 representation is turned into the atom37 at the output of
the network to facilitate easier conversion to existing protein datastructures.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

from alphadock import residue_constants
from alphadock import r3
from alphadock import quat_affine
from alphadock import utils


def squared_difference(x, y):
    return torch.square(x - y)


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in residue_constants.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append(
                [residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    return torch.as_tensor(chi_atom_indices)


def atom14_to_atom37(
        atom14_data: torch.Tensor,  # (N, 14, ...)
        aatype: torch.Tensor,  # (N)
) -> torch.Tensor:  # (N, 37, ...)
    assert atom14_data.shape[1] == 14
    assert aatype.ndim == 1
    assert aatype.shape[0] == atom14_data.shape[0]

    residx_atom37_to_atom14 = torch.tensor(residue_constants.restype_name_to_atom14_ids, device=aatype.device, dtype=aatype.dtype)[aatype]
    atom14_data_flat = atom14_data.reshape(*atom14_data.shape[:2], -1)
    # add 15th field used as placeholder in restype_name_to_atom14_ids
    atom14_data_flat = torch.cat([atom14_data_flat, torch.zeros_like(atom14_data_flat[:, :1])], dim=1)
    out = torch.gather(atom14_data_flat, 1, residx_atom37_to_atom14[..., None].repeat(1, 1, atom14_data_flat.shape[-1]))
    return out.reshape(atom14_data.shape[0], 37, *atom14_data.shape[2:])


def atom37_to_atom14(
    atom37_data: torch.Tensor,  # (N, 37, ...)
    aatype: torch.Tensor,  # (N)
) -> torch.Tensor:  # (N, 14, ...)
    assert atom37_data.shape[1] == 37
    assert aatype.ndim == 1
    assert aatype.shape[0] == atom37_data.shape[0]

    residx_atom14_to_atom37 = torch.tensor(residue_constants.restype_name_to_atom37_ids, device=aatype.device, dtype=aatype.dtype)[aatype]
    atom37_data_flat = atom37_data.reshape(*atom37_data.shape[:2], -1)
    atom37_data_flat = torch.cat([atom37_data_flat, torch.zeros_like(atom37_data_flat[:, :1])], dim=1)
    out = torch.gather(atom37_data_flat, 1, residx_atom14_to_atom37[..., None].repeat(1, 1, atom37_data_flat.shape[-1]))
    return out.reshape(atom37_data.shape[0], 37, *atom37_data.shape[2:])


def atom37_to_frames(
        aatype: torch.Tensor,  # (...)
        all_atom_positions: torch.Tensor,  # (..., 37, 3)
        all_atom_mask: torch.Tensor,  # (..., 37)
) -> Dict[str, torch.Tensor]:
    """Computes the frames for the up to 8 rigid groups for each residue.

    The rigid groups are defined by the possible torsions in a given amino acid.
    We group the atoms according to their dependence on the torsion angles into
    "rigid groups".  E.g., the position of atoms in the chi2-group depend on
    chi1 and chi2, but do not depend on chi3 or chi4.
    Jumper et al. (2021) Suppl. Table 2 and corresponding text.

    Args:
      aatype: Amino acid type, given as array with integers.
      all_atom_positions: atom37 representation of all atom coordinates.
      all_atom_mask: atom37 representation of mask on all atom coordinates.
    Returns:
      Dictionary containing:
        * 'rigidgroups_gt_frames': 8 Frames corresponding to 'all_atom_positions'
             represented as flat 12 dimensional array.
        * 'rigidgroups_gt_exists': Mask denoting whether the atom positions for
            the given frame are available in the ground truth, e.g. if they were
            resolved in the experiment.
        * 'rigidgroups_group_exists': Mask denoting whether given group is in
            principle present for given amino acid type.
        * 'rigidgroups_group_is_ambiguous': Mask denoting whether frame is
            affected by naming ambiguity.
        * 'rigidgroups_alt_gt_frames': 8 Frames with alternative atom renaming
            corresponding to 'all_atom_positions' represented as flat
            12 dimensional array.
    """

    assert all_atom_positions.shape[-2] == 37
    assert all_atom_positions.shape[-1] == 3
    assert all_atom_mask.shape[-1] == 37
    assert list(all_atom_positions.shape[:-2]) == list(all_atom_mask.shape[:-1])

    device = all_atom_positions.device
    dtype = all_atom_positions.dtype

    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'
    aatype_in_shape = aatype.shape

    # If there is a batch axis, just flatten it away, and reshape everything
    # back at the end of the function.
    aatype = torch.reshape(aatype, [-1])
    all_atom_positions = torch.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = torch.reshape(all_atom_mask, [-1, 37])

    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms): (21, 8, 3)
    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], '', dtype=object)

    # 0: backbone frame
    restype_rigidgroup_base_atom_names[:, 0, :] = ['C', 'CA', 'N']

    # 3: 'psi-group'
    restype_rigidgroup_base_atom_names[:, 3, :] = ['CA', 'C', 'O']

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if residue_constants.chi_angles_mask[restype][chi_idx]:
                atom_names = residue_constants.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = atom_names[1:]

    # Create mask for existing rigid groups.
    restype_rigidgroup_mask = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_mask[:, 0] = 1
    restype_rigidgroup_mask[:, 3] = 1
    restype_rigidgroup_mask[:20, 4:] = residue_constants.chi_angles_mask

    # Translate atom names into atom37 indices.
    lookuptable = residue_constants.atom_order.copy()
    lookuptable[''] = 0
    restype_rigidgroup_base_atom37_idx = np.vectorize(lambda x: lookuptable[x])(restype_rigidgroup_base_atom_names)

    # Compute the gather indices for all residues in the chain.
    # shape (N, 8, 3)
    residx_rigidgroup_base_atom37_idx = torch.tensor(restype_rigidgroup_base_atom37_idx, device=device, dtype=aatype.dtype)[aatype]

    # Gather the base atom positions for each rigid group.
    # (N, 8, 3, 3)
    base_atom_pos = torch.gather(
        all_atom_positions[:, :, None, :].repeat([1, 1, 3, 1]), 1,
        residx_rigidgroup_base_atom37_idx[..., None].repeat([1, 1, 1, 3])
    )

    # Compute the Rigids.
    gt_frames = r3.rigids_from_3_points(
        point_on_neg_x_axis=r3.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=r3.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=r3.vecs_from_tensor(base_atom_pos[:, :, 2, :])
    )

    # Compute a mask whether the group exists.
    # (N, 8)
    restype_rigidgroup_mask = torch.tensor(restype_rigidgroup_mask, device=device, dtype=dtype)
    group_exists = restype_rigidgroup_mask[aatype]

    # Compute a mask whether ground truth exists for the group
    # (N, 8, 3)
    gt_atoms_exist = torch.gather(
        all_atom_mask[:, :, None, None].repeat([1, 1, 8, 3]), 1,
        residx_rigidgroup_base_atom37_idx[:, None]
    ).squeeze(1)

    # (N, 8)
    gt_exists = gt_atoms_exist.min(-1).values * group_exists

    # Adapt backbone frame to old convention (mirror x-axis and z-axis).
    rots = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rots[0, 0, 0] = -1
    rots[0, 2, 2] = -1
    gt_frames = r3.rigids_mul_rots(gt_frames, r3.rots_from_tensor3x3(torch.tensor(rots, device=device, dtype=dtype)))

    # The frames for ambiguous rigid groups are just rotated by 180 degree around
    # the x-axis. The ambiguous group is always the last chi-group.
    restype_rigidgroup_is_ambiguous = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_rots = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
        restype = residue_constants.restype_order[residue_constants.restype_3to1[resname]]
        chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[restype, chi_idx + 4, 2, 2] = -1

    # Gather the ambiguity information for each residue.
    residx_rigidgroup_is_ambiguous = torch.tensor(restype_rigidgroup_is_ambiguous, device=device, dtype=dtype)[aatype]
    residx_rigidgroup_ambiguity_rot = torch.tensor(restype_rigidgroup_rots, device=device, dtype=dtype)[aatype]

    # Create the alternative ground truth frames.
    alt_gt_frames = r3.rigids_mul_rots(gt_frames, r3.rots_from_tensor3x3(residx_rigidgroup_ambiguity_rot))

    gt_frames_flat12 = r3.rigids_to_tensor_flat12(gt_frames)
    alt_gt_frames_flat12 = r3.rigids_to_tensor_flat12(alt_gt_frames)

    # reshape back to original residue layout
    gt_frames_flat12 = torch.reshape(gt_frames_flat12, aatype_in_shape + (8, 12))
    gt_exists = torch.reshape(gt_exists, aatype_in_shape + (8,))
    group_exists = torch.reshape(group_exists, aatype_in_shape + (8,))
    gt_frames_flat12 = torch.reshape(gt_frames_flat12, aatype_in_shape + (8, 12))
    residx_rigidgroup_is_ambiguous = torch.reshape(residx_rigidgroup_is_ambiguous, aatype_in_shape + (8,))
    alt_gt_frames_flat12 = torch.reshape(alt_gt_frames_flat12, aatype_in_shape + (8, 12,))

    return {
        'rigidgroups_gt_frames': gt_frames_flat12,  # (..., 8, 12)
        'rigidgroups_gt_exists': gt_exists,  # (..., 8)
        'rigidgroups_group_exists': group_exists,  # (..., 8)
        'rigidgroups_group_is_ambiguous': residx_rigidgroup_is_ambiguous,  # (..., 8)
        'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # (..., 8, 12)
    }


def atom37_to_torsion_angles(
        aatype: torch.Tensor,  # (B, N)
        all_atom_pos: torch.Tensor,  # (B, N, 37, 3)
        all_atom_mask: torch.Tensor,  # (B, N, 37)
        placeholder_for_undefined=False,
) -> Dict[str, torch.Tensor]:
    """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

    The 7 torsion angles are in the order
    '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
    here pre_omega denotes the omega torsion angle between the given amino acid
    and the previous amino acid.

    Args:
      aatype: Amino acid type, given as array with integers.
      all_atom_pos: atom37 representation of all atom coordinates.
      all_atom_mask: atom37 representation of mask on all atom coordinates.
      placeholder_for_undefined: flag denoting whether to set masked torsion
        angles to zero.
    Returns:
      Dict containing:
        * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
          2 dimensions denote sin and cos respectively
        * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
          with the angle shifted by pi for all chi angles affected by the naming
          ambiguities.
        * 'torsion_angles_mask': Mask for which chi angles are present.
    """

    # Map aatype > 20 to 'Unknown' (20).
    aatype = torch.minimum(aatype, torch.full_like(aatype, 20))

    # Compute the backbone angles.
    num_batch, num_res = aatype.shape

    pad = torch.zeros_like(all_atom_pos[:, :1, :, :])
    prev_all_atom_pos = torch.cat([pad, all_atom_pos[:, :-1, :, :]], dim=1)

    pad = torch.zeros_like(all_atom_mask[:, :1, :])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[:, :-1, :]], dim=1)

    # For each torsion angle collect the 4 atom positions that define this angle.
    # shape (B, N, atoms=4, xyz=3)
    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_pos[:, :, 1:3, :],  # prev CA, C
         all_atom_pos[:, :, 0:2, :]  # this N, CA
         ], dim=-2)
    phi_atom_pos = torch.cat(
        [prev_all_atom_pos[:, :, 2:3, :],  # prev C
         all_atom_pos[:, :, 0:3, :]  # this N, CA, C
         ], dim=-2)
    psi_atom_pos = torch.cat(
        [all_atom_pos[:, :, 0:3, :],  # this N, CA, C
         all_atom_pos[:, :, 4:5, :]  # this O
         ], dim=-2)

    # Collect the masks from these atoms.
    # Shape [batch, num_res]
    pre_omega_mask = (
            torch.prod(prev_all_atom_mask[:, :, 1:3], dim=-1)  # prev CA, C
            * torch.prod(all_atom_mask[:, :, 0:2], dim=-1))  # this N, CA
    phi_mask = (
            prev_all_atom_mask[:, :, 2]  # prev C
            * torch.prod(all_atom_mask[:, :, 0:3], dim=-1))  # this N, CA, C
    psi_mask = (
            torch.prod(all_atom_mask[:, :, 0:3], dim=-1) *  # this N, CA, C
            all_atom_mask[:, :, 4])  # this O

    aatype_flat = aatype.flatten()
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices().to(device=aatype.device, dtype=aatype.dtype)
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = chi_atom_indices[aatype_flat].unflatten(0, [num_batch, num_res])
    # Gather atom positions. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].
    chis_atom_pos = torch.gather(all_atom_pos[:, :, None, :, :].repeat(1, 1, 4, 1, 1), 3, atom_indices[..., None].repeat(1, 1, 1, 1, 3))

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(chi_angles_mask, dtype=all_atom_pos.dtype, device=all_atom_pos.device)

    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_res, chis=4].
    chis_mask = chi_angles_mask[aatype_flat].unflatten(0, [num_batch, num_res])

    # Constrain the chis_mask to those chis, where the ground truth coordinates of
    # all defining four atoms are available.
    # Gather the chi angle atoms mask. Shape: [batch, num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = torch.gather(all_atom_mask[:, :, None, :].repeat(1, 1, 4, 1), 3, atom_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1)
    chis_mask = chis_mask * chi_angle_atoms_mask

    # Stack all torsion angle atom positions.
    # Shape (B, N, torsions=7, atoms=4, xyz=3)
    torsions_atom_pos = torch.cat([
        pre_omega_atom_pos[:, :, None, :, :],
        phi_atom_pos[:, :, None, :, :],
        psi_atom_pos[:, :, None, :, :],
        chis_atom_pos
    ], dim=2)

    # Stack up masks for all torsion angles.
    # shape (B, N, torsions=7)
    torsion_angles_mask = torch.cat([
        pre_omega_mask[:, :, None],
        phi_mask[:, :, None],
        psi_mask[:, :, None],
        chis_mask
    ], dim=2)

    # Create a frame from the first three atoms:
    # First atom: point on x-y-plane
    # Second atom: point on negative x-axis
    # Third atom: origin
    # r3.Rigids (B, N, torsions=7)
    torsion_frames = r3.rigids_from_3_points(
        point_on_neg_x_axis=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
        origin=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
        point_on_xy_plane=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :])
    )

    # Compute the position of the forth atom in this frame (y and z coordinate
    # define the chi angle)
    # r3.Vecs (B, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]))

    # Normalize to have the sin and cos of the torsion angle.
    # torch.Tensor (B, N, torsions=7, sincos=2)
    torsion_angles_sin_cos = torch.stack([forth_atom_rel_pos.z, forth_atom_rel_pos.y], dim=-1)
    torsion_angles_sin_cos /= torch.sqrt(torch.sum(torch.square(torsion_angles_sin_cos), dim=-1, keepdim=True) + 1e-8)

    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= torch.tensor(
        [1., 1., -1., 1., 1., 1., 1.], dtype=all_atom_pos.dtype, device=all_atom_pos.device
    )[None, None, :, None]

    # Create alternative angles for ambiguous atom names.
    chi_is_ambiguous = torch.tensor(residue_constants.chi_pi_periodic, dtype=all_atom_pos.dtype, device=all_atom_pos.device)
    chi_is_ambiguous = chi_is_ambiguous[aatype_flat].unflatten(0, [num_batch, num_res])
    mirror_torsion_angles = torch.cat([
        torch.ones([num_batch, num_res, 3], dtype=all_atom_pos.dtype, device=all_atom_pos.device),
         1.0 - 2.0 * chi_is_ambiguous
    ], dim=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])

    if placeholder_for_undefined:
        # Add placeholder torsions in place of undefined torsion angles
        # (e.g. N-terminus pre-omega)
        placeholder_torsions = torch.stack([
            torch.ones(torsion_angles_sin_cos.shape[:-1], dtype=all_atom_pos.dtype, device=all_atom_pos.device),
            torch.zeros(torsion_angles_sin_cos.shape[:-1], dtype=all_atom_pos.dtype, device=all_atom_pos.device)
        ], dim=-1)
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
            ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
            ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,  # (B, N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask  # (B, N, 7)
    }

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def torsion_angles_to_frames(
        aatype: torch.Tensor,  # (N)
        backb_to_global: r3.Rigids,  # (N)
        torsion_angles_sin_cos: torch.Tensor  # (N, 7, 2)
) -> r3.Rigids:  # (N, 8)
    """Compute rigid group frames from torsion angles.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" lines 2-10
    Jumper et al. (2021) Suppl. Alg. 25 "makeRotX"

    Args:
      aatype: aatype for each residue
      backb_to_global: Rigid transformations describing transformation from
        backbone frame to global frame.
      torsion_angles_sin_cos: sin and cosine of the 7 torsion angles
    Returns:
      Frames corresponding to all the Sidechain Rigid Transforms
    """
    assert len(aatype.shape) == 1
    assert len(backb_to_global.rot.xx.shape) == 1
    assert len(torsion_angles_sin_cos.shape) == 3
    assert torsion_angles_sin_cos.shape[1] == 7
    assert torsion_angles_sin_cos.shape[2] == 2

    # Gather the default frames for all rigid groups.
    # r3.Rigids with shape (N, 8)
    m = torch.tensor(residue_constants.restype_rigid_group_default_frame, device=aatype.device)[aatype]
    default_frames = r3.rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = torch.cat([torch.zeros([num_residues, 1], dtype=sin_angles.dtype, device=sin_angles.device), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones([num_residues, 1], dtype=sin_angles.dtype, device=sin_angles.device), cos_angles], dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)

    # all_rots are r3.Rots with shape (N, 8)
    all_rots = r3.Rots(ones, zeros, zeros,
                       zeros, cos_angles, -sin_angles,
                       zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = r3.rigids_mul_rots(default_frames, all_rots)

    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    chi2_frame_to_frame = r3.apply_tree_rigids(lambda x: x[:, 5], all_frames)
    chi3_frame_to_frame = r3.apply_tree_rigids(lambda x: x[:, 6], all_frames)
    chi4_frame_to_frame = r3.apply_tree_rigids(lambda x: x[:, 7], all_frames)

    chi1_frame_to_backb = r3.apply_tree_rigids(lambda x: x[:, 4], all_frames)
    chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
    chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
    chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)

    # Recombine them to a r3.Rigids with shape (N, 8).
    def _concat_frames(xall, x5, x6, x7):
        return torch.cat([xall[:, 0:5], x5[:, None], x6[:, None], x7[:, None]], dim=-1)

    all_frames_to_backb = r3.apply_tree_rigids(
        _concat_frames,
        all_frames,
        chi2_frame_to_backb,
        chi3_frame_to_backb,
        chi4_frame_to_backb)

    # Create the global frames.
    # shape (N, 8)
    all_frames_to_global = r3.rigids_mul_rigids(
        r3.apply_tree_rigids(lambda x: x[:, None], backb_to_global),
        all_frames_to_backb
    )

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
        aatype: torch.Tensor,  # (N)
        all_frames_to_global: r3.Rigids  # (N, 8)
) -> r3.Vecs:  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

    Args:
      aatype: aatype for each residue.
      all_frames_to_global: All per residue coordinate frames.
    Returns:
      Positions of all atom coordinates in global frame.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = torch.tensor(residue_constants.restype_atom14_to_rigid_group, device=aatype.device)[aatype]
    group_mask = F.one_hot(residx_to_group_idx, num_classes=8)  # shape (N, 14, 8)
    #print(torch.stack(all_frames_to_global.rot, dim=-1).unflatten(-1, (3,3)))

    # r3.Rigids with shape (N, 14)
    map_atoms_to_global = r3.apply_tree_rigids(
        lambda x: torch.sum(x[:, None, :] * group_mask, dim=-1),
        all_frames_to_global
    )
    # Gather the literature atom positions for each residue.
    # r3.Vecs with shape (N, 14)
    # restype_atom14_rigid_group_positions (N, 14, 3)
    lit_positions = r3.vecs_from_tensor(torch.tensor(residue_constants.restype_atom14_rigid_group_positions, device=aatype.device)[aatype])

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (N, 14)
    pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = torch.tensor(residue_constants.restype_atom14_mask, device=aatype.device)[aatype]
    pred_positions = r3.apply_tree_vecs(lambda x: x * mask, pred_positions)

    return pred_positions


def l2_normalize(x, axis=-1, epsilon=1e-12):
    buf = torch.sum(x**2, dim=axis, keepdim=True)
    return x / torch.sqrt(torch.maximum(buf, torch.full_like(buf, epsilon)))


def backbone_affine_and_torsions_to_all_atom(
        affine: torch.Tensor,  # (N, 7) QuatAffine in tensor format
        torsions_unnorm: torch.Tensor,  # (N, 14)
        aatype: torch.Tensor  # (N)
):
    affine = quat_affine.QuatAffine(affine[:, :4], [x.squeeze(-1) for x in torch.chunk(affine[:, 4:], 3, dim=-1)])
    backb_to_global = r3.rigids_from_quataffine(affine)

    rec_torsions_unnorm = torsions_unnorm.view(torsions_unnorm.shape[0], 7, 2)
    rec_torsions = l2_normalize(rec_torsions_unnorm)

    outputs = {
        'angles_sin_cos': rec_torsions,  # (N, 7, 2)
        'unnormalized_angles_sin_cos': rec_torsions_unnorm,  # (N, 7, 2)
    }

    all_frames_to_global = torsion_angles_to_frames(
        aatype,
        backb_to_global,
        rec_torsions
    )

    pred_positions = frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global)

    outputs.update({
        'atom_pos': pred_positions,  # r3.Vecs (N, 14)
        'frames': all_frames_to_global,  # r3.Rigids (N, 8),
        'atom_pos_tensor': r3.vecs_to_tensor(pred_positions)  # Tensor (N, 14, 3)
    })
    return outputs


def atom14_to_cbeta_coords(
        atom14_positions,   # (N, 14, 3)
        atom14_mask,   # (N, 14)
        aatype    # (N)
):
    assert len(aatype.shape) == 1, aatype.shape
    assert len(atom14_mask.shape) == 2 and atom14_mask.shape[-1] == 14, atom14_mask.shape
    assert len(atom14_positions.shape) == 3 and list(atom14_positions.shape[-2:]) == [14, 3], atom14_positions.shape

    gly_index = residue_constants.restype_order['G']
    cbeta_index = [4 if x != gly_index else 1 for x in aatype]
    cbeta_coords = atom14_positions[range(len(cbeta_index)), cbeta_index]
    cbeta_mask = atom14_mask[range(len(cbeta_index)), cbeta_index]
    return cbeta_coords, cbeta_mask


def find_optimal_renaming(
        atom14_gt_positions: torch.Tensor,  # (N, 14, 3)
        atom14_alt_gt_positions: torch.Tensor,  # (N, 14, 3)
        atom14_atom_is_ambiguous: torch.Tensor,  # (N, 14)
        atom14_gt_exists: torch.Tensor,  # (N, 14)
        atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
        atom14_atom_exists: torch.Tensor,  # (N, 14)
) -> torch.Tensor:  # (N):
    """Find optimal renaming for ground truth that maximizes LDDT.

    Jumper et al. (2021) Suppl. Alg. 26
    "renameSymmetricGroundTruthAtoms" lines 1-5

    Args:
      atom14_gt_positions: Ground truth positions in global frame of ground truth.
      atom14_alt_gt_positions: Alternate ground truth positions in global frame of
        ground truth with coordinates of ambiguous atoms swapped relative to
        'atom14_gt_positions'.
      atom14_atom_is_ambiguous: Mask denoting whether atom is among ambiguous
        atoms, see Jumper et al. (2021) Suppl. Table 3
      atom14_gt_exists: Mask denoting whether atom at positions exists in ground
        truth.
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type

    Returns:
      Float array of shape [N] with 1. where atom14_alt_gt_positions is closer to
      prediction and 0. otherwise
    """
    assert len(atom14_gt_positions.shape) == 3
    assert len(atom14_alt_gt_positions.shape) == 3
    assert len(atom14_atom_is_ambiguous.shape) == 2
    assert len(atom14_gt_exists.shape) == 2
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2

    # Create the pred distance matrix.
    # shape (N, N, 14, 14)
    pred_dists = torch.sqrt(1e-10 + torch.sum(
        squared_difference(
            atom14_pred_positions[:, None, :, None, :],
            atom14_pred_positions[None, :, None, :, :]),
        dim=-1))

    # Compute distances for ground truth with original and alternative names.
    # shape (N, N, 14, 14)
    gt_dists = torch.sqrt(1e-10 + torch.sum(
        squared_difference(
            atom14_gt_positions[:, None, :, None, :],
            atom14_gt_positions[None, :, None, :, :]),
        dim=-1))
    alt_gt_dists = torch.sqrt(1e-10 + torch.sum(
        squared_difference(
            atom14_alt_gt_positions[:, None, :, None, :],
            atom14_alt_gt_positions[None, :, None, :, :]),
        dim=-1))

    # Compute LDDT's.
    # shape (N, N, 14, 14)
    lddt = torch.sqrt(1e-10 + squared_difference(pred_dists, gt_dists))
    alt_lddt = torch.sqrt(1e-10 + squared_difference(pred_dists, alt_gt_dists))

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    # shape (N ,N, 14, 14)
    mask = (atom14_gt_exists[:, None, :, None] *  # rows
            atom14_atom_is_ambiguous[:, None, :, None] *  # rows
            atom14_gt_exists[None, :, None, :] *  # cols
            (1. - atom14_atom_is_ambiguous[None, :, None, :]))  # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    # shape (N)
    per_res_lddt = torch.sum(mask * lddt, dim=[1, 2, 3])
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=[1, 2, 3])

    # Decide for each residue, whether alternative naming is better.
    # shape (N)
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(atom14_gt_positions.dtype)

    return alt_naming_is_better  # shape (N)


def compute_renamed_ground_truth(
        atom14_gt_positions,
        atom14_alt_gt_positions,
        atom14_atom_is_ambiguous,
        atom14_gt_exists,
        atom14_alt_gt_exists,
        atom14_pred_positions,
        atom14_atom_exists
) -> Dict[str, torch.Tensor]:
    """Find optimal renaming of ground truth based on the predicted positions.

    Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.
    Shape (N).

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
        (N, 14, 3).
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    alt_naming_is_better = find_optimal_renaming(
        atom14_gt_positions=atom14_gt_positions,
        atom14_alt_gt_positions=atom14_alt_gt_positions,
        atom14_atom_is_ambiguous=atom14_atom_is_ambiguous,
        atom14_gt_exists=atom14_gt_exists,
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists
    )

    renamed_atom14_gt_positions = (
            (1. - alt_naming_is_better[:, None, None])
            * atom14_gt_positions
            + alt_naming_is_better[:, None, None]
            * atom14_alt_gt_positions)

    renamed_atom14_gt_mask = (
            (1. - alt_naming_is_better[:, None]) * atom14_gt_exists
            + alt_naming_is_better[:, None] * atom14_alt_gt_exists)

    return {
        'alt_naming_is_better': alt_naming_is_better,  # (N)
        'renamed_atom14_gt_positions': renamed_atom14_gt_positions,  # (N, 14, 3)
        'renamed_atom14_gt_exists': renamed_atom14_gt_mask,  # (N, 14)
    }


def frame_aligned_point_error(
        pred_frames: r3.Rigids,  # shape (..., num_frames)
        target_frames: r3.Rigids,  # shape (..., num_frames)
        frames_mask: torch.Tensor,  # shape (..., num_frames)
        pred_positions: r3.Vecs,  # shape (..., num_positions)
        target_positions: r3.Vecs,  # shape (..., num_positions)
        positions_mask: torch.Tensor,  # shape (..., num_positions)
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        epsilon=1e-6,
        squared=False
) -> torch.Tensor:  # shape ()
    """Measure point error under different alignments.

    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
      pred_frames: num_frames reference frames for 'pred_positions'.
      target_frames: num_frames reference frames for 'target_positions'.
      frames_mask: Mask for frame pairs to use.
      pred_positions: num_positions predicted positions of the structure.
      target_positions: num_positions target positions of the structure.
      positions_mask: Mask on which positions to score.
      length_scale: length scale to divide loss by.
      l1_clamp_distance: Distance cutoff on error beyond which gradients will
        be zero.
      epsilon: small value used to regularize denominator for masked average.
    Returns:
      Masked Frame Aligned Point Error.
    """
    assert list(pred_frames.rot.xx.shape) == list(target_frames.rot.xx.shape), (pred_frames.rot.xx.shape, target_frames.rot.xx.shape)
    assert list(pred_frames.rot.xx.shape) == list(frames_mask.shape), (pred_frames.rot.xx.shape, frames_mask.shape)
    assert list(pred_frames.rot.xx.shape[:-1]) == list(pred_positions.x.shape[:-1]), (pred_frames.rot.xx.shape, pred_positions.x.shape)
    assert list(pred_positions.x.shape) == list(target_positions.x.shape), (pred_positions.x.shape, target_positions.x.shape)
    assert list(pred_positions.x.shape) == list(positions_mask.shape), (pred_positions.x.shape, positions_mask.shape)

    # Compute array of predicted positions in the predicted frames.
    # r3.Vecs (num_frames, num_positions)
    local_pred_pos = r3.rigids_mul_vecs(
        r3.apply_tree_rigids(lambda r: r[..., :, None], r3.invert_rigids(pred_frames)),
        r3.apply_tree_vecs(lambda x: x[..., None, :], pred_positions)
    )

    # Compute array of target positions in the target frames.
    # r3.Vecs (num_frames, num_positions)
    local_target_pos = r3.rigids_mul_vecs(
        r3.apply_tree_rigids(lambda r: r[..., :, None], r3.invert_rigids(target_frames)),
        r3.apply_tree_vecs(lambda x: x[..., None, :], target_positions)
    )

    # Compute errors between the structures.
    error_dist = r3.vecs_squared_distance(local_pred_pos, local_target_pos)
    if not squared:
        error_dist = torch.sqrt(error_dist + epsilon * epsilon)
    else:
        l1_clamp_distance = l1_clamp_distance**2 if l1_clamp_distance is not None else None
        length_scale = length_scale**2

    if l1_clamp_distance:
        error_dist = torch.clip(error_dist, 0, l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= frames_mask.unsqueeze(-1)
    normed_error *= positions_mask.unsqueeze(-2)

    normalization_factor = (
            torch.sum(frames_mask, dim=-1) *
            torch.sum(positions_mask, dim=-1)
    )
    return (torch.sum(normed_error, dim=[-2, -1]) + epsilon) / (epsilon + normalization_factor)


def _make_renaming_matrices():
    """Matrices to map atoms to symmetry partners in ambiguous case."""
    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative groundtruth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ['UNK']
    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[
                resname].index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names[
                resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])
    return renaming_matrices


RENAMING_MATRICES = _make_renaming_matrices()


def format_pdb_line(serial, name, resname, chain, resnum, x, y, z, element, hetatm=False, bfactor=None):
    name = name if len(name) == 4 else ' ' + name
    bfactor = " "*6 if bfactor is None else f"{bfactor: 6.2f}"
    line = f'{"HETATM" if hetatm else "ATOM  "}{serial:>5d} {name:4s} {resname:3s} {chain:1s}{resnum:>4d}' \
           f'    {x: 8.3f}{y: 8.3f}{z: 8.3f}{" "*6}{bfactor}{" "*10}{element:>2s}'
    return line


def atom14_to_pdb_stream(stream, aatypes, atom14_coords, atom14_mask=None, bfactors=None, chain='A', serial_start=1, resnum_start=1):
    assert len(aatypes.shape) == 1, aatypes.shape
    assert len(atom14_coords.shape) == 3, atom14_coords.shape
    assert atom14_coords.shape[0] == aatypes.shape[0], (atom14_coords.shape, aatypes.shape)
    assert atom14_coords.shape[-1] == 3, atom14_coords.shape
    if atom14_mask is not None:
        assert len(atom14_mask.shape) == 2, atom14_mask.shape
        assert atom14_mask.shape[0] == aatypes.shape[0], (atom14_mask.shape, aatypes.shape)
    if bfactors is not None:
        assert len(bfactors.shape) == 1, bfactors.shape
        assert bfactors.shape[0] == aatypes.shape[0], (bfactors.shape[0], aatypes.shape[0])

    serial = serial_start
    for resi, aatype in enumerate(aatypes):
        if aatype >= len(residue_constants.restypes):
            continue
        aa1 = residue_constants.restypes[aatype]
        resname = residue_constants.restype_1to3[aa1]
        for ix, name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if name == '':
                continue
            if atom14_mask is not None and atom14_mask[resi, ix] < 1.0:
                continue
            x, y, z = atom14_coords[resi, ix]
            element = name[0]
            bfactor = None if bfactors is None else bfactors[resi]
            pdb_line = format_pdb_line(serial, name, resname, chain, resi+resnum_start, x, y, z, element, bfactor=bfactor)
            stream.write(pdb_line + '\n')
            serial += 1
    return serial


def ligand_to_pdb_stream(stream, atom_types, coords, resname='LIG', resnum=1, chain='A', serial_start=1):
    assert len(atom_types.shape) == 1, atom_types.shape
    assert len(coords.shape) == 2, coords.shape
    assert coords.shape[0] == atom_types.shape[0], (coords.shape, atom_types.shape)
    assert coords.shape[-1] == 3, coords.shape

    serial = serial_start
    for ix, type_num in enumerate(atom_types):
        element = residue_constants.ELEMENTS_ORDER[type_num]
        x, y, z = coords[ix]
        name = (str(ix+1) + element)[:4]
        pdb_line = format_pdb_line(serial, name, resname, chain, resnum, x, y, z, element, hetatm=True)
        stream.write(pdb_line + '\n')
        serial += 1
    return serial


if __name__ == '__main__':
    pass
