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


from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

from alphadock import residue_constants
from alphadock import utils


def find_structural_violations(
        batch,
        atom14_pred_positions,  # (N, 14, 3)
        config
):
    assert len(atom14_pred_positions.shape) == 3

    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = calc_between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['target']['rec_atom14_atom_exists'][0],
        residue_index=batch['target']['rec_index'][0],
        aatype=batch['target']['rec_aatype'][0],
        tolerance_factor_soft=config['loss']['violation_tolerance_factor'],
        tolerance_factor_hard=config['loss']['violation_tolerance_factor'])

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = np.stack([residue_constants.restype_name_to_atom14_atom_radius[residue_constants.restype_1to3[residue_constants.restypes[aatype]]]
                                for aatype in batch['target']['rec_aatype'][0]])
    atomtype_radius = torch.from_numpy(atomtype_radius).to(device=batch['target']['rec_aatype'].device, dtype=batch['target']['rec_1d'].dtype)
    atom14_atom_radius = batch['target']['rec_atom14_atom_exists'][0] * atomtype_radius

    # Compute the between residue clash loss.
    between_residue_clashes = calc_between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['target']['rec_atom14_atom_exists'][0],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch['target']['rec_index'][0],
        overlap_tolerance_soft=config['loss']['clash_overlap_tolerance'],
        overlap_tolerance_hard=config['loss']['clash_overlap_tolerance'])

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=config['loss']['clash_overlap_tolerance'],
        bond_length_tolerance_factor=config['loss']['violation_tolerance_factor']
    )
    restype_atom14_bounds = {k: torch.from_numpy(restype_atom14_bounds[k]).to(device=batch['target']['rec_1d'].device, dtype=batch['target']['rec_1d'].dtype)
                             for k in restype_atom14_bounds.keys()}
    aatype = batch['target']['rec_aatype'][0][:, None, None].repeat(1, 14, 14)
    atom14_dists_lower_bound = torch.gather(restype_atom14_bounds['lower_bound'], 0, aatype)
    atom14_dists_upper_bound = torch.gather(restype_atom14_bounds['upper_bound'], 0, aatype)
    within_residue_violations = calc_within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['target']['rec_atom14_atom_exists'][0],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0)

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack([
            connection_violations['per_residue_violation_mask'],
            torch.max(between_residue_clashes['per_atom_clash_mask'], dim=-1).values,
            torch.max(within_residue_violations['per_atom_violations'], dim=-1).values
        ]),
        dim=0
    ).values

    violations_extreme_ca_ca = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['target']['rec_atom14_has_coords'][0],
        residue_index=batch['target']['rec_index'][0].to(dtype=batch['target']['rec_atom14_coords'].dtype)
    )

    return {
        'between_residues': {
            'bonds_c_n_loss_mean': connection_violations['c_n_loss_mean'],  # ()
            'angles_ca_c_n_loss_mean': connection_violations['ca_c_n_loss_mean'],  # ()
            'angles_c_n_ca_loss_mean': connection_violations['c_n_ca_loss_mean'],  # ()
            'connections_per_residue_loss_sum': connection_violations['per_residue_loss_sum'],  # (N)
            'connections_per_residue_violation_mask': connection_violations['per_residue_violation_mask'],  # (N)
            'clashes_mean_loss': between_residue_clashes['mean_loss'],  # ()
            'clashes_per_atom_loss_sum': between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
            'clashes_per_atom_clash_mask': between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
            'violations_extreme_ca_ca': violations_extreme_ca_ca
        },
        'within_residues': {
            'per_atom_loss_sum': within_residue_violations['per_atom_loss_sum'],  # (N, 14)
            'per_atom_violations': within_residue_violations['per_atom_violations'],  # (N, 14),
        },
        'total_per_residue_violations_mask': per_residue_violations_mask,  # (N)
    }


def _make_lig_bounds(
        atom_types,  # (N)
        bonds_2d,  # (N, N)
        gt_coords,  # (N, 3)
        gt_mask,  # (N)
        vdw_radii,  # (N)
        overlap_tol=1.5,
        bond_length_tolerance_factor=12.0,
        std_rel=0.015
):
    '''
    We are lazy to search for literature bond length values, so determine
    them from the ground truth ligand instead
    '''
    assert len(atom_types.shape) == 1
    assert len(bonds_2d.shape) == 2
    assert bonds_2d.shape[0] == atom_types.shape[0]
    assert len(gt_coords.shape) == 2, gt_coords.shape
    assert len(gt_mask.shape) == 1
    assert len(vdw_radii.shape) == 1

    num_atoms = atom_types.shape[0]
    eye = torch.eye(num_atoms, device=vdw_radii.device, dtype=vdw_radii.dtype)
    noneye = 1. - eye

    nonbonded_lower = vdw_radii[:, None] + vdw_radii[None, :] - overlap_tol
    nonbonded_upper = torch.full_like(bonds_2d, 10e6)

    gt_bond_exists = gt_mask[:, None] * gt_mask[None, :]
    gt_bond_not_exists = torch.abs(1. - gt_bond_exists)
    dmat = torch.sqrt(1e-10 + torch.sum(utils.squared_difference(gt_coords[:, None, :], gt_coords[None, :, :]), dim=-1))
    bonded_lower = (dmat * (1. - std_rel * bond_length_tolerance_factor)) * gt_bond_exists
    bonded_upper = (dmat * (1. + std_rel * bond_length_tolerance_factor)) * gt_bond_exists + 10e6 * gt_bond_not_exists

    bonded_mask = F.relu(bonds_2d - eye)
    nonbonded_mask = 1. - bonded_mask - eye

    lower = (nonbonded_lower * nonbonded_mask + bonded_lower * bonded_mask) * noneye
    upper = (nonbonded_upper * nonbonded_mask + bonded_upper * bonded_mask) * noneye

    #print(lower)
    #print(upper)
    #print(bonds_2d)

    return {
        'lower': lower,
        'upper': upper
    }


def extreme_ca_ca_distance_violations(
        pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
        pred_atom_mask: torch.Tensor,  # (N, 37(14))
        residue_index: torch.Tensor,  # (N)
        max_angstrom_tolerance=1.5
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]         # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).to(this_ca_pos.dtype)
    ca_ca_distance = torch.sqrt(1e-6 + torch.sum(utils.squared_difference(this_ca_pos, next_ca_pos), dim=-1))  # (N - 1)
    violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return (violations * mask).sum() / (mask.sum() + 1e-6)


def calc_between_residue_bond_loss(
        pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
        pred_atom_mask: torch.Tensor,  # (N, 37(14))
        residue_index: torch.Tensor,  # (N)
        aatype: torch.Tensor,  # (N)
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    assert len(pred_atom_positions.shape) == 3
    assert len(pred_atom_mask.shape) == 2
    assert len(residue_index.shape) == 1
    assert len(aatype.shape) == 1

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]         # (N - 1)
    this_c_pos = pred_atom_positions[:-1, 2, :]   # (N - 1, 3)
    this_c_mask = pred_atom_mask[:-1, 2]          # (N - 1)
    next_n_pos = pred_atom_positions[1:, 0, :]    # (N - 1, 3)
    next_n_mask = pred_atom_mask[1:, 0]           # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]   # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]          # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).to(this_ca_pos.dtype)

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(1e-6 + torch.sum(utils.squared_difference(this_c_pos, next_n_pos), dim=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[1:] == residue_constants.resname_to_idx['PRO']).to(this_ca_pos.dtype)
    gt_length = (
            (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
            + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = (
            (1. - next_is_proline) *
            residue_constants.between_res_bond_length_stddev_c_n[0] +
            next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = torch.sqrt(1e-6 + torch.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = torch.nn.functional.relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(1e-6 + torch.sum(utils.squared_difference(this_ca_pos, this_c_pos), dim=-1))
    n_ca_bond_length = torch.sqrt(1e-6 + torch.sum(utils.squared_difference(next_n_pos, next_ca_pos), dim=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(1e-6 + torch.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = torch.nn.functional.relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(1e-6 + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.nn.functional.relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue) / (torch.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue)
    per_residue_loss_sum = 0.5 * (F.pad(per_residue_loss_sum, [0, 1]) + F.pad(per_residue_loss_sum, [1, 0]))

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack([c_n_violation_mask,
                     ca_c_n_violation_mask,
                     c_n_ca_violation_mask]),
        dim=0).values
    violation_mask = torch.maximum(
        F.pad(violation_mask, [0, 1]),
        F.pad(violation_mask, [1, 0])
    )

    return {'c_n_loss_mean': c_n_loss,  # shape ()
            'ca_c_n_loss_mean': ca_c_n_loss,  # shape ()
            'c_n_ca_loss_mean': c_n_ca_loss,  # shape ()
            'per_residue_loss_sum': per_residue_loss_sum,  # shape (N)
            'per_residue_violation_mask': violation_mask  # shape (N)
            }


def calc_between_residue_clash_loss(
        atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
        atom14_atom_exists: torch.Tensor,  # (N, 14)
        atom14_atom_radius: torch.Tensor,  # (N, 14)
        residue_index: torch.Tensor,  # (N)
        overlap_tolerance_soft=1.5,
        overlap_tolerance_hard=1.5
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2
    assert len(atom14_atom_radius.shape) == 2
    assert len(residue_index.shape) == 1

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(1e-10 + torch.sum(
        utils.squared_difference(
            atom14_pred_positions[:, None, :, None, :],
            atom14_pred_positions[None, :, None, :, :]),
        dim=-1))

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (atom14_atom_exists[:, None, :, None] * atom14_atom_exists[None, :, None, :])

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask *= (residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.zeros(14, dtype=atom14_pred_positions.dtype, device=atom14_pred_positions.device)
    c_one_hot[2] = 1
    n_one_hot = torch.zeros(14, dtype=atom14_pred_positions.dtype, device=atom14_pred_positions.device)
    n_one_hot[0] = 1
    neighbour_mask = ((residue_index[:, None, None, None] + 1) == residue_index[None, :, None, None])
    c_n_bonds = neighbour_mask * c_one_hot[None, None, :, None] * n_one_hot[None, None, None, :]
    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
    cys_sg_one_hot = torch.zeros(14, dtype=atom14_pred_positions.dtype, device=atom14_pred_positions.device)
    cys_sg_one_hot[cys_sg_idx] = 1
    disulfide_bonds = (cys_sg_one_hot[None, None, :, None] * cys_sg_one_hot[None, None, None, :])
    dists_mask *= (1. - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] + atom14_atom_radius[None, :, None, :])

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * F.relu(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = (torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask)))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = (torch.sum(dists_to_low_error, dim=[0, 2]) + torch.sum(dists_to_low_error, dim=[1, 3]))

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        clash_mask.max(2).values.max(0).values,
        clash_mask.max(3).values.max(1).values
    )

    return {'mean_loss': mean_loss,  # shape ()
            'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
            'per_atom_clash_mask': per_atom_clash_mask  # shape (N, 14)
            }


def calc_within_residue_violations(
        atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
        atom14_atom_exists: torch.Tensor,  # (N, 14)
        atom14_dists_lower_bound: torch.Tensor,  # (N, 14, 14)
        atom14_dists_upper_bound: torch.Tensor,  # (N, 14, 14)
        tighten_bounds_for_loss=0.0,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_dists_lower_bound: Lower bound on allowed distances.
      atom14_dists_upper_bound: Upper bound on allowed distances
      tighten_bounds_for_loss: Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2
    assert len(atom14_dists_lower_bound.shape) == 3
    assert len(atom14_dists_upper_bound.shape) == 3

    # Compute the mask for each residue.
    # shape (N, 14, 14)
    dists_masks = (1. - torch.eye(14, 14)[None]).to(device=atom14_pred_positions.device, dtype=atom14_pred_positions.dtype)
    dists_masks = dists_masks * (atom14_atom_exists[:, :, None] * atom14_atom_exists[:, None, :])

    # Distance matrix
    # shape (N, 14, 14)
    dists = torch.sqrt(1e-10 + torch.sum(
        utils.squared_difference(
            atom14_pred_positions[:, :, None, :],
            atom14_pred_positions[:, None, :, :]),
        dim=-1))

    # Compute the loss.
    # shape (N, 14, 14)
    dists_to_low_error = F.relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = F.relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = loss.sum(1) + loss.sum(2)

    # Compute the violations mask.
    # shape (N, 14, 14)
    violations = dists_masks * ((dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound))

    # Compute the per atom violations.
    # shape (N, 14)
    per_atom_violations = torch.maximum(torch.max(violations, dim=1).values, torch.max(violations, dim=2).values)

    return {
        'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
        'per_atom_violations': per_atom_violations  # shape (N, 14)
    }


def calc_lig_rec_clash_loss(
        atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
        atom14_atom_exists: torch.Tensor,  # (N, 14)
        atom14_atom_radius: torch.Tensor,  # (N, 14)
        lig_pred_positions: torch.Tensor,  # (N, 3)
        lig_atom_radius: torch.Tensor,  # (N)
        overlap_tolerance_soft=1.5,
        overlap_tolerance_hard=1.5
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2
    assert len(atom14_atom_radius.shape) == 2
    assert len(lig_pred_positions.shape) == 2
    assert len(lig_atom_radius.shape) == 1

    # Create the distance matrix.
    # (Nlig, Nrec, 14)
    dists = torch.sqrt(1e-10 + torch.sum(
        utils.squared_difference(
            lig_pred_positions[:, None, None, :],
            atom14_pred_positions[None, :, :, :]),
        dim=-1))

    # Compute the lower bound for the allowed distances.
    # shape (Nlig, Nrec, 14)
    dists_lower_bound = atom14_atom_exists[None, :, :] * (lig_atom_radius[:, None, None] + atom14_atom_radius[None, :, :])

    # Compute the error.
    # shape (Nlig, Nrec, 14)
    dists_to_low_error = atom14_atom_exists[None, :, :] * F.relu(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / dists.shape[0]

    # Compute the per atom loss sum.
    # shape (N), (N, 14)
    lig_per_atom_loss_sum = torch.sum(dists_to_low_error, dim=[1, 2])
    rec_per_atom_loss_sum = torch.sum(dists_to_low_error, dim=0)

    # Compute the hard clash mask.
    # shape (Nlig, Nrec, 14)
    clash_mask = atom14_atom_exists[None, :, :] * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N), (N, 14)
    lig_per_atom_clash_mask = clash_mask.max(2).values.max(1).values
    rec_per_atom_clash_mask = clash_mask.max(0).values

    return {
        'mean_loss': mean_loss,  # shape ()
        'rec_per_atom_loss_sum': rec_per_atom_loss_sum,  # shape (N, 14)
        'lig_per_atom_loss_sum': lig_per_atom_loss_sum,  # (N)
        'rec_per_atom_clash_mask': rec_per_atom_clash_mask,  # shape (N, 14)
        'lig_per_atom_clash_mask': lig_per_atom_clash_mask  # shape (N)
    }


def calc_within_ligand_violations(
        lig_pred_positions: torch.Tensor,  # (N, 3)
        lig_dists_lower_bound: torch.Tensor,  # (N, N)
        lig_dists_upper_bound: torch.Tensor,  # (N, N)
        tighten_bounds_for_loss=0.0,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      Dict containing:
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    assert len(lig_pred_positions.shape) == 2
    assert len(lig_dists_lower_bound.shape) == 2
    assert len(lig_dists_upper_bound.shape) == 2

    # Compute the mask for each residue.
    # shape (N, N)
    natoms = lig_pred_positions.shape[0]
    dists_mask = (1. - torch.eye(natoms, natoms)).to(device=lig_pred_positions.device, dtype=lig_pred_positions.dtype)

    # Distance matrix
    # shape (N, N)
    dists = torch.sqrt(1e-10 + torch.sum(utils.squared_difference(lig_pred_positions[:, None, :], lig_pred_positions[None, :, :]), dim=-1))

    # Compute the loss.
    # shape (N, N)
    dists_to_low_error = F.relu(lig_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = F.relu(dists - (lig_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_mask * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # shape (N)
    per_atom_loss_sum = (loss.sum(0) + loss.sum(1)) / 2

    # Compute the violations mask.
    # shape (N, N)
    violations = dists_mask * ((dists < lig_dists_lower_bound) | (dists > lig_dists_upper_bound))

    # Compute the per atom violations.
    # shape (N)
    per_atom_violations = torch.maximum(violations.max(0).values, violations.max(1).values)

    return {
        'per_atom_loss_sum': per_atom_loss_sum,  # shape (N)
        'per_atom_violations': per_atom_violations  # shape (N)
    }


def example4():
    from config import config, DATA_DIR
    from dataset import DockingDataset
    ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
    item = ds[0]

    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)

    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=0.5,
        bond_length_tolerance_factor=12.0)

    for k in restype_atom14_bounds.keys():
        print(k, ":", restype_atom14_bounds[k].shape)
        restype_atom14_bounds[k] = torch.from_numpy(restype_atom14_bounds[k]).to(device=item['target']['rec_1d'].device, dtype=item['target']['rec_1d'].dtype)

    aatype = item['target']['rec_aatype'][0][:, None, None].repeat(1, 14, 14)
    atom14_dists_lower_bound = torch.gather(restype_atom14_bounds['lower_bound'], 0, aatype)
    print(atom14_dists_lower_bound.shape)
    print(torch.all(atom14_dists_lower_bound[10] - restype_atom14_bounds['lower_bound'][item['target']['rec_aatype'][0][10]]) == 0.0)

    atom14_dists_upper_bound = torch.gather(restype_atom14_bounds['upper_bound'], 0, aatype)
    viol = calc_within_residue_violations(
        atom14_pred_positions=item['target']['rec_atom14_coords'][0],
        atom14_atom_exists=item['target']['rec_atom14_has_coords'][0],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0)
    print(viol)

    connection_violations = calc_between_residue_bond_loss(
        pred_atom_positions=item['target']['rec_atom14_coords'][0],
        pred_atom_mask=item['target']['rec_atom14_has_coords'][0],
        residue_index=item['target']['rec_index'][0],
        aatype=item['target']['rec_aatype'][0],
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0)
    print(connection_violations)

    atomtype_radius = np.stack([residue_constants.restype_name_to_atom14_atom_radius[residue_constants.restype_1to3[residue_constants.restypes[aatype]]] for aatype in item['target']['rec_aatype'][0]])
    atomtype_radius = torch.from_numpy(atomtype_radius).to(device=item['target']['rec_aatype'].device, dtype=item['target']['rec_1d'].dtype)
    atom14_atom_radius = item['target']['rec_atom14_has_coords'][0] * atomtype_radius

    # Compute the between residue clash loss.
    between_residue_clashes = calc_between_residue_clash_loss(
        atom14_pred_positions=item['target']['rec_atom14_coords'][0],
        atom14_atom_exists=item['target']['rec_atom14_has_coords'][0],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=item['target']['rec_index'][0],
        overlap_tolerance_soft=1.5,
        overlap_tolerance_hard=1.5)
    print(between_residue_clashes)
    #with torch.cuda.amp.autocast():
    #with torch.autograd.set_detect_anomaly(True):