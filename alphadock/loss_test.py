import torch
import prody
import numpy as np
import pytest

from alphadock import all_atom
from alphadock import docker
from alphadock import config
from alphadock import features_summit
from alphadock import r3
from alphadock import quat_affine
from alphadock import residue_constants
from alphadock import features_summit
from alphadock import utils
from alphadock.config import DTYPE_FLOAT, DTYPE_INT
from alphadock import loss


def make_3ask_ground_truth():
    case_name = '3ASK_D'
    group_name = 'M3L'

    case_dir = config.TEST_DATA_DIR / case_name
    case_dict = utils.read_json(case_dir / 'case.json')
    group_dict = {x['name']: x for x in case_dict['ligand_groups']}[group_name]

    rec_ag = prody.parsePDB(case_dir / 'rec_orig.pdb')
    rec_dict = features_summit.ag_to_features(
        rec_ag,
        case_dict['entity_info']['pdb_aln'],
        case_dict['entity_info']['entity_aln'],
        no_mismatch=True
    )
    rec_dict = features_summit.rec_literal_to_numeric(rec_dict, seq_include_gap=False)

    atom14_gt_positions = rec_dict['atom14_coords']  # (N, 14, 3)
    atom14_gt_exists = rec_dict['atom14_has_coords']  # (N, 14)
    renaming_mats = all_atom.RENAMING_MATRICES[rec_dict['seq_aatype_num']]  # (N, 14, 14)
    atom14_alt_gt_positions = np.sum(atom14_gt_positions[:, :, None, :] * renaming_mats[:, :, :, None], axis=1)
    atom14_alt_gt_exists = np.sum(rec_dict['atom14_has_coords'][:, :, None] * renaming_mats, axis=1)
    atom14_atom_is_ambiguous = (renaming_mats * np.eye(14)[None]).sum(1) == 0

    atom37_gt_positions = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_positions).float(), torch.from_numpy(rec_dict['seq_aatype_num']))
    atom37_gt_exists = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_exists).float(), torch.from_numpy(rec_dict['seq_aatype_num']))
    gt_torsions = all_atom.atom37_to_torsion_angles(torch.from_numpy(rec_dict['seq_aatype_num'][None]), atom37_gt_positions[None].float(), atom37_gt_exists[None].float())

    rec_all_frames = all_atom.atom37_to_frames(torch.from_numpy(rec_dict['seq_aatype_num']), atom37_gt_positions.float(), atom37_gt_exists.float())
    rec_bb_affine = r3.rigids_to_quataffine(r3.rigids_from_tensor_flat12(rec_all_frames['rigidgroups_gt_frames'][..., 0, :]))
    rec_bb_affine.quaternion = quat_affine.rot_to_quat(rec_bb_affine.rotation)
    rec_bb_affine = rec_bb_affine.to_tensor().numpy()
    rec_bb_affine_mask = rec_all_frames['rigidgroups_gt_exists'][..., 0].numpy()

    #      (2)    (3)
    #       N ---- N
    #       |      |
    #  O -- C ---- C -- O
    # (0)  (1)    (4)  (5)
    #
    lig_coords = np.array([
        [[0, -2, 0], [0, 0, 0], [0, 2, 0], [0, 2, 2], [0, 0, 2], [10, 10, 10]],
        [[10, 10, 10], [0, 2, 2], [0, 0, 2], [0, 0, 0], [0, 2, 0], [0, -2, 0]]
    ], dtype=DTYPE_FLOAT)

    lig_has_coords = np.array([
        [1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1]
    ], dtype=DTYPE_FLOAT)

    out = {
        'gt_aatype': rec_dict['seq_aatype_num'].astype(DTYPE_INT),  # same as for target
        'gt_atom14_coords': atom14_gt_positions.astype(DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords': atom14_gt_exists.astype(DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_coords_alt': atom14_alt_gt_positions.astype(DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords_alt': atom14_alt_gt_exists.astype(DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_atom_is_ambiguous': atom14_atom_is_ambiguous.astype(DTYPE_FLOAT),

        'gt_torsions_sin_cos': gt_torsions['torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'gt_torsions_sin_cos_alt': gt_torsions['alt_torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'gt_torsions_mask': gt_torsions['torsion_angles_mask'][0].numpy().astype(DTYPE_FLOAT),

        'gt_rigidgroups_gt_frames': rec_all_frames['rigidgroups_gt_frames'].numpy().astype(DTYPE_FLOAT),  # (..., 8, 12)
        'gt_rigidgroups_gt_exists': rec_all_frames['rigidgroups_gt_exists'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        'gt_rigidgroups_group_exists': rec_all_frames['rigidgroups_group_exists'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        'gt_rigidgroups_group_is_ambiguous': rec_all_frames['rigidgroups_group_is_ambiguous'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        'gt_rigidgroups_alt_gt_frames': rec_all_frames['rigidgroups_alt_gt_frames'].numpy().astype(DTYPE_FLOAT),  # (..., 8, 12)

        'gt_bb_affine': rec_bb_affine.astype(DTYPE_FLOAT),
        'gt_bb_affine_mask': rec_bb_affine_mask.astype(DTYPE_FLOAT),

        'gt_residue_index': np.arange(len(rec_dict['seq_aatype_num']), dtype=DTYPE_INT),  # (N_res)
        'gt_has_frame': rec_dict['has_frame'].astype(DTYPE_FLOAT),  # (N_res)
        'gt_lig_coords': np.stack(lig_coords).astype(DTYPE_FLOAT),  # (N_symm, N_atoms, 3)
        'gt_lig_has_coords': np.stack(lig_has_coords).astype(DTYPE_FLOAT),  # (N_symm, N_atoms)
    }

    return {k: torch.from_numpy(v)[None] for k, v in out.items()}


def test_loss_torsions():
    gt = make_3ask_ground_truth()
    torsion_loss = loss.torsion_loss({'ground_truth': gt}, {'rec_torsions': gt['gt_torsions_sin_cos'][None]})
    assert torsion_loss['chi_loss'][0].item() == pytest.approx(0, abs=1e-5)
    assert torsion_loss['norm_loss'][0].item() == pytest.approx(0, abs=1e-5)


def test_loss_bb_rec_rec_1ask():
    gt = make_3ask_ground_truth()
    pred_bb_frames = r3.rigids_from_tensor_flat12(gt['gt_rigidgroups_gt_frames'][0, :, 0, :])
    gt_bb_frames = r3.rigids_from_tensor_flat12(gt['gt_rigidgroups_gt_frames'][0, :, 0, :])
    gt_bb_mask = gt['gt_rigidgroups_group_exists'][0, :, 0]

    fape = all_atom.frame_aligned_point_error(
        pred_bb_frames,
        gt_bb_frames,
        gt_bb_mask,
        pred_bb_frames.trans,
        gt_bb_frames.trans,
        gt_bb_mask,
        10.0,
        False
    )


def test_loss_identical():
    num_frames = 3
    frames = r3.rigids_from_tensor_flat12(torch.zeros((num_frames, 12)))
    frames.rot.xx[:] = 1
    frames.rot.yy[:] = 1
    frames.rot.zz[:] = 1
    frames.trans.x[:] = torch.tensor([0, 1, 2])
    frames.trans.y[:] = torch.tensor([0, 2, 3])
    frames.trans.z[:] = torch.tensor([0, 3, 4])

    gt_bb_mask = torch.ones(num_frames)

    fape = all_atom.frame_aligned_point_error(
        frames,
        frames,
        gt_bb_mask,
        frames.trans,
        frames.trans,
        gt_bb_mask,
        1.0,
        False
    )
    assert fape == pytest.approx(0.0, abs=1e-4)


def test_loss_shifted():
    num_frames = 3
    frames = r3.rigids_from_tensor_flat12(torch.zeros((num_frames, 12)))
    frames.rot.xx[:] = 1
    frames.rot.yy[:] = 1
    frames.rot.zz[:] = 1
    frames.trans.x[:] = torch.tensor([0, 1, 2])
    frames.trans.y[:] = torch.tensor([0, 2, 3])
    frames.trans.z[:] = torch.tensor([0, 3, 4])

    shifted_pos = r3.Vecs(frames.trans.x + 3, frames.trans.y + 4, frames.trans.z)

    gt_bb_mask = torch.ones(num_frames)

    fape = all_atom.frame_aligned_point_error(
        frames,
        frames,
        gt_bb_mask,
        frames.trans,
        shifted_pos,
        gt_bb_mask,
        1.0,
        False
    )
    assert fape == pytest.approx(5.0, abs=1e-3)


def test_lddt_3points_identical():
    pred_a = torch.tensor([[
        [0., 0., 0.],
        [1., 1., 1.],
        [2., 2., 2.]
    ]])
    pred_b = pred_a.clone()
    true_a = pred_a.clone()
    true_b = pred_a.clone()
    mask_a = torch.ones((1, 3, 1))
    mask_b = torch.ones((1, 3, 1))

    lddt_val = loss.lddt(pred_a, pred_b, true_a, true_b, mask_a, mask_b)
    assert lddt_val == pytest.approx(1.0, abs=1e-5)


def test_lddt_3points_shifted():
    pred_a = torch.tensor([[
        [0., 0., 0.],
        [1., 1., 1.],
        [2., 2., 2.]
    ]])
    pred_b = pred_a.clone()
    true_a = pred_a.clone()
    true_b = pred_a.clone()
    mask_a = torch.ones((1, 3, 1))
    mask_b = torch.ones((1, 3, 1))

    true_b[0, 0, 0] += 10
    lddt_val = loss.lddt(pred_a, pred_b, true_a, true_b, mask_a, mask_b)
    assert lddt_val == pytest.approx(2 / 3, abs=1e-5)

    true_b = pred_a.clone()
    true_b[0, 0, 2] = -1.5
    lddt_val = loss.lddt(pred_a, pred_b, true_a, true_b, mask_a, mask_b)
    assert lddt_val == pytest.approx(5 / 6, abs=1e-5)

    mask_b[0, 0] = 0
    lddt_val = loss.lddt(pred_a, pred_b, true_a, true_b, mask_a, mask_b)
    assert lddt_val == pytest.approx(1., abs=1e-5)


if __name__ == '__main__':
    test_lddt_3points_shifted()
