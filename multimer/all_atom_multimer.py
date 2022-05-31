from alphadock import residue_constants, utils
import numpy as np
import torch
from multimer.rigid import Rigid, Rotation

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
    print(ranges)
    return data[ranges]


def _make_restype_atom14_mask():
  """Mask of which atoms are present for which residue type in atom14."""
  restype_atom14_mask = []

  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]
    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

  restype_atom14_mask.append([0.] * 14)
  restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)
  return restype_atom14_mask

def _make_restype_atom37_mask():
  """Mask of which atoms are present for which residue type in atom37."""
  # create the corresponding mask
  restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
  for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = residue_constants.atom_order[atom_name]
      restype_atom37_mask[restype, atom_type] = 1
  return restype_atom37_mask

def _make_restype_atom37_to_atom14():
  """Map from atom37 to atom14 per residue type."""
  restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]
    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

RESTYPE_ATOM14_MASK = _make_restype_atom14_mask()
RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()

def get_atom14_mask(aatype):
  return batched_gather(torch.tensor(RESTYPE_ATOM14_MASK), aatype)

def get_atom37_mask(aatype):
  return batched_gather(torch.tensor(RESTYPE_ATOM37_MASK), aatype)

def get_atom37_to_atom14_map(aatype):
  return batched_gather(torch.tensor(RESTYPE_ATOM37_TO_ATOM14), aatype)

def atom14_to_atom37(atom14_data, aatype):
    assert atom14_data.shape[1] == 14
    assert aatype.ndim == 1
    assert aatype.shape[0] == atom14_data.shape[0]

    residx_atom37_to_atom14 = torch.tensor(residue_constants.restype_name_to_atom14_ids, device=aatype.device, dtype=aatype.dtype)[aatype]
    atom14_data_flat = atom14_data.reshape(*atom14_data.shape[:2], -1)
    # add 15th field used as placeholder in restype_name_to_atom14_ids
    atom14_data_flat = torch.cat([atom14_data_flat, torch.zeros_like(atom14_data_flat[:, :1])], dim=1)
    out = torch.gather(atom14_data_flat, 1, residx_atom37_to_atom14[..., None].repeat(1, 1, atom14_data_flat.shape[-1]))
    return out.reshape(atom14_data.shape[0], 37, *atom14_data.shape[2:])


def atom_37_mask(aatype):
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=aatype.device
    )
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[aatype]
    return residx_atom37_mask

def torsion_angles_to_frames(rigid, angle, aatype):
    m = torch.tensor(
                residue_constants.restype_rigid_group_default_frame,
                dtype=angle.dtype,
                device=angle.device,
                requires_grad=False,
            )
    default_frames = m[aatype.long(), ...]
    default_rot = rigid.from_tensor_4x4(default_frames)
    backbone_rot = angle.new_zeros((*((1,) * len(angle.shape[:-1])), 2))
    backbone_rot[..., 1] = 1
    angle = torch.cat([backbone_rot.expand(*angle.shape[:-2], -1, -1), angle], dim=-2)
    all_rots = angle.new_zeros(default_rot.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = angle[..., 1]
    all_rots[..., 1, 2] = -angle[..., 0]
    all_rots[..., 2, 1:] = angle
    all_rots = Rigid(Rotation(rot_mats=all_rots), None)
    all_frames = default_rot.compose(all_rots)
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = rigid[..., None].compose(all_frames_to_bb)
    return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(aatype,all_frames_to_global):
    residx_to_group_idx = torch.tensor(
                residue_constants.restype_atom14_to_rigid_group,
                device=all_frames_to_global.get_rots().device,
                requires_grad=False,
            )
    group_mask = residx_to_group_idx[aatype.long(), ...]
    group_mask = torch.nn.functional.one_hot(group_mask, num_classes=8)
    map_atoms_to_global = all_frames_to_global[..., None, :] * group_mask

    map_atoms_to_global = map_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    lit_positions = torch.tensor(
                residue_constants.restype_atom14_rigid_group_positions,
                dtype=all_frames_to_global.get_rots().dtype,
                device=all_frames_to_global.get_rots().device,
                requires_grad=False,
            )
    lit_positions = lit_positions[aatype.long(), ...]

    mask = torch.tensor(
                residue_constants.restype_atom14_mask,
                dtype=all_frames_to_global.get_rots().dtype,
                device=all_frames_to_global.get_rots().device,
                requires_grad=False,
            )
    mask = mask[aatype.long(), ...].unsqueeze(-1)
    pred_positions = map_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * mask
    return pred_positions

