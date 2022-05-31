# Copyright © 2022 Applied BioComputation Group, Stony Brook University
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


import itertools
import numpy as np
import prody
from collections import OrderedDict, defaultdict, Counter
import typing
import torch
import time
from tqdm import tqdm
import traceback
from gemmi import cif
from path import Path

from alphadock import utils
from alphadock import residue_constants
from alphadock import all_atom
from alphadock import r3
from alphadock import quat_affine
from alphadock.config import DTYPE_FLOAT, DTYPE_INT


AATYPE = residue_constants.restype_order
AATYPE_WITH_X = residue_constants.restype_order_with_x

AATYPE_WITH_X_AND_GAP = AATYPE_WITH_X.copy()
AATYPE_WITH_X_AND_GAP['-'] = len(AATYPE_WITH_X)

HHBLITS_WITH_X_AND_GAP = AATYPE_WITH_X_AND_GAP.copy()
HHBLITS_WITH_X_AND_GAP['B'] = HHBLITS_WITH_X_AND_GAP['D']
HHBLITS_WITH_X_AND_GAP['J'] = HHBLITS_WITH_X_AND_GAP['X']
HHBLITS_WITH_X_AND_GAP['O'] = HHBLITS_WITH_X_AND_GAP['X']
HHBLITS_WITH_X_AND_GAP['U'] = HHBLITS_WITH_X_AND_GAP['C']
HHBLITS_WITH_X_AND_GAP['Z'] = HHBLITS_WITH_X_AND_GAP['E']


def residue_to_atom14(residue):
    resname_converted = residue.getResnames()[0].upper() if residue is not None else 'UNK'
    if resname_converted not in residue_constants.restype_name_to_atom14_names:
        resname_converted = 'UNK'

    ideal_names = residue_constants.restype_name_to_atom14_names[resname_converted]
    coords = np.zeros((len(ideal_names), 3), dtype=DTYPE_FLOAT)
    has_coords = np.zeros(len(ideal_names), dtype=DTYPE_FLOAT)

    if residue is not None:
        residue = residue.heavy.copy()
        atom2crd = {a.getName(): a.getCoords() for a in residue}
        for i, a in enumerate(ideal_names):
            coords[i] = atom2crd.get(a, np.zeros(3))
            has_coords[i] = a in atom2crd
        res_names = [x.getName() for x in residue if x.getName() not in ['OXT']]
        #print(sorted(res_names))
        #print(sorted([x for x in ideal_names if x]))
        if resname_converted != 'UNK' and not set(res_names).issubset(set(ideal_names)):
            print('Warning: atoms in ', residue, 'have non-conventional names', set(res_names) - set(ideal_names))
        #assert set(res_names).issubset(set(ideal_names)), (ideal_names, res_names)

    return coords, has_coords


def loop_to_list(block, category):
    cat = block.find_mmcif_category(category)
    assert len(cat) > 0, (category, 'does not exist')
    out = []
    for row in cat:
        row_dict = OrderedDict()
        for key in cat.tags:
            row_dict[key] = row[key[cat.prefix_length:]]
        out.append(row_dict)
    return out


def cbeta_atom(residue):
    if residue is None:
        return None
    name = 'CA' if residue.getResnames()[0] == 'GLY' else 'CB'
    cbeta = residue.select('name ' + name)
    if cbeta is not None:
        return cbeta[0]
    return None


def cif_parse(cif_file, asym_id):
    parsed = cif.read_file(cif_file)
    block = parsed.sole_block()
    keys = ['_pdbx_poly_seq_scheme.asym_id',
            '_pdbx_poly_seq_scheme.entity_id',
            '_pdbx_poly_seq_scheme.seq_id',
            '_pdbx_poly_seq_scheme.mon_id',
            '_pdbx_poly_seq_scheme.ndb_seq_num',
            '_pdbx_poly_seq_scheme.pdb_seq_num',
            '_pdbx_poly_seq_scheme.auth_seq_num',
            '_pdbx_poly_seq_scheme.pdb_mon_id',
            '_pdbx_poly_seq_scheme.auth_mon_id',
            '_pdbx_poly_seq_scheme.pdb_strand_id',
            '_pdbx_poly_seq_scheme.pdb_ins_code',
            '_pdbx_poly_seq_scheme.hetero'
            ]
    pdbx_poly_seq_scheme = loop_to_list(block, '_pdbx_poly_seq_scheme')
    pdbx_poly_seq_scheme = [x for x in pdbx_poly_seq_scheme if x['_pdbx_poly_seq_scheme.asym_id'] == asym_id]
    assert len(pdbx_poly_seq_scheme) > 0, f'File {cif_file} does not have chain with asym_id "{asym_id}"'

    # alt locations have the same residue number,
    # keep only the first one to match the pdbx_seq_one_letter_code_can string
    seen_ids = []
    buf = []
    for item in pdbx_poly_seq_scheme:
        if item['_pdbx_poly_seq_scheme.seq_id'] not in seen_ids:
            seen_ids.append(item['_pdbx_poly_seq_scheme.seq_id'])
            buf.append(item)
    pdbx_poly_seq_scheme = buf

    '_struct_asym.id'
    '_struct_asym.entity_id'

    entity_id = set([x['_pdbx_poly_seq_scheme.entity_id'] for x in pdbx_poly_seq_scheme])
    assert len(entity_id) == 1, (cif_file, entity_id)
    entity_id = entity_id.pop()

    # '_entity_poly_seq.entity_id'
    # '_entity_poly_seq.num'
    # '_entity_poly_seq.mon_id'
    # '_entity_poly_seq.hetero'
    entity_poly_seq = loop_to_list(block, '_entity_poly_seq')
    entity_poly_seq = [x for x in entity_poly_seq if x['_entity_poly_seq.entity_id'] == entity_id]

    # alt locations have the same residue number,
    # keep only the first one to match the pdbx_seq_one_letter_code_can string
    seen_ids = []
    buf = []
    for item in entity_poly_seq:
        if item['_entity_poly_seq.num'] not in seen_ids:
            seen_ids.append(item['_entity_poly_seq.num'])
            buf.append(item)
    entity_poly_seq = buf

    entity_poly_seq_str = ''.join([residue_constants.restype_3to1.get(x['_entity_poly_seq.mon_id'], 'X') for x in entity_poly_seq])
    #print(entity_poly_seq_str)

    # '_entity_poly.entity_id'
    # '_entity_poly.pdbx_strand_id'
    # '_entity_poly.pdbx_seq_one_letter_code'
    # '_entity_poly.pdbx_seq_one_letter_code_can'
    entity_poly = loop_to_list(block, '_entity_poly')
    entity_poly = [x for x in entity_poly if x['_entity_poly.entity_id'] == entity_id]
    assert len(entity_poly) == 1, entity_poly
    pdbx_seq_one_letter_code_can = entity_poly[0]['_entity_poly.pdbx_seq_one_letter_code_can'].replace('\n', '').replace(';', '')

    # replace U and O
    pdbx_seq_one_letter_code_can = pdbx_seq_one_letter_code_can.replace('U', 'X').replace('O', 'X')

    #tmp = utils.global_align(pdbx_seq_one_letter_code_can, entity_poly_seq_str)[0][:2]
    #print(tmp[0])
    #print(tmp[1])
    #print(len(pdbx_poly_seq_scheme))

    assert len(pdbx_seq_one_letter_code_can) == len(entity_poly_seq), (len(pdbx_seq_one_letter_code_can), len(entity_poly_seq))
    assert len(pdbx_seq_one_letter_code_can) == len(pdbx_poly_seq_scheme), (len(pdbx_seq_one_letter_code_can), len(pdbx_poly_seq_scheme))

    '_chem_comp.id'
    '_chem_comp.mon_nstd_parent_comp_id'

    ag = prody.parseMMCIF(cif_file)
    res_list = [(x.getChid(), str(x.getResnum()) + x.getIcode(), x.getResname())
                for x in ag.getHierView().iterResidues()]
    assert all([x == 1 for x in Counter(res_list).values()]), Counter(res_list)
    res_dicts = []
    for x in pdbx_poly_seq_scheme:
        ins_code = '' if x['_pdbx_poly_seq_scheme.pdb_ins_code'] == '.' else x['_pdbx_poly_seq_scheme.pdb_ins_code']
        selstr = 'chain ' + x['_pdbx_poly_seq_scheme.pdb_strand_id'] + \
                 ' and resnum `' + x['_pdbx_poly_seq_scheme.pdb_seq_num'] + ins_code + '`' + \
                 ' and resname ' + x['_pdbx_poly_seq_scheme.mon_id']
        res_ag = ag.select(selstr)
        #cbeta_ag = cbeta_atom(res_ag)
        res_dict = OrderedDict({x.split('.')[-1]: y for x, y in x.items()})
        #res_dict['selstr'] = selstr
        #res_dict['aatype_strict'] = residue_constants.restype_3to1.get(res_dict['mon_id'], 'X')
        res_dict['aatype_can'] = pdbx_seq_one_letter_code_can[int(res_dict['seq_id']) - 1]
        #res_dict['has_coords'] = res_ag is not None
        res_dict['has_frame'] = (res_ag is not None) and ('CA' in res_ag) and ('C' in res_ag) and ('N' in res_ag)
        #res_dict['has_cb'] = cbeta_ag is not None
        #res_dict['crd_cb'] = None if cbeta_ag is None else cbeta_ag.getCoords()
        res_dict['atom14_coords'], res_dict['atom14_mask'] = residue_to_atom14(res_ag)
        res_dicts.append(res_dict)

    return res_dicts


def dmat_to_distogram(dmat, dmin, dmax, num_bins):
    shape = dmat.shape
    dmat = dmat.flatten()
    bin_size = (dmax - dmin) / num_bins
    bin_ids = np.minimum(np.maximum(((dmat - dmin) // bin_size).astype(int), 0), num_bins - 1)

    dgram = np.zeros((len(dmat), num_bins), dtype=DTYPE_FLOAT)
    dgram[np.arange(dgram.shape[0]), bin_ids] = 1.0
    dgram = dgram.reshape(*shape, num_bins)
    return dgram


def one_hot_aatype(aa, alphabet):
    out = np.zeros(len(alphabet), dtype=DTYPE_FLOAT)
    out[alphabet[aa]] = 1
    return out


def target_sequence_featurize(sequence, map_unknown_to_x=True, crop_range=None, af_compatible=True, relpos_max=32):
    if map_unknown_to_x:
        aatype_int = np.array([AATYPE_WITH_X.get(x.upper(), AATYPE_WITH_X['X']) for x in sequence], dtype=DTYPE_INT)
    else:
        aatype_int = np.array([AATYPE_WITH_X[x.upper()] for x in sequence], dtype=DTYPE_INT)

    aatype_onehot = np.zeros((len(aatype_int), len(AATYPE_WITH_X)), dtype=DTYPE_FLOAT)
    aatype_onehot[range(len(aatype_int)), aatype_int] = 1

    # for compatibility with AF parameters, they have domain break flag first
    if af_compatible:
        aatype_onehot = np.pad(aatype_onehot, [(0, 0), (1, 0)])

    relpos_2d = np.arange(len(aatype_int))
    relpos_2d = relpos_2d[:, None] - relpos_2d[None, :]
    relpos_2d = dmat_to_distogram(relpos_2d, -relpos_max, relpos_max + 1, relpos_max * 2 + 1)

    target = {
        'rec_1d': aatype_onehot.astype(DTYPE_FLOAT),
        'rec_relpos': relpos_2d.astype(DTYPE_FLOAT),
        'rec_aatype': aatype_int.astype(DTYPE_INT),
        'rec_index': np.arange(len(aatype_int), dtype=DTYPE_INT),
        'rec_atom14_atom_exists': residue_constants.restype_atom14_mask[aatype_int]
    }

    if crop_range is not None:
        target = {k: v[crop_range[0]:crop_range[1]] for k, v in target.items()}
        target['rec_relpos'] = target['rec_relpos'][:, crop_range[0]:crop_range[1]]

    return target


def cif_featurize(cif_file, asym_id, crop_range=None, use_cache=True):
    cache_path = cif_file + '_' + asym_id + '_cache.npy'
    if use_cache and Path(cache_path).exists():
        res_dicts = np.load(cache_path, allow_pickle=True)
    else:
        res_dicts = cif_parse(cif_file, asym_id)
        if use_cache:
            np.save(cache_path, res_dicts)

    aatype_int = np.array([AATYPE_WITH_X.get(x['aatype_can'].upper(), AATYPE_WITH_X['X']) for x in res_dicts], dtype=DTYPE_INT)

    #atom14_gt_positions_rigids = r3.Vecs(*[x.squeeze(-1) for x in np.split(rec_dict['rec_atom14_coords'], 3, axis=-1)])
    atom14_gt_positions = np.stack([x['atom14_coords'] for x in res_dicts])  # (N, 14, 3)
    atom14_gt_exists = np.stack([x['atom14_mask'] for x in res_dicts])  # (N, 14)
    renaming_mats = all_atom.RENAMING_MATRICES[aatype_int]  # (N, 14, 14)
    atom14_alt_gt_positions = np.sum(atom14_gt_positions[:, :, None, :] * renaming_mats[:, :, :, None], axis=1)
    atom14_alt_gt_exists = np.sum(atom14_gt_exists[:, :, None] * renaming_mats, axis=1)
    atom14_atom_is_ambiguous = (renaming_mats * np.eye(14)[None]).sum(1) == 0

    atom37_gt_positions = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_positions).float(), torch.from_numpy(aatype_int))
    atom37_gt_exists = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_exists).float(), torch.from_numpy(aatype_int))
    gt_torsions = all_atom.atom37_to_torsion_angles(torch.from_numpy(aatype_int[None]), atom37_gt_positions[None].float(), atom37_gt_exists[None].float())

    rec_all_frames = all_atom.atom37_to_frames(torch.from_numpy(aatype_int), atom37_gt_positions.float(), atom37_gt_exists.float())
    rec_bb_affine = r3.rigids_to_quataffine(r3.rigids_from_tensor_flat12(rec_all_frames['rigidgroups_gt_frames'][..., 0, :]))
    rec_bb_affine.quaternion = quat_affine.rot_to_quat(rec_bb_affine.rotation)
    rec_bb_affine = rec_bb_affine.to_tensor().numpy()
    rec_bb_affine_mask = rec_all_frames['rigidgroups_gt_exists'][..., 0].numpy()

    ground_truth = {
        'gt_aatype': aatype_int.astype(DTYPE_INT),  # same as for target
        'gt_atom14_coords': atom14_gt_positions.astype(DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords': atom14_gt_exists.astype(DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_coords_alt': atom14_alt_gt_positions.astype(DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords_alt': atom14_alt_gt_exists.astype(DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_atom_is_ambiguous': atom14_atom_is_ambiguous.astype(DTYPE_FLOAT),

        'gt_torsions_sin_cos': gt_torsions['torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'gt_torsions_sin_cos_alt': gt_torsions['alt_torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'gt_torsions_mask': gt_torsions['torsion_angles_mask'][0].numpy().astype(DTYPE_FLOAT),

        'gt_rigidgroups_gt_frames': rec_all_frames['rigidgroups_gt_frames'].numpy().astype(DTYPE_FLOAT),  # (..., 8, 12)
        'gt_rigidgroups_alt_gt_frames': rec_all_frames['rigidgroups_alt_gt_frames'].numpy().astype(DTYPE_FLOAT),  # (..., 8, 12)
        'gt_rigidgroups_gt_exists': rec_all_frames['rigidgroups_gt_exists'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        #'gt_rigidgroups_group_exists': rec_all_frames['rigidgroups_group_exists'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        'gt_rigidgroups_group_is_ambiguous': rec_all_frames['rigidgroups_group_is_ambiguous'].numpy().astype(DTYPE_FLOAT),  # (..., 8)

        'gt_bb_affine': rec_bb_affine.astype(DTYPE_FLOAT),
        'gt_bb_affine_mask': rec_bb_affine_mask.astype(DTYPE_FLOAT),

        'gt_residue_index': np.arange(len(aatype_int), dtype=DTYPE_INT),  # (N_res)
        'gt_has_frame': np.array([x['has_frame'] for x in res_dicts]).astype(DTYPE_FLOAT),  # (N_res)
    }

    if crop_range is not None:
        ground_truth = {k: v[crop_range[0]:crop_range[1]] for k, v in ground_truth.items()}

    return ground_truth


def parse_a3m(a3m_file):
    with open(a3m_file, 'r') as f:
        ref_name = f.readline()[1:].strip()
        ref_seq = f.readline().strip()
        msa = [x.strip() for x in f.readlines()[1::2]]
    return ref_name, [ref_seq] + msa


def msas_numeric_to_onehot(msa_npy, size):
    msa_num = msa_npy.flatten()
    msa_onehot = np.zeros((msa_num.size, size), dtype=DTYPE_FLOAT)
    msa_onehot[range(msa_num.size), msa_num] = 1
    return msa_onehot.reshape((msa_npy.shape[0], msa_npy.shape[1], size))


def calc_deletion_matrix(msa):
    out = []
    for seq in msa:
        row = []
        count = 0
        for aa in seq:
            if aa.islower():
                #print(aa)
                count += 1
            else:
                row.append(count)
                count = 0
        out.append(row)
    return np.stack(out).astype(np.ushort)  # 16-bit [0, 65535]


def msa_generate_random(probs, seed):
    # unofficial way of seeding pytorch locally
    # https://discuss.pytorch.org/t/is-there-a-randomstate-equivalent-in-pytorch-for-local-random-generator-seeding/37131/2
    #
    # we can use pure numpy to do this starting 1.22 using np.random.multinomial,
    # but current numpy version is older
    rng = torch.Generator()
    rng = rng.manual_seed(seed)
    probs_flat = probs.reshape([-1, probs.shape[-1]])
    result = torch.multinomial(torch.from_numpy(probs_flat), 1, generator=rng).squeeze(-1).numpy()
    return result.reshape(probs.shape[:-1])


def msa_featurize(
        a3m_files,
        rng,
        num_clusters,
        num_extra,
        use_cache=True,
        crop_range=None,
        num_block_del=5,
        block_del_size=0.3,
        random_replace_fraction=0.15,
        uniform_prob=0.1,
        profile_prob=0.1,
        same_prob=0.1,
        keep_true_msa=True
):
    assert num_clusters > 0, num_clusters
    assert num_extra >= 0, num_extra

    # if cached msa don't exist create them, otherwise load from disk
    cache_prefix = a3m_files[0] + '_cache_'
    if not Path(cache_prefix + 'msa.npy').exists() or not use_cache:
        msa = []
        for a3m_file in a3m_files:
            msa += parse_a3m(a3m_file)[1]

        # remove duplicates but keep the original order
        msa = [seq for seq, idx in sorted(dict(reversed([(b, a) for a, b in enumerate(msa)])).items(), key=lambda x: x[1])]

        # calculate del matrix
        all_msa_del_mat = calc_deletion_matrix(msa)

        # remove insertions
        msa = [''.join([aa for aa in seq if aa.isupper() or aa == '-']) for seq in msa]

        # msa to numbers
        assert all([x != '-' for x in msa[0]]), msa[0]
        _fun = np.vectorize(lambda x: HHBLITS_WITH_X_AND_GAP.get(x, HHBLITS_WITH_X_AND_GAP['X']), otypes=[np.byte])
        all_msa_npy = _fun(np.stack([list(x) for i, x in enumerate(msa)]))

        # save converted msa to cache
        if use_cache:
            np.save(cache_prefix + 'del.npy', all_msa_del_mat)
            np.save(cache_prefix + 'msa.npy', all_msa_npy)
    else:
        # load from cache
        all_msa_del_mat = np.load(cache_prefix + 'del.npy')
        all_msa_npy = np.load(cache_prefix + 'msa.npy')

    msa_size = all_msa_npy.shape[0]

    # block deletion like in AF
    if num_block_del > 0 and block_del_size > 0 and msa_size > 1:
        del_rows = set()
        block_del_size = int(block_del_size * msa_size)
        for block_start in rng.integers(1, msa_size, num_block_del):
            del_rows |= set(range(block_start, min(block_start + block_del_size, msa_size)))
        keep_mask = np.ones(msa_size, dtype=np.bool)
        keep_mask[list(del_rows)] = False
        all_msa_del_mat = all_msa_del_mat[keep_mask]
        all_msa_npy = all_msa_npy[keep_mask]

    msa_size = all_msa_npy.shape[0]

    # replace pyrrolysine and selenocysteine
    # update: turns out U and O are already replaced by X in the msas
    #all_msa_npy = np.char.replace(all_msa_npy, 'U', 'C')
    #all_msa_npy = np.char.replace(all_msa_npy, 'O', 'X')

    # featurize
    all_msa_onehot = msas_numeric_to_onehot(all_msa_npy, size=len(AATYPE_WITH_X_AND_GAP) + 1)
    #print(all_msa_onehot.shape, all_msa_onehot.dtype)

    # select cluster centers
    _buf = np.arange(1, msa_size, dtype=int)
    rng.shuffle(_buf)
    msa_shuffled_ids = np.concatenate([np.array([0], dtype=int), _buf])
    main_ids = msa_shuffled_ids[:num_clusters]
    main_msa_npy = all_msa_npy[main_ids]

    # random replacement
    # keep even if random_replace_fraction is 0.0 to prevent rng disruption
    #if random_replace_fraction > 0.0:
    probs = uniform_prob * np.array([0.05] * 20 + [0.0, 0.0, 0.0])[None, None] + \
            profile_prob * all_msa_onehot.mean(0)[None] + \
            same_prob * all_msa_onehot[main_ids]
    probs[..., -1] = 1. - uniform_prob - profile_prob - same_prob
    msa_replacements = msa_generate_random(probs, rng.integers(10e6).item())
    main_msa_mask = rng.random(msa_replacements.shape) < random_replace_fraction
    main_msa_true = main_msa_npy.copy()
    main_msa_npy = np.where(main_msa_mask, msa_replacements, main_msa_npy)
    all_msa_onehot[main_ids] = msas_numeric_to_onehot(main_msa_npy, size=len(AATYPE_WITH_X_AND_GAP) + 1)

    # cluster
    hamming_mask = (main_msa_npy[:, None, :] != AATYPE_WITH_X_AND_GAP['-']) * (all_msa_npy[None, :, :] != AATYPE_WITH_X_AND_GAP['-'])
    hamming_mask *= (main_msa_npy[:, None, :] != len(AATYPE_WITH_X_AND_GAP)) * (all_msa_npy[None, :, :] != len(AATYPE_WITH_X_AND_GAP))
    hamming_dist = ((main_msa_npy[:, None, :] != all_msa_npy[None, :, :]) * hamming_mask).sum(-1)  # (num_clusters, num_msa)
    closest_main_id = hamming_dist.argmin(0)
    closest_main_id[main_ids] = main_ids   # <-- make sure the centers are assigned to themselves
    clusters = [(x, np.where(x == closest_main_id)[0]) for x in main_ids]

    # featurize main part
    main_msa_onehot = all_msa_onehot[main_ids]
    main_msa_has_del = all_msa_del_mat[main_ids] > 0
    main_msa_del_value = np.arctan(all_msa_del_mat[main_ids] / 3) * 2 / np.pi
    main_msa_del_mean = np.arctan(np.stack([all_msa_del_mat[m].mean(0) for c, m in clusters]) / 3) * 2 / np.pi
    main_msa_clus_profile = np.stack([all_msa_onehot[m].mean(0) for c, m in clusters])

    out = {
        'main': np.concatenate([
            main_msa_onehot,
            main_msa_has_del[..., None],
            main_msa_del_value[..., None],
            main_msa_del_mean[..., None],
            main_msa_clus_profile
        ], axis=-1).astype(DTYPE_FLOAT)
    }

    if keep_true_msa:
        out['main_mask'] = main_msa_mask.astype(DTYPE_INT)
        out['main_true'] = main_msa_true.astype(DTYPE_INT)

    # featurize extra msa
    extra_ids = msa_shuffled_ids[num_clusters:num_clusters + num_extra]
    if len(extra_ids) > 0:
        extra_msa_onehot = all_msa_onehot[extra_ids]
        extra_msa_has_del = all_msa_del_mat[extra_ids] > 0
        extra_msa_del_value = np.arctan(all_msa_del_mat[extra_ids] / 3) * 2 / np.pi

        out['extra'] = np.concatenate([
            extra_msa_onehot,
            extra_msa_has_del[..., None],
            extra_msa_del_value[..., None]
        ], axis=-1).astype(DTYPE_FLOAT)

    if crop_range is not None:
        out['main'] = out['main'][:, crop_range[0]:crop_range[1]]
        if 'extra' in out:
            out['extra'] = out['extra'][:, crop_range[0]:crop_range[1]]
        if 'main_mask' in out:
            out['main_mask'] = out['main_mask'][:, crop_range[0]:crop_range[1]]
            out['main_true'] = out['main_true'][:, crop_range[0]:crop_range[1]]

    return out  #.replace('U', 'C').replace('O', 'X')

def template_pair(batch, use_template_unit_vector):
    num_res = batch['template_aatype'].shape[-1]
    num_batch = batch['template_aatype'].shape[0]
    num_temp = batch['template_aatype'].shape[1]
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    temp_pseudo = batch["template_pseudo_beta"]
    dgram = torch.sum(
        (temp_pseudo[..., None, :] - temp_pseudo[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower_breaks = torch.square(torch.linspace(3.25, 50.75, 39, device=temp_pseudo.device))
    upper_breaks = torch.cat([lower_breaks[1:], torch.tensor([1e8], device=temp_pseudo.device)], dim=-1)
    dgram = ((dgram > lower_breaks) * (dgram < upper_breaks)).type(dgram.dtype)
    to_concat = [dgram, template_mask_2d[..., None]]
    aatype = torch.nn.functional.one_hot(batch["template_aatype"].long(), 22)
    to_concat.append(aatype[..., None, :, :].expand(*aatype.shape[:-2], num_res, -1, -1))
    to_concat.append(aatype[..., None, :].expand(*aatype.shape[:-2], -1, num_res, -1))
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    
    rots, trans = quat_affine_multi.make_canonical_transform(n_xyz=batch["template_all_atom_positions"][..., n, :],
        ca_xyz=batch["template_all_atom_positions"][..., ca, :],
        c_xyz=batch["template_all_atom_positions"][..., c, :])
    points_multi = trans[..., None, :, :]
    trans_multi = trans[..., None,:]
    rots_multi = rots[..., None, :, :]
    rigid_vec_multi = quat_affine_multi.invert_point(points_multi, trans_multi, rots_multi)

    inv_distance_scalar = torch.rsqrt(1e-6 + torch.sum(rigid_vec_multi ** 2, dim=-1))
    template_mask = (
        batch['template_all_atom_masks'][..., n] *
        batch['template_all_atom_masks'][..., ca] *
        batch['template_all_atom_masks'][..., c])
    template_mask_2d = template_mask[..., None] * template_mask[...,None, :]
    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec_multi * inv_distance_scalar[..., None]
    if (not use_template_unit_vector):
        unit_vector = torch.zeros_like(unit_vector)
    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])
    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]
    return act

def example():
    inputs = cif_featurize('/data/trainable_folding/data_preparation/data/15k/folding/cifs/4gq2.cif', 'A')

    for k1, v1 in inputs.items():
        print(k1)
        for k2, v2 in v1.items():
            print('    ', k2, v1[k2].shape, v1[k2].dtype)


if __name__ == '__main__':
    #example()
    out = msa_featurize(['/data/trainable_folding/data_preparation/data/15k/folding/MMSEQ_submission_second_try/6eno_1.fa_results/uniref.a3m'], np.random.default_rng(123), 32, 32)
    for k1, v1 in out.items():
        print(k1, ':', v1.shape)

    from path import Path
    a3ms = sorted(Path('/data/trainable_folding/data_preparation/data/15k/folding/MMSEQ_submission_second_try').glob('*/uniref.a3m'))
    for a in a3ms:
        out = msa_featurize([a, a.dirname() / 'bfd.mgnify30.metaeuk30.smag30.a3m'], np.random.default_rng(123), 128, 4096)
        print(a)
        for k1, v1 in out.items():
            print(k1, ':', v1.shape)

