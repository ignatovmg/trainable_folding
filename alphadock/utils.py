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


import os
import subprocess
import json
import prody
import contextlib
import tempfile
import shutil
import numpy as np
from io import StringIO
from path import Path
from copy import deepcopy
import torch
import torch.nn.functional as F

from Bio import BiopythonDeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonDeprecationWarning)
    import Bio
    from Bio.SubsMat import MatrixInfo as matlist
    from Bio.pairwise2 import format_alignment


class GeneratedNans(Exception):
    pass


@contextlib.contextmanager
def isolated_filesystem(dir=None, remove=True):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    if dir is None:
        t = tempfile.mkdtemp(prefix='pocketdock-')
    else:
        t = dir
    os.chdir(t)
    try:
        yield t
    except Exception as e:
        #logger.error(f'Error occured, temporary files are in {t}')
        raise
    else:
        os.chdir(cwd)
        if remove:
            try:
                shutil.rmtree(t)
            except (OSError, IOError):
                pass
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def cwd(dir):
    pwd = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(pwd)


def tmp_file(**kwargs):
    handle, fname = tempfile.mkstemp(**kwargs)
    os.close(handle)
    return Path(fname)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def numpy_to_list(arr):
    return [x.item() for x in arr]


def rank_list(l):
    return zip(*sorted(enumerate(np.argsort(l)), key=lambda x: x[1]))


def safe_read_ag(ag) -> prody.Atomic:
    if isinstance(ag, prody.AtomGroup):
        return ag
    elif isinstance(ag, str):
        return prody.parsePDB(ag)
    else:
        raise RuntimeError(f"Can't read atom group, 'ag' has wrong type {type(ag)}")


def apply_prody_transform(coords, tr):
    return np.dot(coords, tr.getRotation().T) + tr.getTranslation()


def minimize_rmsd(mob_ag, ref_ag, mob_serials=None, ref_serials=None, mob_cset=None, ref_cset=None):
    if mob_serials is not None and ref_serials is not None:
        mob_sel = mob_ag.select('serial ' + ' '.join(map(str, mob_serials)))
        ref_sel = ref_ag.select('serial ' + ' '.join(map(str, ref_serials)))
        mob_s2i = dict(zip(mob_sel.getSerials(), mob_sel.getIndices()))
        ref_s2i = dict(zip(ref_sel.getSerials(), ref_sel.getIndices()))
        mob_ids = [mob_s2i[s] for s in mob_serials]
        ref_ids = [ref_s2i[s] for s in ref_serials]
    else:
        mob_ids = mob_ag.all.getIndices()
        ref_ids = ref_ag.all.getIndices()

    if mob_cset is not None:
        mob_crd = mob_ag.getCoordsets(mob_cset)[mob_ids]
    else:
        mob_crd = mob_ag.getCoords()[mob_ids]

    if ref_cset is not None:
        ref_crd = ref_ag.getCoordsets(ref_cset)[ref_ids]
    else:
        ref_crd = ref_ag.getCoords()[ref_ids]

    tr = prody.calcTransformation(mob_crd, ref_crd)
    rmsd_minimized = prody.calcRMSD(apply_prody_transform(mob_crd, tr), ref_crd)
    transformation = numpy_to_list(tr.getMatrix().flatten())
    return rmsd_minimized, transformation


def global_align(s1, s2):
    aln = Bio.pairwise2.align.globalds(s1, s2, matlist.blosum62, -14.0, -4.0)
    return aln


def calc_d2mat(crd1, crd2):
    return np.square(crd1[:, None, :] - crd2[None, :, :]).sum(2)


def calc_dmat(crd1, crd2):
    return np.sqrt(calc_d2mat(crd1, crd2))


def squared_difference(x, y):
    return torch.square(x - y)


def dmat_to_dgram(dmat, dmin, dmax, num_bins):
    shape = dmat.shape
    dmat = dmat.flatten()
    bin_size = (dmax - dmin) / num_bins
    bin_ids = torch.minimum(torch.div(F.relu(dmat - dmin), bin_size, rounding_mode='floor').to(int), torch.tensor(num_bins - 1, dtype=int, device=dmat.device))

    dgram = torch.zeros((len(dmat), num_bins), dtype=dmat.dtype, device=dmat.device)
    dgram[range(dgram.shape[0]), bin_ids] = 1.0
    dgram = dgram.reshape(*shape, num_bins)
    return bin_ids, dgram


def merge_dicts(a, b, strict=True, compare_types=False, _path=None):
    "merges b into a"
    if _path is None: _path = []
    for key in b:
        key_full_path = '.'.join(_path + [str(key)])
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], strict=strict, compare_types=compare_types, _path=_path + [str(key)])
            elif not compare_types or isinstance(a[key], type(b[key])):
                a[key] = b[key]
            else:
                raise RuntimeError('Conflict at "%s"' % key_full_path)
        elif strict:
            raise RuntimeError(f'Key "{key_full_path}" is not present in target')
        else:
            a[key] = b[key]
    return a


