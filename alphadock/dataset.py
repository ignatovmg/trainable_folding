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


import traceback
import random
import itertools
import numpy as np
import prody
from collections import defaultdict, Counter
from path import Path
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from functools import partial
import torch
import sys
import time
import datetime
#import horovod.torch as hvd

from alphadock.config import DTYPE_FLOAT
from alphadock import utils
from alphadock import features_summit
from alphadock import residue_constants


class DockingDataset(Dataset):
    def __init__(
            self,
            data,
            config_data,
            dataset_dir='.',
            seed=123456,
            shuffle=False,
            sample_to_size=None
    ):
        self.dataset_dir = Path(dataset_dir).abspath()
        self.config = config_data
        self.data = data

        self.rng = np.random.default_rng(seed)

        if shuffle:
            self.rng.shuffle(self.data)

        #if sample_to_size is not None:
        #    probs = np.array([1. / x['seqclus_size'] for x in self.data])
        #    probs /= probs.sum()
        #    self.data = self.rng.choice(self.data, size=min(sample_to_size, len(self.data)), replace=False, p=probs)
        #    #self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def make_features(self, sequence, a3m_files, cif_file=None, asym_ids=None):
        if self.config['crop_size'] is not None:
            crop_start = self.rng.integers(0, max(1, len(sequence) - self.config['crop_size']))
            crop_range = [crop_start, crop_start + self.config['crop_size']]
        else:
            crop_range = None

        out_dict = {}
        out_dict['target'] = features_summit.target_sequence_featurize(
            sequence,
            crop_range=crop_range,
            af_compatible=self.config['target_af_compatible'],
            relpos_max=self.config['relpos_max']
        )

        clamp_fape = self.rng.random() < self.config['clamp_fape_prob']
        if cif_file is not None:
            out_dict['ground_truth'] = features_summit.cif_featurize(
                cif_file,
                asym_ids[0], # choose first asym id
                crop_range=crop_range,
                use_cache=self.config['use_cache']
            )
            assert len(out_dict['target']['rec_1d']) == len(out_dict['ground_truth']['gt_aatype']), \
                (len(out_dict['target']['rec_1d']), len(out_dict['ground_truth']['gt_aatype']))

            out_dict['ground_truth']['clamp_fape'] = torch.tensor(0)
            if clamp_fape:
                out_dict['ground_truth']['clamp_fape'] = torch.tensor(1)

        out_dict['msa'] = features_summit.msa_featurize(
            a3m_files,
            self.rng,
            self.config['msa_max_clusters'],
            self.config['msa_max_extra'],
            use_cache=self.config['use_cache'],
            crop_range=crop_range,
            num_block_del=self.config['msa_block_del_num'],
            block_del_size=self.config['msa_block_del_size'],
            random_replace_fraction=self.config['msa_random_replace_fraction'],
            uniform_prob=self.config['msa_uniform_prob'],
            profile_prob=self.config['msa_profile_prob'],
            same_prob=self.config['msa_same_prob']
        )

        #assert first_seq == seq
        assert out_dict['msa']['main'].shape[1] == out_dict['target']['rec_1d'].shape[0], \
            (out_dict['msa']['main'].shape[1], out_dict['target']['rec_1d'].shape[0], item)

        return out_dict

    def _get_item(self, ix):
        item = self.data[ix]
        #print('sample', ix, ':', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item['pdb_id'], item['entity_id']); sys.stdout.flush()

        out_dict = self.make_features(
            item['entity_info']['pdbx_seq_one_letter_code_can'],
            [self.dataset_dir / x for x in item['a3m_files']],
            self.dataset_dir / item['cif_file'] if item['cif_file'] is not None else None,
            item['entity_info']['asym_ids'] if item['cif_file'] is not None else None
        )
        out_dict['target']['ix'] = ix
        return out_dict

    def __getitem__(self, ix):
        return self._get_item(ix)


if __name__ == '__main__':
    import config
    import tqdm
    ds = DockingDataset(
        config.DATA_DIR,
        '15k/folding/debug_15k.json',
        shuffle=True,
        seed=122 * 100
    )
    for i, x in enumerate(ds.data):
        if x['pdb_id'] == '1jwy' and x['entity_id'] == '1':
            print(i)
            break

    ds[6788]
    exit(0)

    for item in tqdm.tqdm(ds):
        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None] #.cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)

