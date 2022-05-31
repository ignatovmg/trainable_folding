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


import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import sys
from copy import copy

from alphadock import modules
from alphadock import structure
from alphadock import all_atom
from alphadock import loss
from alphadock import utils


class DockerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InputEmbedder = modules.InputEmbedder(config['InputEmbedder'], global_config)
        self.Evoformer = nn.ModuleList([modules.EvoformerIteration(config['Evoformer']['EvoformerIteration'], global_config) for _ in range(config['Evoformer']['num_iter'])])
        self.EvoformerExtractSingle = nn.Linear(global_config['model']['rep1d_feat'], global_config['model']['single_rep_feat'])
        self.StructureModule = structure.StructureModule(config['StructureModule'], global_config)

        if config['msa_bert_block']:
            self.MSA_BERT = nn.Linear(global_config['model']['rep1d_feat'], 23)

        self.config = config
        self.global_config = global_config

        def nan_hook(self, input, output):
            if any([torch.any(torch.isnan(x)) for x in output]):
                print(f'Module {self.man_name} generated nans')
                print('Inputs contains nan: ', [torch.any(torch.isnan(x)) for x in input])
                print('Output contains nan: ', [torch.any(torch.isnan(x)) for x in output])
                print('Inputs were: ', input)
                print('Outputs were: ', output)
                sys.stdout.flush()
                raise utils.GeneratedNans(f'Module {self.man_name} generated nans')

        for name, module in self.Evoformer.named_modules():
            module.man_name = name
            module.register_forward_hook(nan_hook)

    def modules_to_devices(self):
        self.InputEmbedder.modules_to_devices()
        self.Evoformer.to(self.config['Evoformer']['device'])
        self.EvoformerExtractSingle.to(self.config['Evoformer']['device'])
        self.StructureModule.to(self.config['StructureModule']['device'])
        if self.config['msa_bert_block']:
            self.MSA_BERT.to(self.config['Evoformer']['device'])

    def forward(self, input, recycling=None):
        x = self.InputEmbedder(input, recycling=recycling)

        x['r1d'], x['pair'] = x['r1d'].to(self.config['Evoformer']['device']), x['pair'].to(self.config['Evoformer']['device'])

        for evo_i, evo_iter in enumerate(self.Evoformer):
            if self.config['Evoformer']['EvoformerIteration']['checkpoint']:
                x['r1d'], x['pair'] = checkpoint(evo_iter, x['r1d'], x['pair'])
            else:
                x['r1d'], x['pair'] = evo_iter(x['r1d'], x['pair'])

        pair = x['pair']
        rec_single = self.EvoformerExtractSingle(x['r1d'][:, 0])

        msa_bert = None
        if self.config['msa_bert_block'] and 'main_mask' in input['msa']:
            msa_bert = self.MSA_BERT(x['r1d'])

        input = {k: {k1: v1.to(self.config['StructureModule']['device']) for k1, v1 in v.items()} for k, v in input.items()}
        struct_out = self.StructureModule({
            'r1d': rec_single.to(self.config['StructureModule']['device']),
            'pair': pair.to(self.config['StructureModule']['device'])
        })

        # rescale to angstroms
        struct_out['rec_T'][..., -3:] = struct_out['rec_T'][..., -3:] * self.global_config['model']['position_scale']

        # compute all atom representation
        assert struct_out['rec_T'].shape[0] == 1
        final_all_atom = all_atom.backbone_affine_and_torsions_to_all_atom(
            struct_out['rec_T'][0][-1].clone(),
            struct_out['rec_torsions'][0][-1],
            input['target']['rec_aatype'][0]
        )

        out_dict = {}
        out_dict['struct_out'] = struct_out
        out_dict['final_all_atom'] = final_all_atom

        # compute loss
        if self.global_config['loss']['compute_loss']:
            out_dict['loss'] = loss.total_loss(input, struct_out, final_all_atom, self.global_config, msa_bert=msa_bert)

        # make recycling input
        cbeta_coords, cbeta_mask = all_atom.atom14_to_cbeta_coords(
            final_all_atom['atom_pos_tensor'],
            input['target']['rec_atom14_atom_exists'][0],
            input['target']['rec_aatype'][0]
        )
        out_dict['recycling_input'] = {
            'rec_1d_prev': x['r1d'][:, 0],
            'rep_2d_prev': pair,
            'rec_cbeta_prev': cbeta_coords[None],
            'rec_mask_prev': cbeta_mask[None]
        }

        return out_dict


if __name__ == '__main__':
    pass
