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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

from alphadock import utils
from alphadock import features_summit
from alphadock import all_atom

class RowAttentionWithPairBias(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        in_num_c = global_config['model']['rep1d_extra_feat'] if config['msa_extra_stack'] else global_config['model']['rep1d_feat']
        pair_rep_num_c = global_config['model']['rep2d_feat']

        self.norm = nn.LayerNorm(in_num_c)
        self.norm_2d = nn.LayerNorm(pair_rep_num_c)
        # self.qkv = nn.Linear(in_num_c, 3 * attn_num_c * num_heads, bias=False)
        self.q = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.k = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.v = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.x2d_project = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.final = nn.Linear(attn_num_c * num_heads, in_num_c)
        self.gate = nn.Linear(in_num_c, attn_num_c * num_heads)
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

    def forward(self, x1d, x2d):
        x1d = self.norm(x1d)
        x2d = self.norm_2d(x2d)
        bias = self.x2d_project(x2d)
        # bias = self.x2d_project(x2d).view(*x2d.shape[:-1], self.num_heads)
        bias = bias.permute(0, 3, 1, 2)
        # q, k, v = torch.chunk(self.qkv(x1d).view(*x1d.shape[:-1], self.attn_num_c, 3 * self.num_heads), 3, dim=-1)
        q = self.q(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c)
        k = self.k(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c)
        v = self.v(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c)
        factor = 1 / math.sqrt(self.attn_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k)
        weights = torch.softmax(aff + bias, dim=-1)
        gate = torch.sigmoid(self.gate(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c))
        
        out_1d = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v) * gate
        out_1d = self.final(out_1d.flatten(start_dim=-2))
        return out_1d


class MSAColumnAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        in_num_c = global_config['model']['rep1d_feat']

        self.norm = nn.LayerNorm(in_num_c)
        self.q = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.k = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.v = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        #self.qkv = nn.Linear(in_num_c, attn_num_c * num_heads * 3, bias=False)
        self.final = nn.Linear(attn_num_c * num_heads, in_num_c)
        self.gate = nn.Linear(in_num_c, attn_num_c * num_heads)

        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

    def forward(self, x1d):
        x1d = x1d.transpose(-2,-3)
        x1d = self.norm(x1d)
        gate = torch.sigmoid(self.gate(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c))
        q = self.q(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c)
        k = self.k(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c)
        v = self.v(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c)
        factor = 1 / math.sqrt(self.attn_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k)
        weights = torch.softmax(aff, dim=-1)
        out_1d = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v) * gate
        out_1d = self.final(out_1d.flatten(start_dim=-2))
        out_1d = out_1d.transpose(-2,-3)

        return out_1d


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.attn_num_c = config['attention_num_c']
        self.num_heads = config['num_heads']

        self.norm = nn.LayerNorm(global_config['model']['rep1d_extra_feat'])
        self.q = nn.Linear(global_config['model']['rep1d_extra_feat'], self.attn_num_c*self.num_heads, bias=False)
        self.k = nn.Linear(global_config['model']['rep1d_extra_feat'], self.attn_num_c, bias=False)
        self.v = nn.Linear(global_config['model']['rep1d_extra_feat'], self.attn_num_c, bias=False)
        self.gate = nn.Linear(global_config['model']['rep1d_extra_feat'], self.attn_num_c * self.num_heads)
        self.final = nn.Linear(self.attn_num_c * self.num_heads, global_config['model']['rep1d_extra_feat'])

    def forward(self, x1d):
        x1d = x1d.transpose(-2,-3)
        x1d = self.norm(x1d)
        q_avg = torch.sum(x1d, dim=-2)/x1d.shape[-2]
        q = self.q(q_avg).view(*q_avg.shape[:-1], self.num_heads, self.attn_num_c)
        q = q*(self.attn_num_c ** (-0.5))
        k = self.k(x1d)
        v = self.v(x1d)
        #q, k, v = torch.split(self.kqv(x1d).view(*x1d.shape[:-1], self.attn_num_c, self.num_heads + 2), [self.num_heads, 1, 1], dim=-1)
        #q = torch.mean(q, dim=1)
        gate =  torch.sigmoid(self.gate(x1d).view(*x1d.shape[:-1], self.num_heads, self.attn_num_c))
        w = torch.softmax(torch.einsum('bihc,bikc->bihk', q, k), dim=-1)
        out_1d = torch.einsum('bmhk,bmkc->bmhc', w, v)
        out_1d = out_1d.unsqueeze(-3) * gate
        out = self.final(out_1d.view(*out_1d.shape[:-2], self.attn_num_c * self.num_heads))
        return out.transpose(-2,-3)


class Transition(nn.Module):
    def __init__(self, num_c, n):
        super().__init__()
        self.norm = nn.LayerNorm(num_c)
        self.l1 = nn.Linear(num_c, num_c * n)
        self.l2 = nn.Linear(num_c * n, num_c)

    def forward(self, x1d):
        x = self.norm(x1d)
        x = self.l1(x).relu_()
        x = self.l2(x)
        return x


class OuterProductMean(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = global_config['model']['rep1d_extra_feat'] if config['msa_extra_stack'] else global_config['model']['rep1d_feat']
        out_c = global_config['model']['rep2d_feat']
        mid_c = config['mid_c']
        self.norm = nn.LayerNorm(in_c)
        #self.proj = nn.Linear(in_c, mid_c * 2)
        self.proj_left = nn.Linear(in_c, mid_c)
        self.proj_right = nn.Linear(in_c, mid_c)
        self.final = nn.Linear(mid_c * mid_c, out_c)
        self.mid_c = mid_c
        self.out_c = out_c

    def forward(self, x1d):
        x1d = self.norm(x1d)
        i = self.proj_left(x1d)
        j = self.proj_right(x1d)
        #i, j = [x[..., -1] for x in torch.chunk(self.proj(x1d).view(*x1d.shape[:-1], self.mid_c, 2), 2, dim=-1)]
        x2d = torch.einsum('bmix,bmjy->bjixy', i, j) #/ x1d.shape[1]
        out = self.final(x2d.flatten(start_dim=-2)).transpose(-2, -3)
        out = out/(x1d.shape[1]+1e-3)
        return out


class TriangleMultiplication(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = global_config['model']['rep2d_feat']
        mid_c = config['mid_c']
        self.ingoing = config['ingoing']
        self.norm1 = nn.LayerNorm(in_c)
        self.norm2 = nn.LayerNorm(mid_c)
        self.l1i = nn.Linear(in_c, mid_c)
        self.l1j = nn.Linear(in_c, mid_c)
        self.l1i_sigm = nn.Linear(in_c, mid_c)
        self.l1j_sigm = nn.Linear(in_c, mid_c)
        self.l2_proj = nn.Linear(mid_c, in_c)
        self.l3_sigm = nn.Linear(in_c, in_c)

    def forward(self, x2d):
        x2d = self.norm1(x2d)
        i = self.l1i(x2d) * torch.sigmoid(self.l1i_sigm(x2d))
        j = self.l1j(x2d) * torch.sigmoid(self.l1j_sigm(x2d))
        if self.ingoing:
            out = torch.einsum('bkjc,bkic->bijc', i, j)
        else:
            out = torch.einsum('bikc,bjkc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
        return out


class TriangleAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['model']['rep2d_feat']
        self.ending_node = config['ending_node']
        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attention_num_c*num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attention_num_c*num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attention_num_c*num_heads, bias=False)
        self.bias = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attention_num_c * num_heads)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, x2d):
        if self.ending_node:
            x2d = x2d.transpose(-2, -3)
        x2d = self.norm(x2d)

        q = self.q(x2d).view(*x2d.shape[:-1], self.num_heads, self.attention_num_c)
        k = self.k(x2d).view(*x2d.shape[:-1], self.num_heads, self.attention_num_c)
        v = self.v(x2d).view(*x2d.shape[:-1], self.num_heads, self.attention_num_c)
        factor = 1 / math.sqrt(self.attention_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k)
        b = self.bias(x2d)
        b = b.permute(0, 3, 1, 2)
        b = torch.unsqueeze(b, 1)
        weights = torch.softmax(aff + b, dim=-1)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.num_heads, self.attention_num_c))
        out = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v)*g

        out = self.out(out.flatten(start_dim=-2))
        if self.ending_node:
            out = out.transpose(-2,-3)

        return out


class EvoformerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['RowAttentionWithPairBias'], global_config)
        self.MSAColumnAttention = MSAColumnAttention(config['MSAColumnAttention'], global_config)
        self.MSATransition = Transition(global_config['model']['rep1d_feat'], config['MSATransition']['n'])
        self.OuterProductMean = OuterProductMean(config['OuterProductMean'], global_config)

        self.TriangleMultiplicationOutgoing = TriangleMultiplication(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplication(config['TriangleMultiplicationIngoing'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttention(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttention(config['TriangleAttentionEndingNode'], global_config)
        self.PairTransition = Transition(global_config['model']['rep2d_feat'], config['PairTransition']['n'])

        self.dropout1d_15 = nn.Dropout(0.15)
        self.dropout2d_15 = nn.Dropout2d(0.15)
        self.dropout2d_25 = nn.Dropout2d(0.25)
        # TODO: fix dropout everywhere

    def forward(self, r1d, pair):
        r1d = r1d.clone()
        pair = pair.clone()
        a = self.RowAttentionWithPairBias(r1d.clone(), pair.clone())
        # r1d += self.dropout1d_15(a)
        r1d += a #self.dropout2d_15(b)
        r1d += self.MSAColumnAttention(r1d.clone())
        r1d += self.MSATransition(r1d.clone())
        pair += self.OuterProductMean(r1d.clone())

        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        # pair += self.dropout2d_25(self.TriangleMultiplicationIngoing(pair.clone()))
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return r1d.clone(), pair.clone()


class ExtraMsaStackIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['RowAttentionWithPairBias'], global_config)
        self.MSAColumnGlobalAttention = MSAColumnGlobalAttention(config['MSAColumnGlobalAttention'], global_config)
        self.MSATransition = Transition(global_config['model']['rep1d_extra_feat'], config['MSATransition']['n'])
        self.OuterProductMean = OuterProductMean(config['OuterProductMean'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplication(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplication(config['TriangleMultiplicationIngoing'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttention(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttention(config['TriangleAttentionEndingNode'], global_config)
        self.PairTransition = Transition(global_config['model']['rep2d_feat'], config['PairTransition']['n'])

        self.dropout1d_15 = nn.Dropout(0.15)
        self.dropout2d_15 = nn.Dropout2d(0.15)
        self.dropout2d_25 = nn.Dropout2d(0.25)

    def forward(self, extra, pair):
        pair = pair.clone()
        extra = extra.clone()
        a = self.RowAttentionWithPairBias(extra.clone(), pair.clone())
        # extra += self.dropout1d_15(a)
        extra += a #self.dropout2d_15(b)
        extra += self.MSAColumnGlobalAttention(extra.clone())
        extra += self.MSATransition(extra.clone())
        pair += self.OuterProductMean(extra.clone())
        # pair += self.dropout2d_25(self.TriangleMultiplicationOutgoing(pair.clone()))
        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        # pair += self.dropout2d_25(self.TriangleMultiplicationIngoing(pair.clone()))
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return extra.clone(), pair.clone()


class ExtraMsaStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.project = nn.Linear(global_config['data']['msa_extra_feat'], global_config['model']['rep1d_extra_feat'])
        self.layers = nn.ModuleList([ExtraMsaStackIteration(config['ExtraMsaStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.config = config

    def forward(self, extra, pair):
        extra = self.project(extra)
        for l in self.layers:
            if self.config['ExtraMsaStackIteration']['checkpoint']:
                extra, pair = checkpoint(l, extra, pair)
            else:
                extra, pair = l(extra, pair)
        return pair


class InitPairRepresentation(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()

        target_feat = global_config['data']['target_feat']
        relpos_feat = 2 * global_config['data']['relpos_max'] + 1
        pair_num_c = global_config['model']['rep2d_feat']

        self.r_proj1 = nn.Linear(target_feat, pair_num_c)
        self.r_proj2 = nn.Linear(target_feat, pair_num_c)
        self.relpos_proj = nn.Linear(relpos_feat, pair_num_c)

    def forward(self, feats):
        r1d, relpos = feats['rec_1d'], feats['rec_relpos']

        # create pair representation
        r_proj1 = self.r_proj1(r1d)
        r_proj2 = self.r_proj2(r1d)
        rr_pair = r_proj1.unsqueeze(2) + r_proj2.unsqueeze(1)

        # add relpos
        rr_pair += self.relpos_proj(relpos)
        return rr_pair


class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.rec_norm = nn.LayerNorm(global_config['model']['rep1d_feat'])
        self.x2d_norm = nn.LayerNorm(global_config['model']['rep2d_feat'])
        self.rr_proj = nn.Linear(config['rec_num_bins'], global_config['model']['rep2d_feat'])
        self.config = config

    def forward(self, inputs):
        rec_1d = self.rec_norm(inputs['rec_1d_prev'])
        rep_2d = self.x2d_norm(inputs['rep_2d_prev'])

        rec_crd = inputs['rec_cbeta_prev'][0]
        rec_mask = inputs['rec_mask_prev'][0]
        assert len(rec_crd.shape) == 2 and rec_crd.shape[-1] == 3, rec_crd.shape
        dmat = torch.sqrt(torch.square(rec_crd[:, None, :] - rec_crd[None, :, :]).sum(-1) + 10e-10)
        dgram = utils.dmat_to_dgram(dmat, self.config['rec_min_dist'], self.config['rec_max_dist'], self.config['rec_num_bins'])[1]
        rep_2d[0] += self.rr_proj(dgram * rec_mask[:, None, None] * rec_mask[None, :, None])
        return {'pair_update': rep_2d, 'rec_1d_update': rec_1d}


class InputEmbedder(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        
        target_feat = global_config['data']['target_feat']
        r1d_num_c = global_config['model']['rep1d_feat']
        
        self.rec_1d_project = nn.Linear(target_feat, r1d_num_c)
        self.main_msa_project = nn.Linear(global_config['data']['msa_clus_feat'], r1d_num_c)

        self.InitPairRepresentation = InitPairRepresentation(global_config)
        self.ExtraMsaStack = ExtraMsaStack(config['ExtraMsaStack'], global_config)
        self.RecyclingEmbedder = RecyclingEmbedder(config['RecyclingEmbedder'], global_config)

        self.config = config
        self.global_config = global_config

    @staticmethod
    def _zero_init_recycling(pair, rec_1d):
        num_batch, seq_len, rec_2d_c = pair.shape[0], pair.shape[1], pair.shape[-1]
        return {
            'rec_1d_prev': torch.zeros(num_batch, seq_len, rec_1d.shape[-1]),
            'rep_2d_prev': torch.zeros(num_batch, seq_len, seq_len, rec_2d_c),
            'rec_cbeta_prev': torch.zeros(num_batch, seq_len, 3),
            'rec_mask_prev': torch.zeros(num_batch, seq_len)
        }

    def modules_to_devices(self):
        self.rec_1d_project.to(self.config['device'])
        self.main_msa_project.to(self.config['device'])
        self.InitPairRepresentation.to(self.config['device'])
        self.ExtraMsaStack.to(self.config['ExtraMsaStack']['device'])
        self.RecyclingEmbedder.to(self.config['device'])

    def forward(self, inputs, recycling=None):
        # create pair representation
        pair = self.InitPairRepresentation({k: v.to(self.config['device']) for k, v in inputs['target'].items()})

        # make lig 1d rep
        rec_1d = self.rec_1d_project(inputs['target']['rec_1d'].to(self.config['device'])).unsqueeze(1)
        if 'msa' in inputs:
            rec_1d = self.main_msa_project(inputs['msa']['main'].to(self.config['device'])) + rec_1d.clone()

        # initiaze recycling if firt iteration
        if self.global_config['model']['recycling_on'] and recycling is None:
            recycling = self._zero_init_recycling(pair, rec_1d)

        # add recycling
        if recycling is not None:
            recyc_out = self.RecyclingEmbedder({k: v.to(self.config['device']) for k, v in recycling.items()})
            pair += recyc_out['pair_update']
            rec_1d[:, 0] += recyc_out['rec_1d_update']

        # embed extra stack
        if 'msa' in inputs and 'extra' in inputs['msa']:
            pair = self.ExtraMsaStack(
                inputs['msa']['extra'].to(self.config['ExtraMsaStack']['device']),
                pair.to(self.config['ExtraMsaStack']['device'])
            )

        return {'r1d': rec_1d, 'pair': pair}
