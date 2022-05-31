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
import torch.functional as F
from torch.utils.checkpoint import checkpoint
import math

from alphadock import quat_affine


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        self.num_head = config['num_head']
        #self.num_scalar_qk = config['num_scalar_qk']
        self.num_scalar_qk = config['num_scalar_qk']
        self.num_scalar_v = config['num_scalar_v']
        self.num_point_qk = config['num_point_qk']
        self.num_point_v = config['num_point_v']
        self.num_2d_qk = config['num_2d_qk']
        #self.num_scalar_v = config['num_scalar_v']
        #self.num_point_v = config['num_point_v']
        self.num_2d_v = config['num_2d_v']

        self.num_output_c = global_config['model']['single_rep_feat'] #config['num_channel']
        self.rep_1d_num_c = global_config['model']['single_rep_feat']
        self.rep_2d_num_c = global_config['model']['rep2d_feat']

        self.q = nn.Linear(self.rep_1d_num_c, self.num_scalar_qk* self.num_head)
        self.kv = nn.Linear(self.rep_1d_num_c, (self.num_scalar_qk + self.num_scalar_v) * self.num_head)
        self.q_points = nn.Linear(self.rep_1d_num_c, self.num_point_qk * self.num_head * 3)
        self.kv_points = nn.Linear(self.rep_1d_num_c, (self.num_point_qk + self.num_point_v) * self.num_head * 3)
        #self.rec_kqv_1d = nn.Linear(self.rep_1d_num_c, (self.num_scalar_qk * 2 + self.num_scalar_v) * self.num_head, bias=False)
        #self.rec_kqv_point = nn.Linear(self.rep_1d_num_c, (self.num_point_qk * 2 + self.num_point_v) * self.num_head * 3, bias=False)
        self.rr_kqv_2d = nn.Linear(self.rep_2d_num_c, self.num_head)
        self.final_r = nn.Linear(self.num_head * (self.rep_2d_num_c + self.num_scalar_v + 4 * self.num_point_v), self.num_output_c)
        self.trainable_w = nn.Parameter(torch.zeros((self.num_head)))
        self.softplus = nn.Softplus()

    def forward(self, rec_1d, rep_2d, rec_T):
        batch = rec_1d.shape[0]
        num_res = rec_1d.shape[1]
        rec_T = quat_affine.QuatAffine.from_tensor(rec_T)
        
        q_scalar = self.q(rec_1d)
        q_scalar = q_scalar.view(*q_scalar.shape[:-1], self.num_head, -1)
        kv_scalar = self.kv(rec_1d)
        kv_scalar = kv_scalar.view(*kv_scalar.shape[:-1], self.num_head, -1)
        k_scalar, v_scalar = torch.tensor_split(kv_scalar, (self.num_scalar_qk,), dim=-1)

        q_point = self.q_points(rec_1d)
        q_point = torch.split(q_point, q_point.shape[-1]//3, dim=-1)
        q_point_global = rec_T.apply_to_point(q_point)
        q_point_final = [x.view(*x.shape[:-1], self.num_head, -1) for x in q_point_global]
        kv_point = self.kv_points(rec_1d)
        kv_point = torch.split(kv_point, kv_point.shape[-1]//3, dim=-1)
        kv_point_global = rec_T.apply_to_point(kv_point)
        kv_point_final = [x.view(*x.shape[:-1], self.num_head, -1) for x in kv_point_global]
        k_point, v_point = list(zip(*[torch.tensor_split(x, (self.num_point_qk,), dim=-1) for x in kv_point_final]))
        
        scalar_variance = max(self.num_scalar_qk, 1) * 1
        point_variance = max(self.num_point_qk, 1) *9.0/2
        num_logit_terms = 3

        scalar_weights = math.sqrt(1.0/(num_logit_terms * scalar_variance))
        point_weights = math.sqrt(1.0/(num_logit_terms * point_variance))
        trainable_point_weights = self.softplus(self.trainable_w)
        point_w = point_weights * torch.unsqueeze(trainable_point_weights, dim=-1)

        q_point_final = [x.transpose(-2,-3) for x in q_point_final]
        k_point_final = [x.transpose(-2,-3) for x in k_point]
        v_point_final = [x.transpose(-2,-3) for x in v_point]
        dist2 = [torch.square(qx[...,None,:] - kx[...,None, :, :]) for qx, kx in zip(q_point_final, k_point_final)]
        dist_final = dist2[0] + dist2[1] + dist2[2]
        attn_qk_point = -0.5*torch.sum(point_w[:, None, None, :] * dist_final, dim=-1)

        q = scalar_weights * q_scalar
        q = q.transpose(-2,-3)
        k = k_scalar.transpose(-2,-3)
        v = v_scalar.transpose(-2,-3)
        attn_qk_scalar = torch.matmul(q, k.transpose(-2,-1))
        attn_logits = attn_qk_scalar + attn_qk_point
        attn_2d = self.rr_kqv_2d(rep_2d)
        attn_2d = torch.permute(attn_2d, (0,3,1,2))
        attn_2d = math.sqrt(1.0/num_logit_terms) * attn_2d
        attn_logits = attn_logits + attn_2d
        attn = torch.softmax(attn_logits, dim=-1)
        result_scalar = torch.matmul(attn, v)
        result_point_global = [torch.sum(attn[..., None] * vx[...,None,:,:], dim=-2) for vx in v_point_final]

        result_scalar = result_scalar.transpose(-2, -3)
        result_point_global = [x.transpose(-2, -3) for x in result_point_global]
        
        out_feat = []
        result_scalar_final = torch.reshape(result_scalar, (*result_scalar.shape[:-2], self.num_head*self.num_scalar_v))
        out_feat.append(result_scalar_final)

        result_point_global_final = [torch.reshape(x, (*x.shape[:-2], self.num_head * self.num_point_v)) for x in result_point_global]
        result_point_local = rec_T.invert_point(result_point_global_final)
        out_feat.extend(result_point_local)
        out_feat.append(torch.sqrt(1e-8 + torch.square(result_point_local[0])+ torch.square(result_point_local[1]) + torch.square(result_point_local[2])))
        result_attention_over_2d = torch.einsum('...hij, ...ijc->...ihc', attn, rep_2d)
        num_out = self.num_head * result_attention_over_2d.shape[-1]
        out_feat.append(torch.reshape(result_attention_over_2d, (*result_attention_over_2d.shape[:-2], num_out)))

        final_act = torch.cat(out_feat, dim=-1)
        out = self.final_r(final_act)

        return out


class PredictSidechains(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = global_config['model']['single_rep_feat']
        num_c = config['num_c']
        self.num_torsions = global_config['model']['num_torsions']

        self.s_cur = nn.Linear(num_in_c, num_c)
        self.s_ini = nn.Linear(num_in_c, num_c)
        self.relu = nn.ReLU()

        self.res1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c)
        )

        self.res2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c)
        )

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, self.num_torsions * 2)
        )

    def forward(self, s_cur, s_ini):
        a = self.s_cur(self.relu(s_cur.clone())) + self.s_ini(self.relu(s_ini))
        a += self.res1(a.clone())
        a += self.res2(a.clone())
        return self.final(a).reshape(*a.shape[:-1], self.num_torsions, 2)


class PredictLDDT(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = global_config['model']['single_rep_feat']
        num_c = config['num_c']
        num_bins = config['num_bins']

        self.layers = nn.Sequential(
            nn.LayerNorm(num_in_c),
            nn.Linear(num_in_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_bins)
        )

    def forward(self, rep_1d):
        return self.layers(rep_1d)


class StructureModuleIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InvariantPointAttention = InvariantPointAttention(config['InvariantPointAttention'], global_config)
        self.drop = nn.Dropout(0.1)
        self.rec_norm = nn.LayerNorm(global_config['model']['single_rep_feat'])
        self.rec_norm2 = nn.LayerNorm(global_config['model']['single_rep_feat'])

        num_1dc = global_config['model']['single_rep_feat']
        self.transition_r = nn.Sequential(
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc)
        )

        self.backbone_update = nn.Linear(num_1dc, 6)
        self.PredictSidechains = PredictSidechains(config['PredictSidechains'], global_config)
        self.PredictRecLDDT = PredictLDDT(config['PredictRecLDDT'], global_config)

    def forward(self, rec_1d_init, rec_1d, rep_2d, rec_T, rec_torsions):
        #rec_1d_init, rec_1d, rep_2d, rec_T = inputs['rec_1d_init'], inputs['rec_1d'], inputs['rep_2d'], inputs['rec_T']

        # IPA
        rec_1d_update = self.InvariantPointAttention(rec_1d.clone(), rep_2d, rec_T)
        rec_1d = self.rec_norm(rec_1d + rec_1d_update)

        # transition
        rec_1d_update = self.transition_r(rec_1d)
        rec_1d = self.rec_norm2(rec_1d + rec_1d_update)

        # update backbone
        rec_T = quat_affine.QuatAffine.from_tensor(rec_T)
        rec_T = rec_T.pre_compose(self.backbone_update(rec_1d.clone()))

        # sidechains
        rec_torsions = rec_torsions + self.PredictSidechains(rec_1d, rec_1d_init)

        # If output is a dict, gradient checkpointing doesn't calculate gradients,
        # so I have to return a list instead
        return rec_1d, rec_T.to_tensor(), rec_torsions, self.PredictRecLDDT(rec_1d.clone())


class PredictDistogram(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = global_config['model']['rep2d_feat']
        self.rr_proj = nn.Linear(num_in_c, config['rec_num_bins'])

    def forward(self, rep_2d, rec_size):
        sym = rep_2d + rep_2d.transpose(1, 2)
        rr = self.rr_proj(sym)
        return {'rr': rr}


class StructureModule(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.num_iter = config['num_iter']
        num_1dc = global_config['model']['single_rep_feat']

        self.StructureModuleIteration = StructureModuleIteration(config['StructureModuleIteration'], global_config)

        self.layers = [self.StructureModuleIteration for _ in range(self.num_iter)]
        self.norm_rec_1d_init = nn.LayerNorm(num_1dc)
        self.norm_2d_init = nn.LayerNorm(global_config['model']['rep2d_feat'])
        self.rec_1d_proj = nn.Linear(num_1dc, num_1dc)
        self.pred_distogram = PredictDistogram(config['PredictDistogram'], global_config)

        self.position_scale = global_config['model']['position_scale']
        self.config = config
        self.global_config = global_config

    def forward(self, inputs):
        # batch size must be one
        assert inputs['r1d'].shape[0] == 1

        rec_1d_init = self.norm_rec_1d_init(inputs['r1d'])
        pair = self.norm_2d_init(inputs['pair'])
        rec_1d = self.rec_1d_proj(rec_1d_init)

        # Set masked frames to origin
        #rec_T = inputs['rec_bb_affine'].clone().to(rec_1d.device)
        #rec_T_masked = torch.where(inputs['rec_bb_affine_mask'] < 1)[1]
        #rec_T[:, rec_T_masked, :] = 0
        #rec_T[:, rec_T_masked, 0] = 1
        #rec_T[:, :, -3:] = rec_T[:, :, -3:] / self.position_scale

        rec_T = torch.zeros((1, rec_1d.shape[1], 7), device=rec_1d.device, dtype=rec_1d.dtype)
        rec_T[:, :, 0] = 1
        rec_T.requires_grad = True

        rec_torsions = torch.zeros((1, rec_1d.shape[1], self.global_config['model']['num_torsions'], 2), device=rec_1d.device, dtype=rec_1d.dtype)
        rec_torsions[..., 0] = 1

        struct_dict = {
            'rec_1d_init': rec_1d_init,
            'rec_1d': rec_1d,
            'rep_2d': pair,
            'rec_T': rec_T,
            'rec_torsions': rec_torsions
        }

        struct_traj = [struct_dict]
        for l in self.layers:
            args = [
                struct_traj[-1]['rec_1d_init'],
                struct_traj[-1]['rec_1d'],
                struct_traj[-1]['rep_2d'],
                struct_traj[-1]['rec_T'],
                struct_traj[-1]['rec_torsions']
            ]
            update = l(*args)
            struct_traj.append(
                {
                    'rec_1d_init': rec_1d_init,
                    'rec_1d': update[0],
                    'rep_2d': pair,
                    'rec_T': update[1],
                    'rec_torsions': update[2],
                    'rec_lddt': update[3]
                }
            )

        return {
            'rec_T': torch.stack([x['rec_T'] for x in struct_traj[1:]], dim=1),
            'rec_torsions': torch.stack([x['rec_torsions'] for x in struct_traj[1:]], dim=1),
            'rec_1d': struct_traj[-1]['rec_1d'],
            'rec_lddt': torch.stack([x['rec_lddt'] for x in struct_traj[1:]], dim=1),
            'distogram': self.pred_distogram(pair, rec_1d.shape[1])
        }


if __name__ == '__main__':
    pass

