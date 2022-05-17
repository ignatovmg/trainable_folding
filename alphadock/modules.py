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
        in_num_c = config['extra_msa_channel']
        pair_rep_num_c = global_config['rep_2d']['num_c']

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


class LigColumnAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        in_num_c = global_config['rep_1d']['num_c']

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


class ExtraColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.attn_num_c = config['attention_num_c']
        self.num_heads = config['num_heads']

        self.norm = nn.LayerNorm(global_config['extra_msa_channel'])
        self.q = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c*self.num_heads, bias=False)
        self.k = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c, bias=False)
        self.v = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c, bias=False)
        #self.kqv = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c * (self.num_heads + 2), bias=False)
        self.gate = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c * self.num_heads)
        self.final = nn.Linear(self.attn_num_c * self.num_heads, global_config['extra_msa_channel'])

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
        in_c, out_c = config['extra_msa_channel'], global_config['rep_2d']['num_c']
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


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = config['rep_2d_num_c']
        mid_c = config['mid_c']
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
        out = torch.einsum('bikc,bjkc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
        return out


class TriangleMultiplicationIngoing(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = config['rep_2d_num_c']
        mid_c = config['mid_c']
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
        out = torch.einsum('bkjc,bkic->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
        return out


class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attn_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = config['rep_2d_num_c']
        self.rand_remove = config['rand_remove']
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attn_num_c*num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attn_num_c*num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attn_num_c*num_heads, bias=False)
        #self.qkv = nn.Linear(num_in_c, attention_num_c * num_heads * 3, bias=False)
        self.bias = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attn_num_c * num_heads)
        self.out = nn.Linear(attn_num_c * num_heads, num_in_c)

    def forward(self, x2d):
        x2d = self.norm(x2d)

        if self.rand_remove > 0.0 and self.training:
            selection = torch.randperm(x2d.shape[1], device=x2d.device)
            selection = selection[:max(1, int(selection.shape[0] * (1 - self.rand_remove)))]
            shape_full = x2d.shape
            res_ids_cart = torch.cartesian_prod(selection, selection)
            x2d = x2d[:, res_ids_cart[:, 0], res_ids_cart[:, 1]].reshape(x2d.shape[0], len(selection), len(selection), x2d.shape[-1])

        q = self.q(x2d).view(*x2d.shape[:-1], self.num_heads, self.attn_num_c)
        k = self.k(x2d).view(*x2d.shape[:-1], self.num_heads, self.attn_num_c)
        v = self.v(x2d).view(*x2d.shape[:-1], self.num_heads, self.attn_num_c)
        factor = 1 / math.sqrt(self.attn_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k)
        b = self.bias(x2d)
        b = b.permute(0, 3, 1, 2)
        b = torch.unsqueeze(b, 1)
        weights = torch.softmax(aff + b, dim=-1)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.num_heads, self.attn_num_c))
        out = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v)*g

        out = self.out(out.flatten(start_dim=-2))

        if self.rand_remove > 0.0 and self.training:
            out_full = torch.zeros(shape_full, device=out.device, dtype=out.dtype)
            out_full[:, res_ids_cart[:, 0], res_ids_cart[:, 1]] = out.flatten(1, 2)
            out = out_full

        return out


class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = config['rep_2d_num_c']
        self.rand_remove = config['rand_remove']

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
        x2d = x2d.transpose(-2,-3)
        x2d = self.norm(x2d)

        if self.rand_remove > 0.0 and self.training:
            selection = torch.randperm(x2d.shape[1], device=x2d.device)
            selection = selection[:max(1, int(selection.shape[0] * (1 - self.rand_remove)))]
            shape_full = x2d.shape
            res_ids_cart = torch.cartesian_prod(selection, selection)
            x2d = x2d[:, res_ids_cart[:, 0], res_ids_cart[:, 1]].reshape(x2d.shape[0], len(selection), len(selection), x2d.shape[-1])

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
        out = out.transpose(-2,-3)


        if self.rand_remove > 0.0 and self.training:
            out_full = torch.zeros(shape_full, device=out.device, dtype=out.dtype)
            out_full[:, res_ids_cart[:, 0], res_ids_cart[:, 1]] = out.flatten(1, 2)
            out = out_full

        return out


class TemplatePairStackIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationIngoing'], global_config)
        self.PairTransition = Transition(config['TriangleMultiplicationIngoing']['rep_2d_num_c'], config['PairTransition']['n'])
        self.dropout2d_25 = nn.Dropout2d(0.25)

    def forward(self, x2d):
        x2d += self.TriangleAttentionStartingNode(x2d.clone())
        x2d += self.TriangleAttentionEndingNode(x2d.clone())
        x2d += self.TriangleMultiplicationOutgoing(x2d.clone())
        x2d += self.TriangleMultiplicationIngoing(x2d.clone())
        x2d += self.PairTransition(x2d.clone())
        return x2d


class TemplatePairStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.rr_proj = nn.Linear(global_config['hh_rr'], config['TemplatePairStackIteration']['TriangleAttentionEndingNode']['value_dim'])
        self.layers = nn.ModuleList([TemplatePairStackIteration(config['TemplatePairStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.norm = nn.LayerNorm(config['TemplatePairStackIteration']['TriangleAttentionStartingNode']['rep_2d_num_c'])
        self.config = config

    def forward(self, inputs):
        emb_temp = features_summit.template_pair(inputs, False)
        out = self.rr_proj(emb_temp)
        num_batch = out.shape[0]
        out_final = []
        for i in range(num_batch):
            out_tmp = out[i].clone()
            for l in self.layers:
            #if self.config['TemplatePairStackIteration']['checkpoint']:
            #    out = checkpoint(lambda x: l(x), out)
            #else:
                out_tmp = l(out_tmp)
            out_final.append(torch.unsqueeze(out_tmp, 0))
        out_final = torch.cat(out_final, 0) 
        out_final = self.norm(out_final)

        return out_final


class TemplatePointwiseAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['rep_2d']['num_c']
        num_in_c_kv = config['rep_2d_num_c']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads
        self.num_in_c = num_in_c

        #self.norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.k = nn.Linear(num_in_c_kv, attention_num_c * num_heads, bias=False)
        self.v = nn.Linear(num_in_c_kv, attention_num_c * num_heads, bias=False)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, z2d, t2d):
        num_res = z2d.shape[-2]
        num_batch = z2d.shape[0]
        z2d = z2d.view(z2d.shape[0], -1, 1, z2d.shape[-1])
        t2d = t2d.permute(0,2,3,1,4)
        t2d = t2d.view(t2d.shape[0], -1, *t2d.shape[3:])
        q = self.q(z2d).view(*z2d.shape[:-1], self.num_heads, self.attention_num_c)* (self.attention_num_c**(-0.5))
        k = self.k(t2d).view(*t2d.shape[:-1], self.num_heads, self.attention_num_c)
        v = self.v(t2d).view(*t2d.shape[:-1], self.num_heads, self.attention_num_c)
        logits = torch.einsum('bpqhc,bpkhc->bphqk', q, k)
        weights = torch.softmax(logits, dim=-1)
        weighted_avg = torch.einsum('bphqk,bpkhc->bpqhc', weights, v)
        
        out = self.out(weighted_avg.flatten(start_dim=-2))
        out = out.reshape(num_batch, num_res, num_res, self.num_in_c)

        return out


class EvoformerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['RowAttentionWithPairBias'], global_config)
        self.LigColumnAttention = LigColumnAttention(config['LigColumnAttention'], global_config)
        self.RecTransition = Transition(global_config['rep_1d']['num_c'], config['RecTransition']['n'])
        self.OuterProductMean = OuterProductMean(config['OuterProductMean'], global_config)

        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationIngoing'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

        self.dropout1d_15 = nn.Dropout(0.15)
        self.dropout2d_15 = nn.Dropout2d(0.15)
        self.dropout2d_25 = nn.Dropout2d(0.25)
        # TODO: fix dropout everywhere

    def forward(self, r1d, pair):
        a = self.RowAttentionWithPairBias(r1d.clone(), pair.clone())
        # r1d += self.dropout1d_15(a)
        r1d += a #self.dropout2d_15(b)
        r1d += self.LigColumnAttention(r1d.clone())
        r1d += self.RecTransition(r1d.clone())
        pair += self.OuterProductMean(r1d.clone())

        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        # pair += self.dropout2d_25(self.TriangleMultiplicationIngoing(pair.clone()))
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return r1d.clone(), pair.clone()


class FragExtraStackIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['RowAttentionWithPairBias'], global_config)
        self.ExtraColumnGlobalAttention = ExtraColumnGlobalAttention(config['ExtraColumnGlobalAttention'], global_config)
        self.RecTransition = Transition(config['extra_msa_channel'], config['RecTransition']['n'])
        self.OuterProductMean = OuterProductMean(config['OuterProductMean'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

        self.dropout1d_15 = nn.Dropout(0.15)
        self.dropout2d_15 = nn.Dropout2d(0.15)
        self.dropout2d_25 = nn.Dropout2d(0.25)

    def forward(self, extra, pair):
        pair = pair.clone()
        extra = extra.clone()
        a = self.RowAttentionWithPairBias(extra.clone(), pair.clone())
        # extra += self.dropout1d_15(a)
        extra += a #self.dropout2d_15(b)
        extra += self.ExtraColumnGlobalAttention(extra.clone())
        extra += self.RecTransition(extra.clone())
        pair += self.OuterProductMean(extra.clone())
        # pair += self.dropout2d_25(self.TriangleMultiplicationOutgoing(pair.clone()))
        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        # pair += self.dropout2d_25(self.TriangleMultiplicationIngoing(pair.clone()))
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return extra.clone(), pair.clone()


class FragExtraStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.project = nn.Linear(global_config['msa_extra_in_c'], config['extra_msa_channel'])
        self.layers = nn.ModuleList([FragExtraStackIteration(config['FragExtraStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.config = config

    def forward(self, extra, pair):
        extra = self.project(extra)
        #print([x for x in self.layers[0].parameters()])
        #print(list(self.layers[0].parameters())[0])

        for l in self.layers:
            if self.config['FragExtraStackIteration']['checkpoint']:
                extra, pair = checkpoint(l, extra, pair)
            else:
                extra, pair = l(extra, pair)
        return pair


class InitPairRepresentation(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()

        rec_in_c = global_config['rec_in_c']
        relpos_c = global_config['rec_relpos_c']
        pair_num_c = global_config['rep_2d']['num_c']

        self.r_proj1 = nn.Linear(rec_in_c, pair_num_c)
        self.r_proj2 = nn.Linear(rec_in_c, pair_num_c)
        self.relpos_proj = nn.Linear(relpos_c, pair_num_c)

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
        self.rec_norm = nn.LayerNorm(global_config['rep_1d']['num_c'])
        self.x2d_norm = nn.LayerNorm(global_config['rep_2d']['num_c'])
        self.rr_proj = nn.Linear(config['rec_num_bins'], global_config['rep_2d']['num_c'])
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
        
        rec_in_c = global_config['rec_in_c']
        r1d_num_c = global_config['rep_1d']['num_c']
        
        self.rec_1d_project = nn.Linear(rec_in_c, r1d_num_c).to(config['device'])
        self.main_msa_project = nn.Linear(global_config['msa_main_in_c'], r1d_num_c).to(config['device'])

        self.InitPairRepresentation = InitPairRepresentation(global_config).to(config['device'])
        self.TemplatePairStack = TemplatePairStack(config['TemplatePairStack'], global_config).to(config['TemplatePairStack']['device'])
        self.TemplatePointwiseAttention = TemplatePointwiseAttention(config['TemplatePointwiseAttention'], global_config).to(config['TemplatePointwiseAttention']['device'])
        self.FragExtraStack = FragExtraStack(config['FragExtraStack'], global_config).to(config['FragExtraStack']['device'])
        self.RecyclingEmbedder = RecyclingEmbedder(config['RecyclingEmbedder'], global_config).to(config['device'])

        self.config = config
        self.global_config = global_config

    def forward(self, inputs, recycling=None):
        # create pair representation
        pair = self.InitPairRepresentation({k: v.to(self.config['device']) for k, v in inputs['target'].items()})
        num_batch, seq_len, rec_2d_c = pair.shape[0], pair.shape[1], pair.shape[-1]

        # make lig 1d rep
        rec_1d = self.rec_1d_project(inputs['target']['rec_1d'].to(self.config['device'])).unsqueeze(1)
        rec_1d_c = rec_1d.shape[-1]
        if 'msa' in inputs:
            rec_1d = self.main_msa_project(inputs['msa']['main'].to(self.config['device'])) + rec_1d.clone()

        # add recycling
        if (self.global_config['recycling_num_iter'] > 0 and recycling == None):
            tmp_inp = {'rec_1d_prev': torch.zeros(num_batch,seq_len,rec_1d_c),
                    'rep_2d_prev': torch.zeros(num_batch,seq_len,seq_len,rec_2d_c),
                    'rec_cbeta_prev': torch.zeros(num_batch, seq_len ,3),
                    'rec_mask_prev': torch.zeros(num_batch,seq_len)
                    }
            recyc_out = self.RecyclingEmbedder({k: v.to(self.config['device']) for k, v in tmp_inp.items()})
            pair += recyc_out['pair_update']
            rec_1d[:, 0] += recyc_out['rec_1d_update']
        if recycling is not None:
            recyc_out = self.RecyclingEmbedder({k: v.to(self.config['device']) for k, v in recycling.items()})
            pair += recyc_out['pair_update']
            rec_1d[:, 0] += recyc_out['rec_1d_update']

        # make template embedding
        if 'template' in inputs:
            hh_inputs = {k: v.to(self.config['TemplatePairStack']['device']) for k, v in inputs['template'].items()}
            if self.config['TemplatePairStack']['checkpoint']:
                hh_2d = checkpoint(lambda x: self.TemplatePairStack(x), hh_inputs)
            else:
                hh_2d = self.TemplatePairStack(hh_inputs)
            template_embedding = self.TemplatePointwiseAttention(pair.clone().to(self.config['TemplatePointwiseAttention']['device']), hh_2d.to(self.config['TemplatePointwiseAttention']['device']))
            template_embedding = template_embedding * (torch.sum(hh_inputs["template_mask"]) > 0)

            # add embeddings to the pair rep
            pair += template_embedding.to(pair.device)

        # embed extra stack
        if 'msa' in inputs and 'extra' in inputs['msa']:
            pair = self.FragExtraStack(
                inputs['msa']['extra'].to(self.config['FragExtraStack']['device']),
                pair.to(self.config['FragExtraStack']['device'])
            )

        return {'r1d': rec_1d, 'pair': pair}

class TemplateAngleEmbedder(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.rep_1d_num_c = global_config['rep_1d']['num_c']
        self.angle_num_c = global_config['temp_single_emb_c']
        self.template_single_embedding = nn.Linear(self.angle_num_c, self.rep_1d_num_c)
        self.template_projection = nn.Linear(self.rep_1d_num_c, self.rep_1d_num_c)
        self.relu = nn.ReLU()
        
        self.config = config
        self.global_config = global_config
    def forward(self, inputs):
        hh_inputs = {k: v.to(self.config['TemplatePairStack']['device']) for k, v in inputs['template'].items()}
        num_batch = hh_inputs['template_aatype'].shape[0]
        ret = {
                "torsion_angles_sin_cos": [],
                "alt_torsion_angles_sin_cos": [],
                "torsion_angles_mask": []
            }
        for i in range(num_batch):
            torsion_tmp = all_atom.atom37_to_torsion_angles(
                aatype=hh_inputs['template_aatype'][i].clone().long(),
                all_atom_pos=hh_inputs['template_all_atom_positions'][i].clone(),
                all_atom_mask=hh_inputs['template_all_atom_masks'][i].clone(),
                placeholder_for_undefined=not True)
            ret['torsion_angles_sin_cos'].append(torch.unsqueeze(torsion_tmp['torsion_angles_sin_cos'], 0))
            ret['alt_torsion_angles_sin_cos'].append(torch.unsqueeze(torsion_tmp['alt_torsion_angles_sin_cos'], 0))
            ret['torsion_angles_mask'].append(torch.unsqueeze(torsion_tmp['torsion_angles_mask'], 0))

        ret = {k: torch.cat(v, 0) for k, v in ret.items()}

        template_features = torch.cat(
            [
                nn.functional.one_hot(hh_inputs["template_aatype"].long(), 22),
                ret["torsion_angles_sin_cos"].reshape(
                    *ret["torsion_angles_sin_cos"].shape[:-2], 14
                ),
                ret["alt_torsion_angles_sin_cos"].reshape(
                    *ret["alt_torsion_angles_sin_cos"].shape[:-2], 14
                ),
                ret["torsion_angles_mask"],
            ],
            dim=-1
        )
        template_act = self.template_single_embedding(template_features)
        template_act = self.relu(template_act)
        template_act = self.template_projection(template_act)
    
        return template_act




def example3():
    from config import config, DATA_DIR
    with torch.no_grad():
        model = InputEmbedder(config['InputEmbedder'], config).cuda()

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)

        from dataset import DockingDataset
        ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
        #print(ds[0])
        item = ds[0]

        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None].cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)

        #print(item['fragments']['num_res'])
        #print(item['fragments']['num_atoms'])
        out = model(item)
        for k1, v1 in out.items():
            print(k1, v1.shape, v1.dtype)


if __name__ == '__main__':
    example3()
