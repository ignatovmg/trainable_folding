import torch
from torch import nn
from multimer import config_multimer, structure_multimer, all_atom_multimer
import math
from typing import Sequence
import numpy as np
from alphadock import residue_constants as rc
from alphadock import  all_atom
from torch.utils.checkpoint import checkpoint

def gumbel_noise(shape: Sequence[int]) -> torch.tensor:
  """Generate Gumbel Noise of given Shape.

  This generates samples from Gumbel(0, 1).

  Args:
    key: Jax random number key.
    shape: Shape of noise to return.

  Returns:
    Gumbel noise of given shape.
  """
  epsilon = 1e-6
  uniform_noise = torch.rand(shape)
  gumbel = -torch.log(-torch.log(uniform_noise + epsilon) + epsilon)
  return gumbel

def gumbel_argsort_sample_idx(logits: torch.tensor) -> torch.tensor:
  z = gumbel_noise(logits.shape)
  # This construction is equivalent to jnp.argsort, but using a non stable sort,
  # since stable sort's aren't supported by jax2tf.
  axis = len(logits.shape) - 1


def sample_msa(batch, max_seq):
    logits = (torch.clip(torch.sum(batch['msa_mask'], -1), 0., 1.) - 1.) * 1e6
  # The cluster_bias_mask can be used to preserve the first row (target
  # sequence) for each chain, for example.
    if 'cluster_bias_mask' not in batch:
        cluster_bias_mask = nn.functional.pad(torch.zeros(batch['msa'].shape[1] - 1), (1, 0), 'constant', 1.)
    else:
        cluster_bias_mask = batch['cluster_bias_mask']

    # logits += cluster_bias_mask * 1e6
    rand_ind = torch.randperm(logits.shape[-1] - 1) + 1
    index_order = torch.cat((torch.tensor([0]), rand_ind))
    # index_order = gumbel_argsort_sample_idx(key.get(), logits)
    #index_order = torch.arange(891)
    sel_idx = index_order[:max_seq]
    extra_idx = index_order[max_seq:]
    for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
        if k in batch:
            batch['extra_' + k] = batch[k][:, extra_idx]
            batch[k] = batch[k][:, sel_idx]
    return batch

def make_masked_msa(batch):
  """Create data for BERT on raw MSA."""
  # Add a random amino acid uniformly.
  #random_aa = torch.tensor([0.05] * 20 + [0., 0.], dtype=torch.float32)

  #categorical_probs = (
  #    config.uniform_prob * random_aa +
  #    config.profile_prob * batch['msa_profile'] +
  #    config.same_prob * nn.functional.one_hot(batch['msa'].long(), 22))

  # Put all remaining probability on [MASK] which is a new column.
  #pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
  #pad_shapes[-1][1] = 1
  #mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
  #assert mask_prob >= 0.
  #categorical_probs = torch.pad(
  #    categorical_probs, pad_shapes, constant_values=mask_prob)
  #sh = batch['msa'].shape
  #key, mask_subkey, gumbel_subkey = key.split(3)
  #uniform = utils.padding_consistent_rng(jax.random.uniform)
  #mask_position = uniform(mask_subkey.get(), sh) < config.replace_fraction
  #mask_position *= batch['msa_mask']

  #logits = jnp.log(categorical_probs + epsilon)
  #bert_msa = gumbel_max_sample(gumbel_subkey.get(), logits)
  #bert_msa = jnp.where(mask_position,
  #                     jnp.argmax(bert_msa, axis=-1), batch['msa'])
  #bert_msa *= batch['msa_mask']

  # Mix real and masked MSA.
  #if 'bert_mask' in batch:
  #  batch['bert_mask'] *= mask_position.astype(jnp.float32)
  #else:
  #  batch['bert_mask'] = mask_position.astype(jnp.float32)
  batch['true_msa'] = batch['msa']

  return batch


def nearest_neighbor_clusters(batch, gap_agreement_weight=0.):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask.

  weights = torch.tensor(
      [1.] * 21 + [gap_agreement_weight] + [0.], dtype=torch.float32, device='cuda:0')

  msa_mask = batch['msa_mask']
  msa_one_hot = torch.nn.functional.one_hot(batch['msa'].long(), 23)

  extra_mask = batch['extra_msa_mask']
  extra_one_hot = torch.nn.functional.one_hot(batch['extra_msa'].long(), 23)

  msa_one_hot_masked = msa_mask[..., None] * msa_one_hot
  extra_one_hot_masked = extra_mask[..., None] * extra_one_hot

  agreement = torch.einsum('...mrc, ...nrc->...nm', extra_one_hot_masked,
                         weights * msa_one_hot_masked)

  cluster_assignment = torch.nn.functional.softmax(1e3 * agreement, 1)
  cluster_assignment *= torch.einsum('...mr, ...nr->...mn', msa_mask, extra_mask)

  cluster_count = torch.sum(cluster_assignment, dim=-1)
  cluster_count += 1.  # We always include the sequence itself.

  msa_sum = torch.einsum('...nm, ...mrc->...nrc', cluster_assignment, extra_one_hot_masked)
  msa_sum += msa_one_hot_masked

  cluster_profile = msa_sum / cluster_count[..., None, None]

  extra_deletion_matrix = batch['extra_deletion_matrix']
  deletion_matrix = batch['deletion_matrix']

  del_sum = torch.einsum('...nm, ...mc->...nc', cluster_assignment,
                       extra_mask * extra_deletion_matrix)
  del_sum += deletion_matrix  # Original sequence.
  cluster_deletion_mean = del_sum / cluster_count[..., None]

  return cluster_profile, cluster_deletion_mean

def create_msa_feat(batch):
  """Create and concatenate MSA features."""
  msa_1hot = torch.nn.functional.one_hot(batch['msa'].long(), 23)
  deletion_matrix = batch['deletion_matrix']
  has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]

  deletion_mean_value = (torch.arctan(batch['cluster_deletion_mean'] / 3.) *
                         (2. / torch.pi))[..., None]

  msa_feat = [
      msa_1hot,
      has_deletion,
      deletion_value,
      batch['cluster_profile'],
      deletion_mean_value
  ]

  return torch.cat(msa_feat, -1)

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

def create_extra_msa_feature(batch, num_extra_msa):
    extra_msa = batch['extra_msa'][:, :num_extra_msa]
    deletion_matrix = batch['extra_deletion_matrix'][:, :num_extra_msa]
    msa_1hot = torch.nn.functional.one_hot(extra_msa.long(), 23)
    has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]
    extra_msa_mask = batch['extra_msa_mask'][:, :num_extra_msa]
    return torch.cat([msa_1hot, has_deletion, deletion_value], -1), extra_msa_mask

class OuterProductMean(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = config['norm_channel']
        out_c = config['num_output_channel']
        mid_c = config['num_outer_channel']
        self.layer_norm_input = nn.LayerNorm(in_c)
        self.left_projection = nn.Linear(in_c, mid_c)
        self.right_projection = nn.Linear(in_c, mid_c)
        self.output = nn.Linear(mid_c * mid_c, out_c)
        self.mid_c = mid_c
        self.out_c = out_c

    def forward(self, act, mask):
        act = self.layer_norm_input(act)
        mask = mask[..., None]
        left_act = mask*self.left_projection(act)
        right_act = mask*self.right_projection(act)
        x2d = torch.einsum('bmix,bmjy->bjixy', left_act, right_act) #/ x1d.shape[1]
        out = self.output(x2d.flatten(start_dim=-2)).transpose(-2, -3)
        norm = torch.einsum('...abc,...adc->...bdc', mask, mask)
        out = out/(norm +1e-3)
        return out

class RowAttentionWithPairBias(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_channel']
        num_heads = config['num_head']
        in_num_c = config['norm_channel']
        pair_rep_num_c = global_config['pair_channel']

        self.query_norm = nn.LayerNorm(in_num_c)
        self.feat_2d_norm = nn.LayerNorm(pair_rep_num_c)
        self.q = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.k = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.v = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.feat_2d_weights = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.output = nn.Linear(attn_num_c * num_heads, in_num_c)
        self.gate = nn.Linear(in_num_c, attn_num_c * num_heads)
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads
    
    def forward(self, msa_act, pair_act, msa_mask):
        msa_act = self.query_norm(msa_act)
        pair_act = self.feat_2d_norm(pair_act)
        nonbatched_bias = self.feat_2d_weights(pair_act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        bias = (1e9 * (msa_mask - 1.))[...,:, None, None, :]
        q = self.q(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c)
        k = self.k(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c)
        v = self.v(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c)
        factor = 1 / math.sqrt(self.attn_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k) + bias
        weights = torch.softmax(aff + nonbatched_bias, dim=-1)
        gate = torch.sigmoid(self.gate(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c))

        out_1d = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v) * gate
        out_1d = self.output(out_1d.flatten(start_dim=-2))
        return out_1d

class ExtraColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.attn_num_c = config['attention_channel']
        self.num_heads = config['num_head']

        self.query_norm = nn.LayerNorm(global_config['extra_msa_channel'])
        self.q = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c*self.num_heads, bias=False)
        self.k = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c, bias=False)
        self.v = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c, bias=False)
        self.gate = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c * self.num_heads)
        self.output = nn.Linear(self.attn_num_c * self.num_heads, global_config['extra_msa_channel'])

    def forward(self, msa_act, msa_mask):
        msa_act = msa_act.transpose(-2,-3)
        msa_mask = msa_mask.transpose(-1,-2)
        msa_act = self.query_norm(msa_act)
        q_avg = torch.sum(msa_act, dim=-2)/msa_act.shape[-2]
        q = self.q(q_avg).view(*q_avg.shape[:-1], self.num_heads, self.attn_num_c)
        q = q*(self.attn_num_c ** (-0.5))
        k = self.k(msa_act)
        v = self.v(msa_act)
        gate =  torch.sigmoid(self.gate(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c))
        w = torch.softmax(torch.einsum('bihc,bikc->bihk', q, k), dim=-1)
        out_1d = torch.einsum('bmhk,bmkc->bmhc', w, v)
        out_1d = out_1d.unsqueeze(-3) * gate
        out = self.output(out_1d.view(*out_1d.shape[:-2], self.attn_num_c * self.num_heads))
        return out.transpose(-2,-3)

class LigColumnAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_channel']
        num_heads = config['num_head']
        in_num_c = global_config['msa_channel']

        self.query_norm = nn.LayerNorm(in_num_c)
        self.q = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.k = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.v = nn.Linear(in_num_c, attn_num_c*num_heads, bias=False)
        self.output = nn.Linear(attn_num_c * num_heads, in_num_c)
        self.gate = nn.Linear(in_num_c, attn_num_c * num_heads)

        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

    def forward(self, msa_act, msa_mask):
        msa_act = msa_act.transpose(-2,-3)
        msa_mask = msa_mask.transpose(-1,-2)
        bias = (1e9 * (msa_mask - 1.))[...,:, None, None, :]
        msa_act = self.query_norm(msa_act)
        gate = torch.sigmoid(self.gate(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c))
        q = self.q(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c)
        k = self.k(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c)
        v = self.v(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c)
        factor = 1 / math.sqrt(self.attn_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k) + bias
        weights = torch.softmax(aff, dim=-1)
        out_1d = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v) * gate
        out_1d = self.output(out_1d.flatten(start_dim=-2))
        out_1d = out_1d.transpose(-2,-3)

        return out_1d

class Transition(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(config['norm_channel'])
        self.transition1 = nn.Linear(config['norm_channel'], config['norm_channel'] * config['num_intermediate_factor'])
        self.transition2 = nn.Linear(config['norm_channel'] * config['num_intermediate_factor'], config['norm_channel'])

    def forward(self, act, mask):
        act = self.input_layer_norm(act)
        act = self.transition1(act).relu_()
        act = self.transition2(act)
        return act

class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = config['norm_channel']
        mid_c = config['num_intermediate_channel']
        self.layer_norm_input = nn.LayerNorm(in_c)
        self.center_layer_norm = nn.LayerNorm(mid_c)
        self.left_projection = nn.Linear(in_c, mid_c)
        self.right_projection = nn.Linear(in_c, mid_c)
        self.left_gate = nn.Linear(in_c, mid_c)
        self.right_gate = nn.Linear(in_c, mid_c)
        self.output_projection = nn.Linear(mid_c, in_c)
        self.gating_linear = nn.Linear(in_c, in_c)

    def forward(self, act, mask):
        act = self.layer_norm_input(act)
        mask = mask[..., None]
        left_proj = mask*self.left_projection(act) * torch.sigmoid(self.left_gate(act))
        right_proj = mask*self.right_projection(act) * torch.sigmoid(self.right_gate(act))
        out = torch.einsum('bikc,bjkc->bijc', left_proj, right_proj)
        out = self.center_layer_norm(out)
        out = self.output_projection(out)
        out = out * torch.sigmoid(self.gating_linear(act))
        return out

class TriangleMultiplicationIngoing(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = config['norm_channel']
        mid_c = config['num_intermediate_channel']
        self.layer_norm_input = nn.LayerNorm(in_c)
        self.center_layer_norm = nn.LayerNorm(mid_c)
        self.left_projection = nn.Linear(in_c, mid_c)
        self.right_projection = nn.Linear(in_c, mid_c)
        self.left_gate = nn.Linear(in_c, mid_c)
        self.right_gate = nn.Linear(in_c, mid_c)
        self.output_projection = nn.Linear(mid_c, in_c)
        self.gating_linear = nn.Linear(in_c, in_c)

    def forward(self, act, mask):
        act = self.layer_norm_input(act)
        mask = mask[..., None]
        left_proj = mask*self.left_projection(act) * torch.sigmoid(self.left_gate(act))
        right_proj = mask*self.right_projection(act) * torch.sigmoid(self.right_gate(act))    
        out = torch.einsum('bkjc,bkic->bijc', left_proj, right_proj)
        out = self.center_layer_norm(out)
        out = self.output_projection(out)
        out = out * torch.sigmoid(self.gating_linear(act))
        return out

class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attn_num_c = config['attention_channel']
        num_heads = config['num_head']
        num_in_c = config['norm_channel']
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attn_num_c*num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attn_num_c*num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attn_num_c*num_heads, bias=False)
        self.feat_2d_weights = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attn_num_c * num_heads)
        self.output = nn.Linear(attn_num_c * num_heads, num_in_c)

    def forward(self, act, mask):
        act = self.query_norm(act)
        bias = (1e9 * (mask - 1.))[...,:, None, None, :]

        q = self.q(act).view(*act.shape[:-1], self.num_heads, self.attn_num_c)
        k = self.k(act).view(*act.shape[:-1], self.num_heads, self.attn_num_c)
        v = self.v(act).view(*act.shape[:-1], self.num_heads, self.attn_num_c)
        factor = 1 / math.sqrt(self.attn_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k) + bias
        nonbatched_bias = self.feat_2d_weights(act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        weights = torch.softmax(aff + nonbatched_bias, dim=-1)
        g = torch.sigmoid(self.gate(act).view(*act.shape[:-1], self.num_heads, self.attn_num_c))
        out = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v)*g

        out = self.output(out.flatten(start_dim=-2))

        return out

class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_channel']
        num_heads = config['num_head']
        num_in_c = config['norm_channel']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attention_num_c*num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attention_num_c*num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attention_num_c*num_heads, bias=False)
        self.feat_2d_weights = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attention_num_c * num_heads)
        self.output = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, act, mask):
        act = act.transpose(-2,-3)
        act = self.query_norm(act)
        mask = mask.transpose(-1,-2)
        bias = (1e9 * (mask - 1.))[...,:, None, None, :]

        q = self.q(act).view(*act.shape[:-1], self.num_heads, self.attention_num_c)
        k = self.k(act).view(*act.shape[:-1], self.num_heads, self.attention_num_c)
        v = self.v(act).view(*act.shape[:-1], self.num_heads, self.attention_num_c)
        factor = 1 / math.sqrt(self.attention_num_c)
        aff = torch.einsum('bmihc,bmjhc->bmhij', q*factor, k) + bias
        nonbatched_bias = self.feat_2d_weights(act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        weights = torch.softmax(aff + nonbatched_bias, dim=-1)
        g = torch.sigmoid(self.gate(act).view(*act.shape[:-1], self.num_heads, self.attention_num_c))
        out = torch.einsum('bmhqk,bmkhc->bmqhc', weights, v)*g

        out = self.output(out.flatten(start_dim=-2))
        out = out.transpose(-2,-3)

        return out

class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prev_pos_linear = nn.Linear(config['model']['embeddings_and_evoformer']['prev_pos']['num_bins'], config['model']['embeddings_and_evoformer']['pair_channel'])
        self.max_bin = config['model']['embeddings_and_evoformer']['prev_pos']['max_bin']
        self.min_bin = config['model']['embeddings_and_evoformer']['prev_pos']['min_bin']
        self.num_bins = config['model']['embeddings_and_evoformer']['prev_pos']['num_bins']
        self.config = config
        self.prev_pair_norm = nn.LayerNorm(config['model']['embeddings_and_evoformer']['pair_channel'])
        self.prev_msa_first_row_norm = nn.LayerNorm(config['model']['embeddings_and_evoformer']['msa_channel'])
        self.position_activations = nn.Linear(config['rel_feat'], config['model']['embeddings_and_evoformer']['pair_channel'])

    def _relative_encoding(self, batch):
        c = self.config['model']['embeddings_and_evoformer']
        rel_feats = []
        pos = batch['residue_index']
        asym_id = batch['asym_id']
        asym_id_same = torch.eq(asym_id[..., None], asym_id[...,None, :])
        offset = pos[..., None] - pos[...,None, :]

        clipped_offset = torch.clip(offset + c['max_relative_idx'], min=0, max=2 * c['max_relative_idx'])

        if c['use_chain_relative']:
            final_offset = torch.where(asym_id_same, clipped_offset,
                               (2 * c['max_relative_idx'] + 1) *
                               torch.ones_like(clipped_offset))

            rel_pos = torch.nn.functional.one_hot(final_offset.long(), 2 * c['max_relative_idx'] + 2)

            rel_feats.append(rel_pos)

            entity_id = batch['entity_id']
            entity_id_same = torch.eq(entity_id[..., None], entity_id[...,None, :])
            rel_feats.append(entity_id_same.type(rel_pos.dtype)[..., None])

            sym_id = batch['sym_id']
            rel_sym_id = sym_id[..., None] - sym_id[...,None, :]

            max_rel_chain = c['max_relative_chain']

            clipped_rel_chain = torch.clip(rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)

            final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                  (2 * max_rel_chain + 1) *
                                  torch.ones_like(clipped_rel_chain))
            rel_chain = torch.nn.functional.one_hot(final_rel_chain.long(), 2 * c['max_relative_chain'] + 2)

            rel_feats.append(rel_chain)

        else:
            rel_pos = torch.nn.functional.one_hot(clipped_offset.long(), 2 * c['max_relative_idx'] + 1)
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, -1)
        return rel_feat

        #return common_modules.Linear(
        #c.pair_channel,
        #name='position_activations')(
        #    rel_feat)

    def forward(self, batch, recycle):
        prev_pseudo_beta = pseudo_beta_fn(batch['aatype'], recycle['prev_pos'], None)
        dgram = torch.sum((prev_pseudo_beta[..., None, :] - prev_pseudo_beta[..., None, :, :]) ** 2, dim=-1, keepdim=True)
        lower = torch.linspace(self.min_bin, self.max_bin, self.num_bins, device=prev_pseudo_beta.device) ** 2
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
        prev_pos_linear = self.prev_pos_linear(dgram)
        pair_activation_update = prev_pos_linear + self.prev_pair_norm(recycle['prev_pair'])
        rel_feat = self._relative_encoding(batch)
        pair_activation_update += self.position_activations(rel_feat.float())
        prev_msa_first_row = self.prev_msa_first_row_norm(recycle['prev_msa_first_row'])

        return prev_msa_first_row, pair_activation_update

class FragExtraStackIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['msa_row_attention_with_pair_bias'], global_config)
        self.ExtraColumnGlobalAttention = ExtraColumnGlobalAttention(config['msa_column_attention'], global_config)
        self.RecTransition = Transition(config['msa_transition'], global_config)
        self.OuterProductMean = OuterProductMean(config['outer_product_mean'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['triangle_multiplication_outgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['triangle_multiplication_incoming'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['triangle_attention_starting_node'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['triangle_attention_ending_node'], global_config)
        self.PairTransition = Transition(config['pair_transition'], global_config)

    def forward(self, extra_msa_stack_inputs, extra_mask):
        msa_act, pair_act = extra_msa_stack_inputs['msa'], extra_msa_stack_inputs['pair']
        msa_mask, pair_mask = extra_mask['msa'], extra_mask['pair']
        pair_act += self.OuterProductMean(msa_act, msa_mask)
        msa_act += self.RowAttentionWithPairBias(msa_act, pair_act, msa_mask)
        msa_act += self.ExtraColumnGlobalAttention(msa_act, msa_mask)
        msa_act += self.RecTransition(msa_act, msa_mask)
        pair_act += self.TriangleMultiplicationOutgoing(pair_act, pair_mask)
        pair_act += self.TriangleMultiplicationIngoing(pair_act, pair_mask)
        pair_act += self.TriangleAttentionStartingNode(pair_act, pair_mask)
        pair_act += self.TriangleAttentionEndingNode(pair_act, pair_mask)
        pair_act += self.PairTransition(pair_act, pair_mask)

        return {'msa': msa_act, 'pair': pair_act}


class FragExtraStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([FragExtraStackIteration(config['model']['embeddings_and_evoformer']['extra_msa'], config['model']['embeddings_and_evoformer']) for _ in range(config['model']['embeddings_and_evoformer']['extra_msa_stack_num_block'])])
    
    def forward(self, extra_msa_stack_inputs, extra_mask):
        for l in self.layers:
            extra_msa_stack_inputs = checkpoint(l, extra_msa_stack_inputs, extra_mask)
        return extra_msa_stack_inputs

class EvoformerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['msa_row_attention_with_pair_bias'], global_config)
        self.LigColumnAttention = LigColumnAttention(config['msa_column_attention'], global_config)
        self.RecTransition = Transition(config['msa_transition'], global_config)
        self.OuterProductMean = OuterProductMean(config['outer_product_mean'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['triangle_multiplication_outgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['triangle_multiplication_incoming'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['triangle_attention_starting_node'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['triangle_attention_ending_node'], global_config)
        self.PairTransition = Transition(config['pair_transition'], global_config)

    def forward(self, evoformer_inputs, evoformer_mask):
        msa_act, pair_act = evoformer_inputs['msa'], evoformer_inputs['pair']
        msa_mask, pair_mask = evoformer_mask['msa'], evoformer_mask['pair']
        pair_act += self.OuterProductMean(msa_act, msa_mask)
        msa_act += self.RowAttentionWithPairBias(msa_act, pair_act, msa_mask)
        msa_act += self.LigColumnAttention(msa_act, msa_mask)
        msa_act += self.RecTransition(msa_act, msa_mask)
        pair_act += self.TriangleMultiplicationOutgoing(pair_act, pair_mask)
        pair_act += self.TriangleMultiplicationIngoing(pair_act, pair_mask)
        pair_act += self.TriangleAttentionStartingNode(pair_act, pair_mask)
        pair_act += self.TriangleAttentionEndingNode(pair_act, pair_mask)
        pair_act += self.PairTransition(pair_act, pair_mask)
        
        return {'msa': msa_act, 'pair': pair_act}

        
class InputEmbedding(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.preprocessing_1d = nn.Linear(global_config['aatype'], global_config['model']['embeddings_and_evoformer']['msa_channel'])
        self.left_single = nn.Linear(global_config['aatype'], global_config['model']['embeddings_and_evoformer']['pair_channel'])
        self.right_single = nn.Linear(global_config['aatype'], global_config['model']['embeddings_and_evoformer']['pair_channel'])
        self.preprocess_msa = nn.Linear(global_config['msa'], global_config['model']['embeddings_and_evoformer']['msa_channel'])
        self.max_seq = global_config['model']['embeddings_and_evoformer']['num_msa']
        self.msa_channel = global_config['model']['embeddings_and_evoformer']['msa_channel']
        self.pair_channel = global_config['model']['embeddings_and_evoformer']['pair_channel']
        self.num_extra_msa = global_config['model']['embeddings_and_evoformer']['num_extra_msa']
        self.global_config = global_config

        self.RecyclingEmbedder = RecyclingEmbedder(global_config)
        self.extra_msa_activations = nn.Linear(global_config['extra_msa_act'], global_config['model']['embeddings_and_evoformer']['extra_msa_channel'])
        self.FragExtraStack = FragExtraStack(global_config)

    def forward(self, batch, recycle=None):
        num_batch, num_res = batch['aatype'].shape[0], batch['aatype'].shape[1]
        batch_profile = torch.sum(batch['msa_mask'][...,None] * nn.functional.one_hot(batch['msa'].long(), 22), 1)/(torch.sum(batch['msa_mask'][...,None], 1)+ 1e-10)
        target_feat = nn.functional.one_hot(batch['aatype'].long(), 21).float()
        preprocessed_1d = self.preprocessing_1d(target_feat)
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        pair_activations = left_single.unsqueeze(2) + right_single.unsqueeze(1)
        batch = sample_msa(batch, self.max_seq)
        batch = make_masked_msa(batch)
        (batch['cluster_profile'], batch['cluster_deletion_mean']) = nearest_neighbor_clusters(batch)
        msa_feat = create_msa_feat(batch)
        preprocess_msa = self.preprocess_msa(msa_feat)
        msa_activations = preprocess_msa + preprocessed_1d
        if(self.global_config['recycle'] and recycle is None):
            recycle = {
                    'prev_pos': torch.zeros(num_batch, num_res, 37, 3).to('cuda:0'),
                    'prev_msa_first_row': torch.zeros(num_batch, num_res, self.msa_channel).to('cuda:0'),
                    'prev_pair': torch.zeros(num_batch, num_res, num_res, self.pair_channel).to('cuda:0')
                    }

        if(recycle is not None):
            prev_msa_first_row, pair_activation_update = self.RecyclingEmbedder(batch, recycle)
            pair_activations += pair_activation_update
            msa_activations[:,0] += prev_msa_first_row

        extra_msa_feat, extra_msa_mask = create_extra_msa_feature(batch, self.num_extra_msa)
        mask_2d = batch['seq_mask'][..., None] * batch['seq_mask'][...,None, :]
        mask_2d = mask_2d.type(torch.float32)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        extra_msa_stack_inputs = {
                'msa': extra_msa_activations,
                'pair': pair_activations
                }
        extra_masks = {
                'msa': extra_msa_mask.type(torch.float32),
                'pair': mask_2d
                }
        extra_msa_output = self.FragExtraStack(extra_msa_stack_inputs, extra_masks)
        return {'msa': msa_activations, 'pair': extra_msa_output['pair'], 'msa_mask': batch['msa_mask'], 'pair_mask': mask_2d}


class DockerIteration(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.InputEmbedder = InputEmbedding(global_config).to('cuda:0')
        self.Evoformer = nn.ModuleList([EvoformerIteration(global_config['model']['embeddings_and_evoformer']['evoformer'], global_config['model']['embeddings_and_evoformer']).to('cuda:0') for _ in range(global_config['model']['embeddings_and_evoformer']['evoformer_num_block'])])
        self.EvoformerExtractSingleRec = nn.Linear(global_config['model']['embeddings_and_evoformer']['msa_channel'], global_config['model']['embeddings_and_evoformer']['seq_channel']).to('cuda:0')
        self.StructureModule = structure_multimer.StructureModule(global_config['model']['heads']['structure_module'], global_config['model']['embeddings_and_evoformer']).to('cuda:0')

    def forward(self, batch, recycle=None):
        evoformer_inp = self.InputEmbedder(batch, recycle=recycle)
        evoformer_inputs = {'msa': evoformer_inp['msa'], 'pair': evoformer_inp['pair']}
        evoformer_mask = {'msa': evoformer_inp['msa_mask'], 'pair': evoformer_inp['pair_mask']}

        def checkpoint_fun(function):
            return lambda a, b: function(a, b)

        for evo_i, evo_iter in enumerate(self.Evoformer):
            evoformer_out = checkpoint(checkpoint_fun(evo_iter), evoformer_inputs, evoformer_mask)

        msa_activations = evoformer_out['msa'] #evoformer_inputs['msa'] #evoformer_out['msa']
        pair_activations = evoformer_out['pair'] #evoformer_inputs['pair'] #evoformer_out['pair']
        single_activations = self.EvoformerExtractSingleRec(msa_activations[:,0])
        representations = {'single': single_activations, 'pair': pair_activations}
        struct_out = self.StructureModule(representations, batch)
        atom14_pred_positions = struct_out['sc']['atom_pos'][-1]
        atom37_pred_positions = all_atom_multimer.atom14_to_atom37(atom14_pred_positions.squeeze(0), batch['aatype'].squeeze(0).long())
        atom37_mask = all_atom_multimer.atom_37_mask(batch['aatype'][0].long())

        out_dict = {}
        out_dict['final_all_atom'] = atom37_pred_positions
        out_dict['struct_out'] = struct_out
        out_dict['final_atom_mask'] = atom37_mask
        out_dict['recycling_input'] = {
            'prev_msa_first_row': msa_activations[:,0],
            'prev_pair': pair_activations,
            'prev_pos': atom37_pred_positions.unsqueeze(0)
        }

        return out_dict









