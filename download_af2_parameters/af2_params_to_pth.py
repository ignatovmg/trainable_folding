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


from enum import Enum
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
from typing import Union, List
import sys
import os
sys.path.insert(1, '../')
from alphadock import docker
from alphadock import config


_NPZ_KEY_PREFIX = "alphafold/alphafold_iteration/"


class ParamType(Enum):
    LinearWeight = partial(  # hack: partial prevents fns from becoming methods
        lambda w: w.transpose(-1, -2)
    )
    LinearWeightMHA = partial(
        lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2)
    )
    LinearMHAOutputWeight = partial(
        lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2)
    )
    LinearBiasMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearWeightOPM = partial(
        lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2)
    )
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn


@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False


def _process_translations_dict(d, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            prefix = _NPZ_KEY_PREFIX if top_layer else ""
            sub_flat = {
                (prefix + "/".join([k, k_prime])): v_prime
                for k_prime, v_prime in _process_translations_dict(
                    v, top_layer=False
                ).items()
            }
            flat.update(sub_flat)
        else:
            k = "/" + k if not top_layer else k
            flat[k] = v

    return flat


def stacked(param_dict_list, out=None):
    """
    Args:
        param_dict_list:
            A list of (nested) Param dicts to stack. The structure of
            each dict must be the identical (down to the ParamTypes of
            "parallel" Params). There must be at least one dict
            in the list.
    """
    if out is None:
        out = {}
    template = param_dict_list[0]
    for k, _ in template.items():
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = {}
            stacked(v, out=out[k])
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )

            out[k] = stacked_param

    return out


def assign(translation_dict, orig_weights):
    for k, param in translation_dict.items():
        with torch.no_grad():
            weights = torch.as_tensor(orig_weights[k])
            ref, param_type = param.param, param.param_type
            if param.stacked:
                weights = torch.unbind(weights, 0)
            else:
                weights = [weights]
                ref = [ref]

            try:
                weights = list(map(param_type.transformation, weights))
                for p, w in zip(ref, weights):
                    p.copy_(w)
            except:
                print(k)
                print(ref[0].shape)
                print(weights[0].shape)
                raise


def import_jax_weights_(model, version="model_1"):

    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))

    LinearBias = lambda l: (Param(l))
    
    LinearWeightMHA = lambda l: (Param(l, param_type=ParamType.LinearWeightMHA))

    LinearBiasMHA = lambda b: (Param(b, param_type=ParamType.LinearBiasMHA))

    LinearWeightOPM = lambda l: (Param(l, param_type=ParamType.LinearWeightOPM))

    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }
    LayerNormParams = lambda l: {
        "scale": Param(l.weight),
        "offset": Param(l.bias),
    }
    MSAAttParams = lambda matt: {
        "query_norm": LayerNormParams(matt.norm),
        "attention": 
        {
            "query_w": LinearWeightMHA(matt.q.weight),
            "key_w": LinearWeightMHA(matt.k.weight),
            "value_w": LinearWeightMHA(matt.v.weight),
            "gating_w": LinearWeightMHA(matt.gate.weight),
            "gating_b": LinearBiasMHA(matt.gate.bias),
            "output_w": Param(
                matt.final.weight,
                param_type=ParamType.LinearMHAOutputWeight,
            ),
            "output_b": LinearBias(matt.final.bias),
        }
    }
    MSAAttPairBiasParams = lambda matt: dict(
        **MSAAttParams(matt),
        **{
            "feat_2d_norm": LayerNormParams(matt.norm_2d),
            "feat_2d_weights": LinearWeight(matt.x2d_project.weight),
        },
    )
    GlobalColMSAAttParams = lambda matt: {
        "query_norm": LayerNormParams(matt.norm),
        "attention":
        {
            "query_w": LinearWeightMHA(matt.q.weight),
            "key_w": LinearWeight(matt.k.weight),
            "value_w": LinearWeight(matt.v.weight),
            "gating_w": LinearWeightMHA(matt.gate.weight),
            "gating_b": LinearBiasMHA(matt.gate.bias),
            "output_w": Param(
                matt.final.weight,
                param_type=ParamType.LinearMHAOutputWeight,
            ),
            "output_b": LinearBias(matt.final.bias),
        }
    }

    MSATransitionParams = lambda m: {
        "input_layer_norm": LayerNormParams(m.norm),
        "transition1": LinearParams(m.l1),
        "transition2": LinearParams(m.l2),
    }
    OuterProductMeanParams = lambda o: {
        "layer_norm_input": LayerNormParams(o.norm),
        "left_projection": LinearParams(o.proj_left),
        "right_projection": LinearParams(o.proj_right),
        "output_w": LinearWeightOPM(o.final.weight),
        "output_b": LinearBias(o.final.bias),
    }
    TriMulOutParams = lambda tri_mul: {
        "layer_norm_input": LayerNormParams(tri_mul.norm1),
        "left_projection": LinearParams(tri_mul.l1i),
        "right_projection": LinearParams(tri_mul.l1j),
        "left_gate": LinearParams(tri_mul.l1i_sigm),
        "right_gate": LinearParams(tri_mul.l1j_sigm),
        "center_layer_norm": LayerNormParams(tri_mul.norm2),
        "output_projection": LinearParams(tri_mul.l2_proj),
        "gating_linear": LinearParams(tri_mul.l3_sigm),
    }
    TriMulInParams = lambda tri_mul: {
        "layer_norm_input": LayerNormParams(tri_mul.norm1),
        "left_projection": LinearParams(tri_mul.l1i),
        "right_projection": LinearParams(tri_mul.l1j),
        "left_gate": LinearParams(tri_mul.l1i_sigm),
        "right_gate": LinearParams(tri_mul.l1j_sigm),
        "center_layer_norm": LayerNormParams(tri_mul.norm2),
        "output_projection": LinearParams(tri_mul.l2_proj),
        "gating_linear": LinearParams(tri_mul.l3_sigm),
    }
    TriAttParams = lambda tri_att: {
        "query_norm": LayerNormParams(tri_att.norm),
        "feat_2d_weights": LinearWeight(tri_att.bias.weight),
        "attention": {
            "query_w": LinearWeightMHA(tri_att.q.weight),
            "key_w": LinearWeightMHA(tri_att.k.weight),
            "value_w": LinearWeightMHA(tri_att.v.weight),
            "gating_w": LinearWeightMHA(tri_att.gate.weight),
            "gating_b": LinearBiasMHA(tri_att.gate.bias),
            "output_w": Param(
                tri_att.out.weight,
                param_type=ParamType.LinearMHAOutputWeight,
            ),
            "output_b": LinearBias(tri_att.out.bias),
        }
    }

    IPAParams = lambda ipa: {
        "q_scalar": LinearParams(ipa.q),
        "kv_scalar": LinearParams(ipa.kv),
        "q_point_local": LinearParams(ipa.q_points),
        "kv_point_local": LinearParams(ipa.kv_points),
        "trainable_point_weights": Param(
            param=ipa.trainable_w, param_type=ParamType.Other
        ),
        "attention_2d": LinearParams(ipa.rr_kqv_2d),
        "output_projection": LinearParams(ipa.final_r),
    }

    FoldIterationParams = lambda sm: {
        "invariant_point_attention": IPAParams(sm.InvariantPointAttention),
        "attention_layer_norm": LayerNormParams(sm.rec_norm),
        "transition": LinearParams(sm.transition_r[0]),
        "transition_1": LinearParams(sm.transition_r[2]),
        "transition_2": LinearParams(sm.transition_r[4]),
        "transition_layer_norm": LayerNormParams(sm.rec_norm2),
        "affine_update": LinearParams(sm.backbone_update),
        "rigid_sidechain": {
            "input_projection": LinearParams(sm.PredictSidechains.s_cur),
            "input_projection_1": LinearParams(sm.PredictSidechains.s_ini),
            "resblock1": LinearParams(sm.PredictSidechains.res1[1]),
            "resblock2": LinearParams(sm.PredictSidechains.res1[3]),
            "resblock1_1": LinearParams(sm.PredictSidechains.res2[1]),
            "resblock2_1": LinearParams(sm.PredictSidechains.res2[3]),
            "unnormalized_angles": LinearParams(sm.PredictSidechains.final[1]),
        },
    }

    TemplatePairBlockParams = lambda b: {
        "triangle_attention_starting_node": TriAttParams(b.TriangleAttentionStartingNode),
        "triangle_attention_ending_node": TriAttParams(b.TriangleAttentionEndingNode),
        "triangle_multiplication_outgoing": TriMulOutParams(b.TriangleMultiplicationOutgoing),
        "triangle_multiplication_incoming": TriMulInParams(b.TriangleMultiplicationIngoing),
        "pair_transition": MSATransitionParams(b.PairTransition),
    }

    def EvoformerBlockParams(b, is_extra_msa=False):
        if is_extra_msa:
           col_att_name = "msa_column_global_attention"
           msa_col_att_params = GlobalColMSAAttParams(b.MSAColumnGlobalAttention)
        else:
           col_att_name = "msa_column_attention"
           msa_col_att_params = MSAAttParams(b.MSAColumnAttention)

        d = {
            "msa_row_attention_with_pair_bias": MSAAttPairBiasParams(
                b.RowAttentionWithPairBias
            ),
            col_att_name: msa_col_att_params,
            "msa_transition": MSATransitionParams(b.MSATransition),
            "outer_product_mean":
                OuterProductMeanParams(b.OuterProductMean),
            "triangle_multiplication_outgoing":
                TriMulOutParams(b.TriangleMultiplicationOutgoing),
            "triangle_multiplication_incoming":
                TriMulInParams(b.TriangleMultiplicationIngoing),
           "triangle_attention_starting_node":
               TriAttParams(b.TriangleAttentionStartingNode),
           "triangle_attention_ending_node":
               TriAttParams(b.TriangleAttentionEndingNode),
           "pair_transition":
               MSATransitionParams(b.PairTransition),
        }

        return d
    ExtraMSABlockParams = partial(EvoformerBlockParams, is_extra_msa=True)
    
    tps_blocks = model.InputEmbedder.TemplatePairStack.layers
    tps_blocks_params = stacked([TemplatePairBlockParams(b) for b in tps_blocks])

    ems_blocks = model.InputEmbedder.ExtraMsaStack.layers
    ems_blocks_params = stacked([ExtraMSABlockParams(b) for b in ems_blocks])

    evo_blocks = model.Evoformer
    evo_blocks_params = stacked([EvoformerBlockParams(b) for b in evo_blocks])

    translations = {
        "evoformer": {
            "preprocess_1d": LinearParams(model.InputEmbedder.rec_1d_project),
            "preprocess_msa": LinearParams(model.InputEmbedder.main_msa_project),
            "left_single": LinearParams(model.InputEmbedder.InitPairRepresentation.r_proj1),
            "right_single": LinearParams(model.InputEmbedder.InitPairRepresentation.r_proj2),
            "prev_pos_linear": LinearParams(model.InputEmbedder.RecyclingEmbedder.rr_proj),
            "prev_msa_first_row_norm": LayerNormParams(
               model.InputEmbedder.RecyclingEmbedder.rec_norm
            ),
            "prev_pair_norm": LayerNormParams(
               model.InputEmbedder.RecyclingEmbedder.x2d_norm
            ),
            "pair_activiations": LinearParams(
                model.InputEmbedder.InitPairRepresentation.relpos_proj
            ),
            "template_embedding": {
                "single_template_embedding": {
                    "embedding2d": LinearParams(
                        model.InputEmbedder.TemplatePairStack.rr_proj
                    ),
                    "template_pair_stack": {
                        "__layer_stack_no_state": tps_blocks_params,
                    },
                    "output_layer_norm": LayerNormParams(
                        model.InputEmbedder.TemplatePairStack.norm
                    ),
                },
                "attention": {
                    "query_w": LinearWeightMHA(model.InputEmbedder.TemplatePointwiseAttention.q.weight),
                    "key_w": LinearWeightMHA(model.InputEmbedder.TemplatePointwiseAttention.k.weight),
                    "value_w": LinearWeightMHA(model.InputEmbedder.TemplatePointwiseAttention.v.weight),
                    "output_w": Param(
                        model.InputEmbedder.TemplatePointwiseAttention.out.weight,
                        param_type=ParamType.LinearMHAOutputWeight,
                    ),
                    "output_b": LinearBias(model.InputEmbedder.TemplatePointwiseAttention.out.bias),
                }

            },
            "extra_msa_activations": LinearParams(
                model.InputEmbedder.ExtraMsaStack.project
            ),
            "extra_msa_stack": ems_blocks_params,
            "template_single_embedding": LinearParams(
                model.TemplateAngleEmbedder.template_single_embedding
            ),
            "template_projection": LinearParams(
                model.TemplateAngleEmbedder.template_projection
            ),
            "evoformer_iteration": evo_blocks_params,
            "single_activations": LinearParams(model.EvoformerExtractSingle),
        },
        "structure_module": {
            "single_layer_norm": LayerNormParams(
                model.StructureModule.norm_rec_1d_init
            ),
            "initial_projection": LinearParams(
                model.StructureModule.rec_1d_proj
            ),
            "pair_layer_norm": LayerNormParams(
                model.StructureModule.norm_2d_init
            ),
            "fold_iteration": FoldIterationParams(model.StructureModule.StructureModuleIteration),
        },
        "predicted_lddt_head": {
            "input_layer_norm": LayerNormParams(
                model.StructureModule.StructureModuleIteration.PredictRecLDDT.layers[0]
            ),
            "act_0": LinearParams(model.StructureModule.StructureModuleIteration.PredictRecLDDT.layers[1]),
            "act_1": LinearParams(model.StructureModule.StructureModuleIteration.PredictRecLDDT.layers[3]),
            "logits": LinearParams(model.StructureModule.StructureModuleIteration.PredictRecLDDT.layers[5]),
        },
        "distogram_head": {
            "half_logits": LinearParams(model.StructureModule.pred_distogram.rr_proj),
        },
    }
    flat = _process_translations_dict(translations)
    assign(flat, data)


if __name__ == '__main__':
    param_path = sys.argv[1]
    data = np.load(param_path)
    model = docker.DockerIteration(config.config['model'], config.config)
    import_jax_weights_(model)
    
    param_out_path = os.path.basename(param_path)[:-4] + ".pth"
    torch.save({'model_state_dict': model.state_dict()}, param_out_path)
