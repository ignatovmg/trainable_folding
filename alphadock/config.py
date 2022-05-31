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


from path import Path
import numpy as np

SRC_DIR = Path(__file__).abspath().dirname()
TEST_DATA_DIR = Path(__file__).abspath().dirname() / 'test_data'

DTYPE_FLOAT = np.float32
DTYPE_INT = np.int64

config = {
    'data': {
        'crop_size': 256,
        'target_af_compatible': True,
        'use_cache': True,
        'msa_max_clusters': 128,
        'msa_max_extra': 1024,
        'msa_block_del_num': 5,
        'msa_block_del_size': 0.3,
        'msa_random_replace_fraction': 0.15,
        'msa_uniform_prob': 0.1,
        'msa_profile_prob': 0.1,
        'msa_same_prob': 0.1,
        'msa_keep_true_msa': True,
        'template_max': 4,
        'template_use_prob': 0.5,
        'clamp_fape_prob': 0.9,

        'target_feat': 22,
        'msa_extra_feat': 25,
        'msa_clus_feat': 49,
        'relpos_max': 32,
        'hh_rec': 24,
        'hh_rr': 84,
    },

    'loss': {
        'compute_loss': True,

        'loss_fape_aa_weight': 0.5,          # L_FAPE

        'loss_fape_bb_weight': 0.5,          #
        'loss_chi_value_weight': 0.5,        # L_aux
        'loss_chi_norm_weight': 0.5 * 0.02,  #

        'loss_lddt_weight': 0.01,            # L_conf

        'loss_pred_dmat_weight': 0.3,        # L_dist

        'loss_msa_bert_weight': 2.0,         # L_msa

        'loss_violation_weight': 0.0,        # L_viol

        'lddt_bin_size': 2,
        'fape_loss_unit_distance': 10.0,
        'fape_clamp_distance': 10.0,
        'violation_tolerance_factor': 12.0,
        'clash_overlap_tolerance': 1.5
    },

    'model': {
        'recycling_on': True,
        'recycling_num_iter': 3,
        'position_scale': 10,
        'num_torsions': 7,
        'rep1d_feat': 256,
        'rep2d_feat': 128,
        'single_rep_feat': 384,
        'rep1d_extra_feat': 64,
        'msa_bert_block': True,

        'Evoformer': {
            'num_iter': 48,
            'device': 'cuda:0',
            'EvoformerIteration': {
                'checkpoint': True,
                'RowAttentionWithPairBias': {
                    'attention_num_c': 32,
                    'num_heads': 8,
                    'msa_extra_stack': False
                },
                'MSAColumnAttention': {
                    'attention_num_c': 32,
                    'num_heads': 8
                },
                'MSATransition': {
                    'n': 4
                },
                'OuterProductMean': {
                    'mid_c': 32,
                    'msa_extra_stack': False
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 128,
                    'ingoing': True
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 128,
                    'ingoing': False
                },
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'ending_node': False
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'ending_node': True
                },
                'PairTransition': {
                    'n': 4
                }
            }
        },
        'InputEmbedder': {
            'device': 'cuda:0',
            'RecyclingEmbedder': {
                'rec_num_bins': 15,
                'rec_min_dist': 3.375,
                'rec_max_dist': 21.375
            },
            'ExtraMsaStack': {
                'num_iter': 4,
                'device': 'cuda:0',
                'ExtraMsaStackIteration': {
                    'checkpoint': True,
                    'RowAttentionWithPairBias': {
                        'attention_num_c': 8,
                        'num_heads': 8,
                        'msa_extra_stack': True
                    },
                    'MSAColumnGlobalAttention': {
                        'attention_num_c': 8,
                        'num_heads': 8
                    },
                    'MSATransition': {
                        'n': 4
                    },
                    'OuterProductMean': {
                        'mid_c': 32,
                        'msa_extra_stack': True
                    },
                    'TriangleMultiplicationIngoing': {
                        'mid_c': 128,
                        'ingoing': True
                    },
                    'TriangleMultiplicationOutgoing': {
                        'mid_c': 128,
                        'ingoing': False
                    },
                    'TriangleAttentionStartingNode': {
                        'attention_num_c': 32,
                        'num_heads': 4,
                        'ending_node': False
                    },
                    'TriangleAttentionEndingNode': {
                        'attention_num_c': 32,
                        'num_heads': 4,
                        'ending_node': True
                    },
                    'PairTransition': {
                        'n': 4
                    }
                }
            }
        },
        'StructureModule': {
            'num_iter': 8,
            'device': 'cuda:0',
            'StructureModuleIteration': {
                'InvariantPointAttention': {
                    'num_head': 12,
                    'num_scalar_qk': 16,
                    'num_scalar_v': 16,
                    'num_point_qk': 4,
                    'num_point_v': 8,
                    'num_2d_qk': 16,
                    'num_2d_v': 16
                },
                'PredictSidechains': {
                    'num_c': 128
                },
                'PredictRecLDDT': {
                    'num_c': 128,
                    'num_bins': 50
                }
            },
            'PredictDistogram': {
                'rec_num_bins': 64,
                'rec_min_dist': 2,
                'rec_max_dist': 22
            }
        }
    }
}
