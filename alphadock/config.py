from path import Path
import numpy as np

DATA_DIR = Path(__file__).abspath().dirname().dirname() / 'data_preparation' / 'data' / 'small_debug_set'
SRC_DIR = Path(__file__).abspath().dirname()

TEST_DATA_DIR = Path(__file__).abspath().dirname() / 'test_data'

DTYPE_FLOAT = np.float32

DTYPE_INT = np.int64

config = {
    'rec_in_c': 22,
    'msa_extra_in_c': 25,
    'msa_main_in_c': 49,
    'rec_relpos_c': 65,
    'hh_rec': 24,
    'hh_rr': 88,
    'temp_single_emb_c': 57,

    'position_scale': 10,
    'num_torsions': 7,
    'rep_1d': {
        'num_c': 256
    },
    'rep_2d': {
        'num_c': 128
    },
    'num_single_c': 384,
    'extra_msa_channel': 64,

    'recycling_on': True,
    'recycling_num_iter': 3,
    'embed_torsion_angles': True,

    'loss': {
        'loss_aa_rec_rec_weight': 0.5,       # L_FAPE

        'loss_bb_rec_rec_weight': 0.5,       #
        'loss_chi_value_weight': 0.5,        # L_aux
        'loss_chi_norm_weight': 0.5 * 0.02,  #

        'loss_rec_rec_lddt_weight': 0.01,    # L_conf

        'loss_pred_dmat_rr_weight': 0.3,     # L_dist

        'loss_violation_weight': 0.0,        # L_viol

        'lddt_rec_bin_size': 2,
        'fape_loss_unit_distance': 10.0,
        'fape_clamp_distance': 10.0,
        'violation_tolerance_factor': 12.0,
        'clash_overlap_tolerance': 1.5
    },

    'Evoformer': {
        'num_iter': 48,
        'device': 'cuda:0',
        'EvoformerIteration': {
            'checkpoint': True,
            'RowAttentionWithPairBias': {
                'attention_num_c': 32,
                'num_heads': 8,
                'extra_msa_channel': 256,
            },
            'LigColumnAttention': {
                'attention_num_c': 32,
                'num_heads': 8
            },
            'LigTransition': {
                'n': 4
            },
            'RecTransition': {
                'n': 4
            },
            'OuterProductMean': {
                'mid_c': 32,
                'extra_msa_channel': 256
            },
            'TriangleMultiplicationIngoing': {
                'mid_c': 128,
                'rep_2d_num_c': 128
            },
            'TriangleMultiplicationOutgoing': {
                'mid_c': 128,
                'rep_2d_num_c': 128
            },
            'TriangleAttentionStartingNode': {
                'attention_num_c': 32,
                'num_heads': 4,
                'rand_remove': 0.0,
                'rep_2d_num_c': 128
            },
            'TriangleAttentionEndingNode': {
                'attention_num_c': 32,
                'num_heads': 4,
                'rand_remove': 0.0,
                'rep_2d_num_c': 128
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
            'rec_min_dist': 3,       # originally 3.375
            'rec_max_dist': 21.75    # originally 21.375
        },
        'TemplatePairStack': {
            'num_iter': 2,
            'checkpoint': True,
            'device': 'cuda:0',
            'TemplatePairStackIteration': {
                #'checkpoint': True,
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 16,
                    'num_heads': 4,
                    'rand_remove': 0.25,
                    'value_dim': 64,
                    'rep_2d_num_c': 64
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 16,
                    'num_heads': 4,
                    'rand_remove': 0.25,
                    'value_dim': 64,
                    'rep_2d_num_c': 64
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 64,
                    'rep_2d_num_c': 64
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 64,
                    'rep_2d_num_c': 64
                },
                'PairTransition': {
                    'n': 2
                }
            }
        },
        'TemplatePointwiseAttention': {
            'device': 'cuda:0',
            'attention_num_c': 16,
            'num_heads': 4,
            'rep_2d_num_c': 64
        },
        'FragExtraStack': {
            'num_iter': 4,
            'device': 'cuda:0',
            'extra_msa_channel': 64,
            'FragExtraStackIteration': {
                'checkpoint': True,
                'extra_msa_channel': 64,
                'RowAttentionWithPairBias': {
                    'attention_num_c': 8,
                    'num_heads': 8,
                    'extra_msa_channel': 64,
                },
                'ExtraColumnGlobalAttention': {
                    'attention_num_c': 8,
                    'num_heads': 8
                },
                'RecTransition': {
                    'n': 4
                },
                'OuterProductMean': {
                    'mid_c': 32,
                    'extra_msa_channel': 64
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 128,
                    'rep_2d_num_c': 128
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 128,
                    'rep_2d_num_c': 128
                },
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'rand_remove': 0.0,
                    'rep_2d_num_c': 128
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'rand_remove': 0.0,
                    'rep_2d_num_c': 128
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
            'checkpoint': False,
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
