import torch
import sys
sys.path.insert(1, '../')
import math

from alphadock import docker
from alphadock import config
from alphadock import utils
from alphadock import all_atom
from copy import deepcopy
import torchvision
import pickle


config_diff = {
    'Evoformer': {
        #'num_iter': 8,
        'device': 'cuda:0'
    },
    'InputEmbedder': {
        'device': 'cuda:0',
        'TemplatePairStack': {
            'num_iter': 2,
            'device': 'cuda:0'
        },
        'TemplatePointwiseAttention': {
            'device': 'cuda:0',
            'attention_num_c': 16,
            'num_heads': 4
        }
    },
    'StructureModule': {
        'num_iter': 8,
        'device': 'cuda:0',
        'StructureModuleIteration': {
            'checkpoint': True
        }
    },
    'loss': {
        'loss_violation_weight': 0.0,
    }
}


config_summit = utils.merge_dicts(deepcopy(config.config), config_diff)

def pred_to_pdb(out_pdb, input_dict, out_dict):
    with open(out_pdb, 'w') as f:
        f.write(f'test.pred\n')
        serial = all_atom.atom14_to_pdb_stream(
            f,
            input_dict['target']['rec_aatype'][0].cpu(),
            out_dict['final_all_atom']['atom_pos_tensor'].detach().cpu(),
            chain='A',
            serial_start=1,
            resnum_start=1
        )


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(123456)
    #with open('features.pkl', 'rb') as f:
    #    inputs = pickle.load(f)

    with open('features_with_template.pkl', 'rb') as f2:
        inputs = pickle.load(f2)

    model = docker.DockerIteration(config_summit, config_summit)
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    num_recycles = config_summit['recycling_num_iter'] if config_summit['recycling_on'] else 1
    with torch.no_grad():
        for recycle_iter in range(num_recycles):
            output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)
    pred_to_pdb("test.pdb", inputs, output)

