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
import sys
import math

from alphadock import docker
from alphadock import config
from alphadock import utils
from alphadock import all_atom
from copy import deepcopy
import torchvision
import pickle


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

    model = docker.DockerIteration(config.config['model'], config.config)
    model.load_state_dict(torch.load(sys.argv[1])['model_state_dict'])
    model.modules_to_devices()
    model.eval()
    num_recycles = config.config['model']['recycling_num_iter'] if config.config['model']['recycling_on'] else 1
    with torch.no_grad():
        for recycle_iter in range(num_recycles):
            output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)
    pred_to_pdb("prediction.pdb", inputs, output)

