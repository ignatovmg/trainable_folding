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
import torch.optim as optim
import torch.nn.functional as F
import logging
import sys
from copy import deepcopy
from path import Path
import math
from tqdm import tqdm
import traceback
import socket
import click

from alphadock import docker
from alphadock import config
from alphadock import dataset
from alphadock import all_atom
from alphadock import utils

import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


HOROVOD = False
HOROVOD_RANK = 0
hvd = None


def pred_to_pdb(out_pdb, input_dict, out_dict):
    out_pdb = Path(out_pdb)
    with open(out_pdb, 'w') as f:
        f.write(f'HEADER {out_pdb.basename().stripext()}.pred\n')
        all_atom.atom14_to_pdb_stream(
            f,
            input_dict['target']['rec_aatype'][0].cpu(),
            out_dict['final_all_atom']['atom_pos_tensor'].detach().cpu(),
            bfactors=out_dict['struct_out']['rec_lddt'][0, -1].detach().cpu().argmax(dim=-1) + 50,
            chain='A',
            serial_start=1,
            resnum_start=1
        )
        if 'ground_truth' in input_dict:
            f.write(f'HEADER {out_pdb.basename().stripext()}.crys\n')
            all_atom.atom14_to_pdb_stream(
                f,
                input_dict['ground_truth']['gt_aatype'][0].cpu(),
                input_dict['ground_truth']['gt_atom14_coords'][0].detach().cpu(),
                atom14_mask=input_dict['ground_truth']['gt_atom14_has_coords'][0].detach().cpu(),
                chain='A',
                serial_start=1,
                resnum_start=1
            )


def add_loss_to_stats(stats, output):
    stats['Loss_Total'] = output['loss']['loss_total'].item()
    if 'lddt_values' in output['loss']:
        stats['LDDT_Rec_Final'] = output['loss']['lddt_values']['rec_rec_lddt_true_total'][-1].item()
        stats['LDDT_Rec_MeanTraj'] = output['loss']['lddt_values']['rec_rec_lddt_true_total'].mean().item()
        stats['Loss_LDDT_Rec'] = output['loss']['lddt_loss_rec_rec'].item()
        stats['Loss_Torsions'] = output['loss']['loss_torsions']['chi_loss'].mean().item()
        stats['Loss_Norm'] = output['loss']['loss_torsions']['norm_loss'].mean().item()
        stats['Loss_FAPE_BB_Rec_Rec_Final'] = output['loss']['loss_fape']['loss_bb_rec_rec'][-1].item()
        stats['Loss_FAPE_AA_Rec_Rec_Final'] = output['loss']['loss_fape']['loss_aa_rec_rec'].item()
        stats['Loss_FAPE_BB_Rec_Rec_MeanTraj'] = output['loss']['loss_fape']['loss_bb_rec_rec'].mean().item()
        stats['Loss_PredDmat_RecRec'] = output['loss']['loss_pred_dmat'].item()
        stats['Loss_MSA_BERT'] = output['loss']['loss_msa_bert'].item()

    if 'violations' in output['loss']:
        viol = output['loss']['violations']
        stats['Violations/Loss'] = viol['loss'].item()
        stats['Violations/Extreme_CA_CA'] = viol['between_residues']['violations_extreme_ca_ca'].item()

        stats['Violations/Inter_ResRes_Bonds'] = viol['between_residues']['connections_per_residue_violation_mask'].mean().item()
        stats['Violations/Inter_ResRes_Clash'] = viol['between_residues']['clashes_per_atom_clash_mask'].max(-1).values.mean().item()
        stats['Violations/Intra_Residue_Violations'] = viol['within_residues']['per_atom_violations'].max(-1).values.mean().item()
        stats['Violations/Total_Residue_Violations'] = viol['total_per_residue_violations_mask'].mean().item()

        num_rec_atoms = torch.sum(input['target']['rec_atom14_atom_exists'][0]).item()
        stats['Violations/between_bonds_c_n_mean_loss'] = viol['between_residues']['bonds_c_n_loss_mean'].item()
        stats['Violations/between_angles_ca_c_n_mean_loss'] = viol['between_residues']['angles_ca_c_n_loss_mean'].item()
        stats['Violations/between_angles_c_n_ca_mean_loss'] = viol['between_residues']['angles_c_n_ca_loss_mean'].item()
        stats['Violations/between_clashes_mean_loss'] = viol['between_residues']['clashes_per_atom_loss_sum'].sum().item() / (1e-6 + num_rec_atoms)
        stats['Violations/within_mean_loss'] = viol['within_residues']['per_atom_loss_sum'].sum().item() / (1e-6 + num_rec_atoms)
    return stats


def report_step(input, output, global_stats, out_dir):
    stats = {}
    if 'loss' in output:
        add_loss_to_stats(stats, output)

    sample_idx = input["target"]["ix"].item()
    pred_to_pdb(Path(out_dir).mkdir_p() / f'prediction_{sample_idx:06d}.pdb', input, output)

    if 'loss' in output:
        utils.write_json(stats, Path(out_dir).mkdir_p() / f'{sample_idx:06d}.json')

    if HOROVOD:
        all_stats = hvd.allgather_object(stats)
    else:
        all_stats = [stats]

    if HOROVOD_RANK == 0:
        for idx, case_stats in enumerate(all_stats):
            for key, val in case_stats.items():
                if key not in global_stats:
                    global_stats[key] = []
                global_stats[key].append(val)

    return all_stats


def parse_fasta(fasta):
    parsed = []
    new_item = None
    with open(fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if new_item is not None:
                    parsed.append(new_item)
                new_item = [line.strip(), '']
            else:
                new_item[1] += line.strip()
    return parsed


def parse_first_sequence_a3m(a3m_file):
    with open(a3m_file, 'r') as f:
        line = f.readline()
        assert line[0] == '>', line
        seq = f.readline().strip()
        line = f.readline()
        assert line[0] == '>', line
    return seq


def print_input_shapes(inputs):
    for k1, v1 in inputs.items():
        print(HOROVOD_RANK, ':', k1)
        for k2, v2 in v1.items():
            print(HOROVOD_RANK, ':', '    ', k2, v1[k2].shape, v1[k2].dtype)
    sys.stdout.flush()


def main(
        model_pth,
        seed=123456,
        config_update_json=None,
        batch_json=None,
        data_dir='.',
        a3m_file=None,  # list of files
        extra_msa_size=4096,
        cif_file=None,
        cif_asym_id=None,
        out_dir='.',
        horovod=False,
        gpu=True,
):
    global HOROVOD, HOROVOD_RANK, hvd

    if horovod:
        HOROVOD = True
        import horovod.torch as hvd
        hvd.init()
        HOROVOD_RANK = hvd.rank()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    logging.getLogger('.prody').setLevel('CRITICAL')

    if gpu:
        assert torch.cuda.is_available(), 'CUDA is not available'
        device = 'cuda:0'
        if horovod:
            device = f'cuda:{hvd.local_rank()}'
    else:
        device = 'cpu'

    # remove checkpointing to get rid of the grad is none warning
    config_dict = deepcopy(config.config)
    config_dict = utils.merge_dicts(config_dict, {
        'data': {
            'crop_size': None,
            'msa_max_extra': extra_msa_size,
            'use_cache': False,
            'msa_block_del_num': 0,
            'msa_keep_true_msa': False
        },
        'loss': {
            'compute_loss': False
        },
        'model': {
            'msa_bert_block': False,
            'Evoformer': {'device': device, 'EvoformerIteration': {'checkpoint': False}},
            'InputEmbedder': {'device': device, 'ExtraMsaStack': {'device': device, 'ExtraMsaStackIteration': {'checkpoint': False}}},
            'StructureModule': {'device': device}
        }
    })

    if config_update_json:
        config_dict = utils.merge_dicts(config_dict, utils.read_json(config_update_json))

    model = docker.DockerIteration(config_dict['model'], config_dict)

    if HOROVOD_RANK == 0:
        print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Num param sets:', len([p for p in model.parameters() if p.requires_grad]))
        #for x in range(torch.cuda.device_count()):
        #    print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
        sys.stdout.flush()

    if HOROVOD_RANK == 0:
        print('Loading saved model from', model_pth)
        pth = torch.load(model_pth, map_location=device)
        if 'MSA_BERT.weight' in pth['model_state_dict']:
            del pth['model_state_dict']['MSA_BERT.weight']
            del pth['model_state_dict']['MSA_BERT.bias']
        model.load_state_dict(pth['model_state_dict'])

    if HOROVOD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    if batch_json is None:
        assert a3m_file
        target_seq = parse_first_sequence_a3m(a3m_file[0])
        batch_data = [{
            'entity_info': {
                'pdbx_seq_one_letter_code_can': target_seq,
                'asym_ids': [cif_asym_id]
            },
            'cif_file': cif_file,
            'a3m_files': a3m_file
        }]
        data_dir = '.'
    else:
        batch_data = utils.read_json(batch_json)

    dset = dataset.DockingDataset(
        batch_data,
        config_dict['data'],
        dataset_dir=data_dir,
        seed=seed,
        shuffle=False
    )

    kwargs = {'num_workers': 0, 'pin_memory': True, 'batch_size': 1, 'shuffle': False}
    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=HOROVOD_RANK, shuffle=False)
        loader = torch.utils.data.DataLoader(dset, sampler=sampler, **kwargs)
        sampler.set_epoch(0)
    else:
        loader = torch.utils.data.DataLoader(dset, **kwargs)

    global_stats = {}
    num_recycles = config_dict['model']['recycling_num_iter'] if config_dict['model']['recycling_on'] else 1

    model.modules_to_devices()
    model.eval()
    with torch.no_grad():
        for inputs in (tqdm(loader, desc='Processed') if HOROVOD_RANK == 0 else loader):
            print_input_shapes(inputs)

            for recycle_iter in range(num_recycles):
                output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)

            report_step(inputs, output, global_stats, out_dir)
            sys.stdout.flush()
            torch.cuda.empty_cache()

    if HOROVOD_RANK == 0:
        for key in global_stats.keys():
            vals = [x for x in global_stats[key] if not math.isnan(x)]
            global_stats[key] = sum(vals) / len(vals) if len(vals) > 0 else math.nan

        if len(dset) > 1 and len(global_stats) > 0:
            utils.write_json(global_stats, Path(out_dir) / 'average.json')


@click.command()
@click.argument('model_pth')
@click.option('--seed', default=123456, show_default=True, type=click.INT,
              help='Seed for RNG. Ensures reproducibility')
@click.option('--config_update_json',
              type=click.Path(exists=True, dir_okay=False),
              help='JSON containing configuration update. Will be merged with default alphafold.config.CONFIG')
@click.option('--batch_json',
              type=click.Path(exists=True, dir_okay=False),
              help='JSON containing a list of proteins to fold')
@click.option('--data_dir', default='./', show_default=True,
              type=click.Path(exists=True, file_okay=False, writable=True),
              help='Directory containing files specified in batch_json, paths in batch_json will be prepended')
@click.option('--a3m_file', multiple=True, help='Protein MSA file. Multiple MSAs will be concatenated')
@click.option('--extra_msa_size', default=4096, show_default=True, type=click.INT, help='Extra MSA size')
#@click.option('--cif_file',
#              type=click.Path(exists=True, dir_okay=False),
#              help='Protein reference structure CIF. Loss scores will be computed if provided')
#@click.option('--cif_asym_id', help='Reference chain asym_id in the CIF file')
@click.option('--out_dir', default='./', show_default=True,
              type=click.Path(exists=True, file_okay=False, writable=True),
              help='Directory where to put predicted models')
@click.option('--horovod', is_flag=True, help='Use Horovod for multi-GPU batch calculation')
@click.option('--gpu/--no_gpu', default=True, show_default=True,
              help='Use GPU or CPU. If GPU the device will be cuda:0 or cuda:<<local_rank>> when using Horovod')
def cli(**kwargs):
    """Predict structures for a single protein or a batch using MSAs in a3m format.

    MODEL_PTH - pth file with model parameters

    Inference can be run in two modes: single protein or batch. To predict a single
    protein provide one or several MSAs for it using --a3m_file.

    Example:

    \b
    > python inference.py model.pth --horovod --a3m_file msa1.a3m --a3m_file msa2.a3m --a3m_file msa3.a3m

    To run multiple proteins instead of --a3m_file provide --batch_json containing a
    list of proteins in the following format:

    \b
    [...,
        {
            'entity_info': {
                'pdbx_seq_one_letter_code_can': ACDADFF..TYHHEE,
                'asym_ids': [A] / None
            },
            'cif_file': xxxx.cif / None,
            'a3m_files': [msa1.a3m, msa2.a3m, ...]
        },
    ...]

    You can use Horovod to run a protein batch using multiple GPUs across multiple machines:

    \b
    > horovodrun -np 4 python inference.py model.pth --horovod --batch_json proteins.json

    """

    if not kwargs['a3m_file'] and not kwargs['batch_json']:
        raise ValueError('Either --a3m_file or --batch_json must be provided')

    #if kwargs['cif_file'] and not kwargs['cif_asym_id']:
    #    raise ValueError('--cif_asym_id must be provided with --cif_file')

    main(**kwargs)


cli()
