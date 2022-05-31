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
import time

from alphadock import docker
from alphadock import config
from alphadock import dataset
from alphadock import all_atom
from alphadock import utils

#import warnings
#warnings.filterwarnings("error")

import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


OUT_DIR = Path('.')
LOG_PDB_EVERY_NSTEPS = 500
GLOBAL_STEP = 0
DATALOADER_KWARGS = {'num_workers': 0, 'pin_memory': True}
CONFIG_DICT = deepcopy(config.config)
TB_WRITE_STEP = False
SAVE_MODEL_EVERY_NEPOCHS = 1

MAX_NAN_ITER_FRAC = 0.05
CLIP_GRADIENT = True
CLIP_GRADIENT_VALUE = 0.1
USE_AMP = False
USE_AMP_SCALER = False

model = None
scheduler = None
optimizer = None
amp_scaler = None
tb_writer = None

HOROVOD_RANK = 0
HOROVOD = False


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


def report_step(input, output, epoch, dataset, global_stats, train=True):
    stage = 'Train' if train else 'Valid'
    stats = {'Generated_NaN': output.get('Generated_NaN', 0)}

    if 'loss' in output:
        add_loss_to_stats(stats, output)

        if (not train) or (LOG_PDB_EVERY_NSTEPS is not None and ((GLOBAL_STEP + HOROVOD_RANK) % LOG_PDB_EVERY_NSTEPS == 0)):
            ix = input['target']['ix'][0].item()
            case_name = dataset.data[ix]['pdb_id'] + '_' + dataset.data[ix]['entity_id']
            if train:
                file_name = f'train_epoch_{epoch}_step_{GLOBAL_STEP + HOROVOD_RANK:07d}_{case_name}_{stats["Loss_Total"]:.3f}.pdb'
            else:
                file_name = f'valid_epoch_{epoch}_step_{GLOBAL_STEP + HOROVOD_RANK:07d}_{case_name}_{stats["Loss_Total"]:.3f}.pdb'
            pred_to_pdb((OUT_DIR / 'models').mkdir_p() / file_name, input, output)
            stats_dump = stats.copy()
            #stats_dump['Used_HH_templates'] = 'hhpred' in input
            #stats_dump['Used_frag_templates'] = 'fragments' in input
            utils.write_json(stats_dump, (OUT_DIR / 'models' / file_name).stripext() + '.json')

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

                if train and TB_WRITE_STEP:
                    tb_writer.add_scalar(key + '/Step/' + stage, case_stats[key], GLOBAL_STEP + idx)

    return all_stats


def report_epoch_end(epoch, global_stats, stage='Train', save_model=True):
    global GLOBAL_STEP

    if HOROVOD_RANK == 0:
        tb_writer.add_scalar('HasNans/Epoch/' + stage, math.isnan(sum(global_stats['Loss_Total'])), epoch)
        for key in global_stats.keys():
            vals = [x for x in global_stats[key] if not math.isnan(x)]
            global_stats[key] = sum(vals) / len(vals) if len(vals) > 0 else math.nan
            tb_writer.add_scalar(key + '/Epoch/' + stage, global_stats[key], epoch)
        tb_writer.add_scalar('LearningRate/Epoch/' + stage, optimizer.param_groups[0]['lr'], epoch)
        print('Epoch_stats', global_stats)

    if HOROVOD:
        global_stats = hvd.broadcast_object(global_stats, root_rank=0)

    scheduler.step(global_stats['Loss_Total'])

    if HOROVOD_RANK == 0 and save_model and (epoch % SAVE_MODEL_EVERY_NEPOCHS == 0):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': global_stats,
            'global_step': GLOBAL_STEP,
            'hvd_size': 1 if not HOROVOD else hvd.size()
        }, OUT_DIR / f'epoch_{epoch}_loss_{global_stats["Loss_Total"]:.3f}.pth')


def check_grads(inputs):
    grads_are_nan = [name for name, x in model.named_parameters() if x.grad is not None and torch.isnan(x.grad).any().item()]
    if len(grads_are_nan) > 0:
        for x in sorted(grads_are_nan):
            print(x)
        raise utils.GeneratedNans(f'Process {HOROVOD_RANK}: gradients are nan')

    modules = list(model.StructureModule.named_parameters()) + list(model.Evoformer.named_parameters())
    if 'msa' in inputs and 'extra' in inputs['msa']:
        modules += list(model.InputEmbedder.ExtraMsaStack.named_parameters())
    grads_are_none = [name for name, x in modules if x.grad is None]
    if len(grads_are_none) > 0:
        for x in sorted(grads_are_none):
            print(x)
        assert len(grads_are_none) == 0, f'Process {HOROVOD_RANK}: gradients are None'


def print_input_shapes(inputs):
    for k1, v1 in inputs.items():
        print(HOROVOD_RANK, ':', k1)
        for k2, v2 in v1.items():
            print(HOROVOD_RANK, ':', '    ', k2, v1[k2].shape, v1[k2].dtype)
    sys.stdout.flush()


def validate(epoch, set_json, data_dir, seed):
    model.eval()

    config_eval = deepcopy(CONFIG_DICT)
    config_eval['data']['crop_size'] = None
    config_eval['data']['msa_block_del_num'] = 0
    dset = dataset.DockingDataset(
        utils.read_json(set_json),
        config_eval['data'],
        data_dir,
        seed=seed,
        shuffle=False
    )

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=HOROVOD_RANK, shuffle=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, shuffle=False, **DATALOADER_KWARGS)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, **DATALOADER_KWARGS)

    global_stats = {}
    local_step = 0
    num_recycles = CONFIG_DICT['model']['recycling_num_iter'] if CONFIG_DICT['model']['recycling_on'] else 1

    for inputs in (tqdm(loader, desc=f'Epoch {epoch} (valid)') if HOROVOD_RANK == 0 else loader):
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(USE_AMP):
                    for recycle_iter in range(num_recycles):
                        output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)
        except:
            print(HOROVOD_RANK, ':', 'Exception in validation', 'sample id:', inputs['target']['ix'])
            traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
            output = {}

        step_stats = report_step(inputs, output, epoch, dset, global_stats, train=False)
        local_step += 1
        sys.stdout.flush()
        torch.cuda.empty_cache()

    report_epoch_end(epoch, global_stats, stage='Valid', save_model=False)


def train(epoch, set_json, data_dir, seed):
    model.train()

    dset = dataset.DockingDataset(
        utils.read_json(set_json),
        CONFIG_DICT['data'],
        data_dir,
        seed=seed + epoch * 100,
        shuffle=True
    )

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=HOROVOD_RANK, shuffle=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, shuffle=False, **DATALOADER_KWARGS)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, **DATALOADER_KWARGS)

    global_stats = {}
    local_step = 0
    global GLOBAL_STEP
    global_step_start = GLOBAL_STEP

    # number of recycling iterations
    recycling_on = CONFIG_DICT['model']['recycling_on']
    num_recycles = CONFIG_DICT['model']['recycling_num_iter'] if recycling_on else 1
    recycling_rng = torch.Generator()
    recycling_rng = recycling_rng.manual_seed(seed + epoch * 100)

    t0 = time.time()
    for inputs in (tqdm(loader, desc=f'Epoch {epoch} (train)') if HOROVOD_RANK == 0 else loader):
        print(HOROVOD_RANK, ': time retrieving', inputs['target']['ix'].item(), ':', time.time() - t0, '(s)'); sys.stdout.flush()
        optimizer.zero_grad()
        generated_nan = 0

        if True and HOROVOD_RANK == 0:
            print_input_shapes(inputs)

        # sync recycling iteration for which the grad will be computed
        recycle_iter_grad_on = torch.randint(0, num_recycles, [1], generator=recycling_rng)[0].item() if HOROVOD_RANK == 0 and recycling_on else 0
        if HOROVOD:
            recycle_iter_grad_on = hvd.broadcast_object(recycle_iter_grad_on, root_rank=0)

        try:
            print(HOROVOD_RANK, ": sample id - ", inputs['target']['ix'])
            losses = []

            with torch.cuda.amp.autocast(USE_AMP):
                for recycle_iter in range(num_recycles):
                    with torch.set_grad_enabled((recycle_iter == recycle_iter_grad_on) or not recycling_on):
                        output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)
                    losses.append(output['loss']['loss_total'])

                    if HOROVOD_RANK == 0:
                        print(HOROVOD_RANK, ':', f'loss[{recycle_iter}]', losses[-1].item()); sys.stdout.flush()

            # calculate grads for selected recycling iteration
            if USE_AMP and USE_AMP_SCALER:
                amp_scaler.scale(losses[recycle_iter_grad_on]).backward()
            else:
                losses[recycle_iter_grad_on].backward()

            # check that grads are not nan, throw GeneratedNans if yes
            check_grads(inputs)

        except RuntimeError:
            # this is for CUDA out of memory error, if encountered we will just move to the next sample
            traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
            print_input_shapes(inputs)
            output = {}

        except utils.GeneratedNans:
            traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
            print_input_shapes(inputs)
            generated_nan = 1
            output = {}

        if HOROVOD:
            # following pytorch example from horovod docs
            optimizer.synchronize()

        if USE_AMP and USE_AMP_SCALER and CLIP_GRADIENT:
            amp_scaler.unscale_(optimizer)
        if CLIP_GRADIENT:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENT_VALUE)

        if HOROVOD:
            with optimizer.skip_synchronize():
                if USE_AMP and USE_AMP_SCALER:
                    amp_scaler.step(optimizer)
                else:
                    optimizer.step()
        else:
            if USE_AMP and USE_AMP_SCALER:
                amp_scaler.step(optimizer)
            else:
                optimizer.step()

        if USE_AMP and USE_AMP_SCALER:
            amp_scaler.update()

        output['Generated_NaN'] = generated_nan

        step_stats = report_step(inputs, output, epoch, dset, global_stats, train=True)
        GLOBAL_STEP += len(step_stats)
        local_step += 1
        sys.stdout.flush()

        if HOROVOD_RANK == 0:
            for k, v in step_stats[0].items():
                print(HOROVOD_RANK, ':', f'stats[{k}] = {v}')
            sys.stdout.flush()
            if (GLOBAL_STEP - global_step_start) / len(dset) > 0.05:
                nan_frac = sum(global_stats['Generated_NaN']) / len(global_stats['Generated_NaN'])
                assert nan_frac < MAX_NAN_ITER_FRAC, (nan_frac, MAX_NAN_ITER_FRAC)

        torch.cuda.empty_cache()
        t0 = time.time()

    report_epoch_end(epoch, global_stats, stage='Train', save_model=True)


def find_last_pth(dir):
    pths = Path(dir).glob('epoch_*.pth')
    if len(pths) == 0:
        return None
    return sorted(pths, key=lambda x: -int(x.basename().split('_')[1]))[0]


def main(
        train_json,
        valid_json=None,
        data_dir=None,
        horovod=False,
        gpu=True,
        seed=123456,
        model_pth=None,
        config_update_json=None,
        out_dir='.',
        max_epoch=None,
        tb_write_step=False,
        save_model_every_nepoch=1,
        log_pdb_every_nsteps=500,
        lr=0.001 / 128,
        lr_reset=False,
        lr_scale=True,
        scheduler_patience=50,
        scheduler_factor=1 / 3,
        scheduler_min_lr=1e-6 / 128,
        scheduler_reset=False,
        clip_gradient=True,
        clip_gradient_value=0.1,
        amp=False,
        amp_scale=False,
        gradient_compression=False
):
    global HOROVOD, HOROVOD_RANK, hvd, \
        OUT_DIR, TB_WRITE_STEP, LOG_PDB_EVERY_NSTEPS, \
        SAVE_MODEL_EVERY_NEPOCHS, GLOBAL_STEP, \
        CONFIG_DICT, CLIP_GRADIENT, CLIP_GRADIENT_VALUE, \
        USE_AMP, USE_AMP_SCALER, model, optimizer, \
        scheduler, amp_scaler, tb_writer

    if horovod:
        HOROVOD = True
        import horovod.torch as hvd
        hvd.init()
        HOROVOD_RANK = hvd.rank()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    logging.getLogger('.prody').setLevel('CRITICAL')

    OUT_DIR = Path(out_dir).mkdir_p()
    TB_WRITE_STEP = tb_write_step
    LOG_PDB_EVERY_NSTEPS = log_pdb_every_nsteps
    SAVE_MODEL_EVERY_NEPOCHS = save_model_every_nepoch
    CLIP_GRADIENT = clip_gradient
    CLIP_GRADIENT_VALUE = clip_gradient_value
    USE_AMP = amp
    USE_AMP_SCALER = amp_scale

    if gpu:
        assert torch.cuda.is_available(), 'CUDA is not available'
        device = 'cuda:0'
        if horovod:
            device = f'cuda:{hvd.local_rank()}'
    else:
        device = 'cpu'

    print(HOROVOD_RANK, ':', 'Using device', device)

    # remove checkpointing to get rid of the grad is none warning
    config_dict = deepcopy(config.config)
    config_dict = utils.merge_dicts(config_dict, {
        'model': {
            'msa_bert_block': True,
            'Evoformer': {'device': device, 'EvoformerIteration': {'checkpoint': True}},
            'InputEmbedder': {'device': device, 'ExtraMsaStack': {'device': device, 'ExtraMsaStackIteration': {'checkpoint': True}}},
            'StructureModule': {'device': device}
        }
    })

    if config_update_json:
        if HOROVOD_RANK == 0:
            print('Updating configuration using', config_update_json)
        config_dict = utils.merge_dicts(config_dict, utils.read_json(config_update_json))

    CONFIG_DICT = config_dict

    model = docker.DockerIteration(config_dict['model'], config_dict)
    model.modules_to_devices()

    if HOROVOD_RANK == 0:
        print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Num param sets:', len([p for p in model.parameters() if p.requires_grad]))

    lr_scaler = 1 if not HOROVOD or not lr_scale else hvd.size()
    if HOROVOD_RANK == 0:
        print('Scaling learning rate by', lr_scaler)
        print('Resulting learning rate', lr * lr_scaler)
    optimizer = optim.Adam(model.parameters(), lr=lr * lr_scaler) #, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr * lr_scaler)

    start_epoch = 1
    scheduler_state = scheduler.state_dict()

    if model_pth is None and HOROVOD_RANK == 0:
        print('model_pth is not set, looking for pth in the current directory')
        model_pth = find_last_pth(OUT_DIR)
        if model_pth is not None:
            print('Found', model_pth)
        else:
            print('Did not find any saved state, starting from scratch')

    if model_pth is not None and HOROVOD_RANK == 0:
        print('Loading saved model from', model_pth)
        dict_pth = torch.load(model_pth)
        model.load_state_dict(dict_pth['model_state_dict'])

        if 'global_step' in dict_pth:
            print('Setting global step to', dict_pth['global_step'])
            GLOBAL_STEP = dict_pth['global_step']

        if 'epoch' in dict_pth:
            print('Setting starting epoch to', dict_pth['epoch'] + 1)
            start_epoch = dict_pth['epoch'] + 1

        if 'optimizer_state_dict' in dict_pth:
            print('Loading optimizer_state_dict')
            optimizer.load_state_dict(dict_pth['optimizer_state_dict'])

        if 'scheduler_state_dict' in dict_pth and not scheduler_reset:
            print('Loading scheduler_state_dict')
            scheduler_state = dict_pth['scheduler_state_dict']

        if 'hvd_size' in dict_pth and lr_scale:
            _hvd_size = 1 if not HOROVOD else hvd.size()
            if _hvd_size != dict_pth['hvd_size']:
                print('Rescaling learning rate by ', _hvd_size / dict_pth['hvd_size'])
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * _hvd_size / dict_pth['hvd_size']

    if lr_reset:
        for g in optimizer.param_groups:
            g['lr'] = lr * lr_scaler

    if HOROVOD:
        GLOBAL_STEP = hvd.broadcast_object(GLOBAL_STEP, root_rank=0)
        start_epoch = hvd.broadcast_object(start_epoch, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        scheduler_state = hvd.broadcast_object(scheduler_state, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average,
                                             compression=hvd.Compression.fp16 if gradient_compression else hvd.Compression.none)

    scheduler.load_state_dict(scheduler_state)

    if USE_AMP and USE_AMP_SCALER:
        amp_scaler = torch.cuda.amp.GradScaler()

    if HOROVOD_RANK == 0:
        tb_writer = SummaryWriter(OUT_DIR)

    epoch = start_epoch
    while True:
        #with torch.autograd.set_detect_anomaly(True):
        if max_epoch is not None and epoch > max_epoch:
            if HOROVOD_RANK == 0:
                print(f'Reached max epoch {max_epoch}')
            break
        train(epoch, train_json, data_dir, seed)
        if valid_json:
            validate(epoch, valid_json, data_dir, seed)
        epoch += 1


@click.command()
@click.argument('train_json')
@click.option('--valid_json', type=click.Path(exists=True, dir_okay=False),
              help='Path to validation set')
@click.option('--data_dir', default='./', show_default=True,
              type=click.Path(exists=True, file_okay=False, writable=True),
              help='Directory containing files specified in train_json, paths in train_json will be prepended')
@click.option('--model_pth', type=click.Path(exists=True, dir_okay=False),
              help='Resume training from the saved state')
@click.option('--gpu/--no_gpu', default=True, show_default=True,
              help='Use GPU or CPU. If GPU the device will be cuda:0 or cuda:<<local_rank>> when using Horovod')
@click.option('--seed', default=123456, show_default=True, type=click.INT,
              help='Seed for RNG. Ensures reproducibility')
@click.option('--config_update_json', type=click.Path(exists=True, dir_okay=False),
              help='JSON containing configuration update. Will be merged with default alphafold.config.CONFIG')
@click.option('--out_dir', default='./', show_default=True,
              type=click.Path(exists=True, file_okay=False, writable=True),
              help='Output directory, must exist')
@click.option('--horovod', is_flag=True,
              help='Use Horovod for multi-GPU batch training')
@click.option('--tb_write_step', is_flag=True,
              help='Write every step to tensorboard writer (not recommended as TB files can get very large)')
@click.option('--max_epoch', default=None, type=click.INT,
              help='Stop training when MAX_EPOCH is reached')
@click.option('--save_model_every_nepoch', default=1, show_default=True, type=click.INT,
              help='Save model to pth file every Nth epoch')
@click.option('--log_pdb_every_nsteps', default=500, show_default=True, type=click.INT,
              help='Write predictions to OUT_DIR/pdb every N steps')
@click.option('--lr', default=0.001 / 128, show_default=True, type=click.FLOAT,
              help='Learning rate. Effective only when starting training from scratch unless --lr_reset is used')
@click.option('--lr_reset', is_flag=True,
              help='Do not use LR from previous epoch')
@click.option('--lr_scale/--no_lr_scale', default=True, show_default=True,
              help='Multiply learning rate by Horovod batch size')
@click.option('--scheduler_patience', default=50, show_default=True, type=click.INT,
              help='LR scheduler patience')
@click.option('--scheduler_factor', default=1 / 3, show_default=True, type=click.FLOAT,
              help='LR scheduler factor')
@click.option('--scheduler_min_lr', default=1e-6 / 128, show_default=True, type=click.FLOAT,
              help='LR scheduler minimum lr')
@click.option('--scheduler_reset', is_flag=True,
              help='Erase LR scheduler memory')
@click.option('--clip_gradient/--no_clip_gradient', default=True, show_default=True,
              help='Clip gradient')
@click.option('--clip_gradient_value', default=0.1, show_default=True, type=click.FLOAT,
              help='Clip gradient value')
@click.option('--amp/--no_amp', default=False, show_default=True,
              help='Use Automatic Mixed Precision')
@click.option('--amp_scale/--no_amp_scale', default=False, show_default=True,
              help='Use Gradient Scaler with AMP')
@click.option('--gradient_compression/--no_gradient_compression', default=False, show_default=True,
              help='Use Horovod gradient compression (compression=hvd.Compression.fp16)')
def cli(**kwargs):
    """Run model training

    TRAIN_JSON - JSON file with training dataset

    You can use Horovod to run a protein batch using multiple GPUs across multiple machines:

    \b
    > horovodrun -np 4 python train.py --horovod train.json

    """
    main(**kwargs)


cli()
