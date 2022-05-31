import sys
sys.path.insert(1, '../')
from alphadock import all_atom, residue_constants
import pickle
import numpy as np
import torch
from multimer import modules_multimer, config_multimer, load_param_multimer

def pred_to_pdb(out_pdb, input_dict, out_dict):
    with open(out_pdb, 'w') as f:
        f.write(f'test.pred\n')
        serial = all_atom.atom14_to_pdb_stream(
            f,
            input_dict['aatype'][0].cpu(),
            out_dict['final_all_atom'].detach().cpu(),
            chain='A',
            serial_start=1,
            resnum_start=1
        )
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')

def protein_to_pdb(aatype, atom_positions, residue_index, chain_index, atom_mask):
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []
    residue_index = residue_index.astype(np.int32)
    chain_index = chain_index.astype(np.int32)
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    for i in range(aatype.shape[0]):
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
            atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
            residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask in zip(
            atom_types, atom_positions[i], atom_mask[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

  # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'

if __name__ == '__main__':
    with open('/home/thu/Downloads/alphafold/output/multimer_feat.pkl', 'rb') as f:
        processed_feature_dict = pickle.load(f)
    feats = {k: torch.unsqueeze(torch.tensor(v, device='cuda:0'),0) for k,v in processed_feature_dict.items()}
    model = modules_multimer.DockerIteration(config_multimer.config_multimer)
    load_param_multimer.import_jax_weights_(model)
    num_recycle = 3
    with torch.no_grad():
        for recycle_iter in range(num_recycle):
            output = model(feats, recycle=output['recycling_input'] if recycle_iter > 0 else None)

    pdb_out = protein_to_pdb(feats['aatype'][0].cpu().numpy(), output['final_all_atom'].detach().cpu().numpy(), feats['residue_index'][0].cpu().numpy() + 1, feats['asym_id'][0].cpu().numpy(), output['final_atom_mask'].cpu().numpy())
    with open('test.pdb', 'w') as f:
        f.write(pdb_out)

    # model(feats)
    #
    # with open('/home/thu/Downloads/trainable_folding/test/features.pkl', 'rb') as f:
    #     processed_feature_dict_mono = pickle.load(f)
    # for k, v in processed_feature_dict_mono.items():
    #     print(k)
    #     for i,j in v.items():
    #         print(i, j.shape)

