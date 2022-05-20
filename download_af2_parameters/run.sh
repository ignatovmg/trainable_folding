#!/bin/bash

wget https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar
tar xvf alphafold_params_2022-03-02.tar
for i in {1..5}; do
  python ../test/af2_params_to_pth.py params_model_${i}.npz
  python ../test/af2_params_to_pth.py params_model_${i}_ptm.npz
done
rm -f alphafold_params_2022-03-02.tar