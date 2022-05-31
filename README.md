# Trainable AlphaFold 2 ported to Pytorch

Our version of AlphaFold 2 written using Pytorch library. 

The repository provides a training script `alphafold/train.py`, which contains a similar to 
AlphaFold 2 training procedure.

We also provide a script for protein folding `alphafold/inference.py`, which can use 
AlphaFold 2 original parameters as well as our trained parameters.

## Inference with AlphaFold 2 original parameters

Go to `download_af2_parameters/` and follow the steps. This will download the parameters and convert
them to `pth` format, which can be used with `alphafold/inference.py` to fold proteins from MSAs. 
Check `examples/inference_1ezi_with_af2_params/` for an example of how to use AF2 parameters with 
our code.

## Training example

Follow the steps in `examples/training_toy_example/` to run training on a small debug dataset (3 proteins).

## Deployment

The project was developed and deployed on the 
[Summit Supercomputer](https://www.olcf.ornl.gov/summit) 
inside a conda enviroment, which can be recreated on Summit using `bootstrap_summit.sh`. 
The base for the enviroment is https://github.com/open-ce/open-ce, version 1.4.1. 
Some of the main packages are listed below

- python == 3.8
- pytorch == 1.9.0
- prody == 2.0.1
- numpy == 1.19.2
- mpi4py == 3.1.1
- nccl == 2.8.3
- horovod == 0.22.1
- cudatoolkit == 11.0.221
- biopython == 1.79

For installation instructions refer to https://osuosl.org/services/powerdev/opence/