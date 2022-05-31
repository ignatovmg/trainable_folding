#!/bin/bash
#BSUB -P BIP215
#BSUB -W 2:00
#BSUB -nnodes 4
#?SUB -J adock[2-50]%1
#?SUB -w done(1565114)
#?SUB -q debug
#?SUB -J RunSim123
#?SUB -o RunSim123.%J
#?SUB -e RunSim123.%J

source /ccs/home/ignatovmg/projects/alphadock/activate_summit.sh
wdir=/ccs/home/ignatovmg/projects/alphadock/data_preparation/data/runs/run2
train=/ccs/home/ignatovmg/projects/alphadock/alphadock/train.py

mkdir -p $wdir && cd $wdir
cp $train .

jsrun -n $((6 * 4)) -r 6 -a 1 -g 1 -c 4 python $train &> log.$LSB_JOBID
#jsrun -n 6 -a 1 -g 1 -c 4 python /ccs/home/ignatovmg/projects/alphadock/alphadock/train.py