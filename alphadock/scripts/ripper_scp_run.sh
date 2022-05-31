#!/bin/bash

echo scp events* ripper:~/projects/trainable_folding/data_preparation/data/runs/$(basename $(pwd))
scp events* ripper:~/projects/trainable_folding/data_preparation/data/runs/$(basename $(pwd))