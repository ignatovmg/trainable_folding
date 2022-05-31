#!/bin/bash

files="${@}"
if [ $# -eq 0 ]; then
        files="{*sh,*py}"
fi

ROOT="$(cd "$(dirname ${BASH_SOURCE})" && pwd)/.."
echo scp ripper:"/home/ignatovmg/projects/trainable_folding/$(realpath --relative-to=$ROOT $(pwd))/$files" .
scp ripper:"/home/ignatovmg/projects/trainable_folding/$(realpath --relative-to=$ROOT $(pwd))/$files" .