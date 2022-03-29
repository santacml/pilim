#!/bin/bash

varb=${varb:-0}
reg=${reg:-0}

savedir=${savedir:-./ckpts}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

for depth in 2
do
    MUP_ARGS=""
    IO_ARGS="--save-dir $savedir/$depth"
    OPTIM_ARGS="--varb $varb \
                --depth $depth \
                --kernel-reg $reg"
    TRAIN_ARGS="--gp \
                --float \
                --batch-size 2000 \
                --test-batch-size 2000"
    python -m cifar10.cifar10test $OPTIM_ARGS $TRAIN_ARGS $MUP_ARGS $IO_ARGS
done

echo "job done!"
