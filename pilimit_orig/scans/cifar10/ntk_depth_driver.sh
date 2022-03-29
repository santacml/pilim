#!/bin/bash

varb=${varb:-0}
bias_lr_mult=${bias_lr_mult:-0}
first_layer_lr_mult=${first_layer_lr_mult:-0}
last_layer_lr_mult=${last_layer_lr_mult:-0}
reg=${reg:-0}


savedir=${savedir:-./ckpts}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

for depth in 1 2 3
do
    MUP_ARGS=""
    IO_ARGS="--save-dir $savedir/$depth"
    OPTIM_ARGS="--varb $varb \
                --depth $depth \
                --first-layer-lr-mult $first_layer_lr_mult \
                --last-layer-lr-mult $last_layer_lr_mult \
                --bias-lr-mult $bias_lr_mult \
                --kernel-reg $reg"
    TRAIN_ARGS="--ntk \
                --float \
                --batch-size 2000 \
                --test-batch-size 2000"
    python -m cifar10.cifar10test $OPTIM_ARGS $TRAIN_ARGS $MUP_ARGS $IO_ARGS
done

echo "job done!"
