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

for depth in 1 2 3
do
    MUP_ARGS="--step-size 0.5 \
              --hidden-size -1 \
              --gp 
              "
    IO_ARGS="--output-folder  $savedir/$depth"
    OPTIM_ARGS="--batch-size 8 \
                --num-epochs 1 \
                --scheduler multistep \
                --varb $varb \
                --depth $depth \
                --kernel-reg $reg 
                "
    TRAIN_ARGS="--dataset omniglot \
                --num-ways 5 \
                --num-shots 1 \
                --use-cuda \
                --num-workers 8 \
                --num-shots-test 1 \
                --first-order \
                --verbose \
                --validate-only \
                --overwrite-existing \
                --test-dataset-split val \
                --num-test-batches 500
                "
    # python -m cifar10.cifar10infmlp 
    python -m meta.train dataset $OPTIM_ARGS $TRAIN_ARGS $MUP_ARGS $IO_ARGS
done

echo "job done!"
