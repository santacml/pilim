#!/bin/bash

step_size=${step_size:-0}
meta_lr=${meta_lr:-0}
grad_clip=${grad_clip:-0}
bias_alpha=${bias_alpha:-0}
first_layer_alpha=${first_layer_alpha:-0}
first_layer_lr_mult=${first_layer_lr_mult:-0}


savedir=${savedir:-./ckpts}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done


for depth in 1 2 3 4
# for depth in 3 4
# for depth in 1 2 
do
    # for r in 50 100 200 400
    for r in  400
    do
        # echo "Performing test for r " r
        MUP_ARGS="--infnet_r $r "
        IO_ARGS="--output-folder  $savedir/$depth/$r"
        OPTIM_ARGS="--num-epochs 50 \
                    --meta-lr $meta_lr \
                    --step-size $step_size \
                    --batch-size 8 \
                    --grad-clip $grad_clip \
                    --meta-momentum 0 \
                    --bias-alpha $bias_alpha \
                    --first-layer-alpha $first_layer_alpha \
                    --first-layer-lr-mult $first_layer_lr_mult \
                    --last-layer-lr-mult 0 \
                    --scheduler cosine \
                    --readout-zero-init"
        TRAIN_ARGS="--dataset omniglot \
                    --num-ways 5 \
                    --num-shots 1 \
                    --use-cuda \
                    --num-workers 2 \
                    --num-shots-test 1  \
                    --normalize None \
                    --hidden-size -1 \
                    --depth $depth \
                    --dtype float16 \
                    --num-batches 1000  \
                    --num-test-batches 500 \
                    --adapt-readout-only \
                    --Gproj-inner  \
                    --Gproj-outer "
        # python -m cifar10.cifar10infmlp 
        python -m meta.train dataset $OPTIM_ARGS $TRAIN_ARGS $MUP_ARGS $IO_ARGS
    done
done


echo "job done!"


                     