#!/bin/bash

step_size=${step_size:-0}
meta_lr=${meta_lr:-0}
grad_clip=${grad_clip:-0}
bias_alpha=${bias_alpha:-0}
first_layer_alpha=${first_layer_alpha:-0}
first_layer_lr_mult=${first_layer_lr_mult:-0}
first_layer_init_alpha=${first_layer_init_alpha:-0}
second_layer_init_alpha=${second_layer_init_alpha:-0}
batch_size=${batch_size:-0}


savedir=${savedir:-./ckpts}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

for depth in 1 2 3 4
do
    MUP_ARGS="--meta-lr $meta_lr \
              --meta-momentum 0 \
              --grad-clip $grad_clip \
              --gclip-per-param \
              --hidden-size 2048 \
              --bias-alpha $bias_alpha \
              --first-layer-alpha $first_layer_alpha \
              --first-layer-init-alpha $first_layer_init_alpha \
              --second-layer-init-alpha $second_layer_init_alpha \
              --last-layer-lr-mult 0 "
    IO_ARGS="--output-folder  $savedir/$depth"
    OPTIM_ARGS="--step-size $step_size \
                --batch-size $batch_size \
                --num-epochs 50 \
                --first-layer-lr-mult $first_layer_lr_mult \
                --scheduler cosine "
    TRAIN_ARGS="--num-ways 5 \
                --num-shots 1 \
                --use-cuda \
                --num-workers 2 \
                --num-shots-test 1  \
                --normalize None  \
                --dataset omniglot \
                --mu-init \
                --depth $depth \
                --verbose \
                --dtype float16 \
                --num-batches 1000 \
                --num-test-batches 500 \
                --adapt-readout-only 
                --readout-zero-init"
    # python -m cifar10.cifar10infmlp 
    python -m meta.train dataset $OPTIM_ARGS $TRAIN_ARGS $MUP_ARGS $IO_ARGS
done

echo "job done!"
