#!/bin/bash

# Optim
lr=${lr:-0}
gclip=${gclip:-0}
wd=${wd:-0}
milestones=${milestones:-0}
first_layer_lr_mult=${first_layer_lr_mult:-0}
last_layer_lr_mult=${last_layer_lr_mult:-0}
first_layer_alpha=${first_layer_alpha:-0}
last_layer_alpha=${last_layer_alpha:-0}
batch_size=${batch_size:-0}
# width=${width:--1}

# MUP
bias_alpha=${bias_alpha:-0}

# IO
savedir=${savedir:-./ckpts}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

# for width in 128 256 512 1024 2048
for width in  2048
do
    # for r in 50 100 200 400
    for r in 400
    do
        MUP_ARGS="--bias-alpha $bias_alpha \
                --first-layer-lr-mult $first_layer_lr_mult \
                --last-layer-lr-mult $last_layer_lr_mult \
                --first-layer-alpha $first_layer_alpha \
                --last-layer-alpha $last_layer_alpha \
                --no-apply-lr-mult-to-wd"
        IO_ARGS="--save-dir $savedir/$width/$r"
        OPTIM_ARGS="--lr $lr \
                    --gclip-per-param \
                    --gclip $gclip \
                    --lr-drop-ratio 0.15 \
                    --lr-drop-milestones $milestones \
                    --scheduler multistep \
                    --wd $wd"
        TRAIN_ARGS="--r $r \
                    --width $width \
                    --batch-size $batch_size \
                    --epochs 50 \
                    --cuda"
        python -m cifar10.cifar10infmlp $OPTIM_ARGS $TRAIN_ARGS $MUP_ARGS $IO_ARGS
    done
done

echo "job done!"
