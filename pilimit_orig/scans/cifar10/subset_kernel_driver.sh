#!/bin/bash

# load_path=${load_path:-0}
# width=${width:--1}
seed=${seed:-0}

# IO
savedir=${savedir:-./ckpts}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

# for dataset in 100 500 1000 2500 5000
# for dataset in 500 1000
for dataset in 200 300 400
do
    # for width in 50 100 500 1000 5000 10000 15000 20000 30000 40000 0
    for width in 64 128 256 512 1024 2048 4096 8192 16384 32768 0
    do
        ARGS="--float \
                --cuda \
                --seed $seed \
                --kernel-reg 1e-06 \
                --width $width \
                --train-subset-size $dataset \
                --test-subset-size $dataset \
                --test-kernel \
                --save-kernel \
                --save-dir $savedir/$seed/$dataset/$width \
                --batch-size 32 \
                --test-batch-size 32"
        echo $dataset $width 
        python -m cifar10.cifar10test $ARGS
    done
done

echo "job done!"
