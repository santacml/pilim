# π-Limit

Here we provide all of the original code for our paper. This repo could be quite difficult to reuse or modify given its highly specific structure tailored to  the π-Limit. 

For instance, in the π-Net, the A and B matrices are not accessible as torch parameters and the layers are not modules. This means that things such as torch-native saving, dataparallel, .named_parameters() iteration, and so on, will not work. 

We would not recommend using this repo except to reproduce the results from the paper using the commands below.

# Structure

[inf](inf) contains the implementation of the infinite and π-Net, the finite μ-Net, and NNGP/NTK kernels. It also contains π-Net specific mathematical functions, InfSGD, dynamically expanding arrays. and any other supporting π-Net infrastructure.

[cifar10](cifar10), [imagenet](imagenet), and [meta](meta) contain the training and testing files for each of their respective tasks. 

[scans](scans) contains all of the files which were used for scanning and testing hyperparameters. These testing files are in a Microsoft-specific format so they are not useful for non-Microsoft employees, but they are kept for completeness.

This folder also contains a .zip file with all of the original dataframes with results from the paper, and various files to create figures from these results which are in the paper.

[utils](utils) contains a few scattered useful utility functions and classes.



# Hyperparameter Commands

Here we provide all of the commands and hyperparameters to reproduce any of the results from the best-performing networks of each variety on each task (Table 1). Each link in this table goes to a command which, when run, should reproduce the results (though may be very slightly off due to seeding/machine specific issues).

|  | NNGP   | NTK | Finite μ-Net  | Finite π-Net | Infinite π-Net | 
| ------------- |---------| --| --| --| --|
| CIFAR10     | [58.92](#cifar10-nngp)  | [ 59.63](#cifar10-ntk) | [61.31](#cifar10-munet) | [60.64](#cifar10-finpinet) |  [61.50](#cifar10-infpinet)| 
| MAML      | [ 43.80](#maml-nngp)  | [ 51.72](#maml-ntk) | [91.22](#maml-munet) | [92.21](#maml-finpinet) |  [ 91.46](#maml-infpinet) | 




## CIFAR10 NNGP

python -m cifar10.cifar10test --varb 0 --depth 2   --kernel-reg 1e-4 --gp --float --batch-size 2000 --test-batch-size 2000  --save-dir "./"

## CIFAR10 NTK

python -m cifar10.cifar10test --varb 0 --depth 2 --first-layer-lr-mult 1 --last-layer-lr-mult 1 --bias-lr-mult 1 --kernel-reg $reg --ntk --float --batch-size 2000 --test-batch-size 2000  --save-dir "./"

## CIFAR10 MuNet

python -m cifar10.cifar10infmlp --lr 1.00 --gclip-per-param --gclip 0.10 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00016 --batch-size 32 --epochs 50 --width 2048 --cuda --seed 0  --depth 2 --bias-alpha 4.000 --first-layer-lr-mult 0.2 --last-layer-lr-mult 16.0 --first-layer-alpha 2.000 --last-layer-alpha 0.250 --no-apply-lr-mult-to-wd --save-dir "./" --gaussian-init --no-Gproj

## CIFAR10 FinPiNet

python -m cifar10.cifar10infmlp --lr 0.5 --gclip-per-param --gclip 3.20 --lr-drop-ratio 0.15 --lr-drop-milestones 35 --scheduler multistep --wd 0.00016 --r 400 --batch-size 32 --epochs 50 --width 2048 --cuda --seed 0  --depth 2 --bias-alpha 4.000 --first-layer-lr-mult 0.20 --last-layer-lr-mult 8.0 --first-layer-alpha 0.125 --last-layer-alpha 0.500 --no-apply-lr-mult-to-wd --save-dir "./"

## CIFAR10 InfPiNet

python -m cifar10.cifar10infmlp --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 400 --batch-size 8 --epochs 50 --width 0 --cuda --seed 0  --depth 2 --bias-alpha $bias_alpha --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir "./"



Todo: need to find maml hparams, they are not saveed


## MAML NNGP

python -m meta.train dataset --batch-size 8 --num-epochs 1 --scheduler multistep --varb 1 --depth 2 --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 8 --num-shots-test 1 --first-order --verbose --validate-only --overwrite-existing --test-dataset-split val --num-test-batches 500 --step-size 0.5 --hidden-size -1 --gp --output-folder  "./"

## MAML NTK

python -m meta.train dataset --batch-size 8 --num-epochs 1 --scheduler multistep --varb 1 --depth 2 --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 8 --num-shots-test 1 --first-order --verbose --validate-only --overwrite-existing --test-dataset-split val --num-test-batches 500 --step-size 0.5 --hidden-size -1 --ntk --output-folder  "./"

## MAML MuNet

python -m meta.train dataset --num-epochs 50 --meta-lr  4.0 --step-size 0.594604 --batch-size 14 --grad-clip 0.15 --meta-momentum 0 --bias-alpha  0.5 --first-layer-alpha  0.594604 --first-layer-lr-mult 0.4 --first-layer-init-alpha  0.840896 --second-layer-init-alpha 0.594604 --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 2 --num-shots-test 1  --normalize None --hidden-size -1 --depth 2 --dtype float16 --num-batches 1000  --num-test-batches 500 --adapt-readout-only --mu-init  --output-folder  "./"

## MAML FinPiNet

python -m meta.train dataset --num-epochs 50 --meta-lr $meta_lr --step-size 0.25 --batch-size 8 --grad-clip 0.282843 --meta-momentum 0 --bias-alpha  2.828427  --first-layer-alpha 1.0 --first-layer-lr-mult 0.4 --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 2 --num-shots-test 1  --normalize None --hidden-size -1 --depth 2 --dtype float16 --num-batches 1000  --num-test-batches 500 --adapt-readout-only --Gproj-inner  --Gproj-outer  --infnet_r 400  --output-folder  "./"

## MAML InfPiNet

python -m meta.train dataset --num-epochs 50 --meta-lr 32.0 --step-size 0.353553 --batch-size 8 --grad-clip  0.1   --meta-momentum 0 --bias-alpha 1.414214 --first-layer-alpha 1.0 --first-layer-lr-mult 0.400000 --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 2 --num-shots-test 1  --normalize None --hidden-size -1 --depth 2 --dtype float16 --num-batches 1000  --num-test-batches 500 --adapt-readout-only --Gproj-inner  --Gproj-outer  --infnet_r 400  --output-folder  "./"






