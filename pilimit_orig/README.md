# π-Limit

Here we provide all of the original code for our paper. This repo could be quite difficult to reuse or modify given its highly specific structure tailored to  the π-Limit. 

For instance, in the π-Net, the A and B matrices are not accessible as torch parameters and the layers are not modules. This means that things such as torch-native saving, dataparallel, .named_parameters() iteration, and so on, will not work. 

**We would not recommend using this repo except to reproduce the results from the paper using the commands below.**

# Structure

[inf](inf) contains the implementation of the infinite and π-Net, the finite μ-Net, and NNGP/NTK kernels. It also contains π-Net specific mathematical functions, InfSGD, dynamically expanding arrays. and any other supporting π-Net infrastructure.

The file pimlp.py defines the infinite and finite width networks. However, if this file is called directly (i.e. python -m inf.pimlp) it will create a very tiny infinite-width MLP and run it on a dummy data sample. This is useful for testing and getting a "feel" for the network.

[cifar10](cifar10), [imagenet](imagenet), and [meta](meta) contain the training and testing files for each of their respective tasks. Each file has hyperparams for the command line. See below commands for example usage.

For CIFAR10, training and testing accuracy will be shown during training.

For imagenet transfer learning, it is necessary to save the network and then use cifar10test.py to obtain the final feature kernel performance.

For omniglot, it is necessary to save the network and use test.py to obtain the final testing performance.

[scans](scans) contains all of the files which were used for scanning and testing hyperparameters. These testing files are in a Microsoft-specific format so they are not useful for non-Microsoft employees, but they are kept for completeness.

This folder also contains a .zip file with all of the original dataframes with results from the paper, and various files to create figures from these results which are in the paper.

[utils](utils) contains a few scattered useful utility functions and classes.


# MAML testing notes

For maml, each of the below training commands will only go through the training cycle. To obtain final testing performance, it's necessary to run the saved config file from the training cycle through this command:

python -m meta.test --config '/path/to/config.json' --use-cuda --seed 0"

And also note that performance varies per seed, so it is recommended to run over many seeds.


# Imagenet testing notes

For Imagenet transfer to CIFAR10 results, each of these commands will only go through the training cycle on imagenet and save the model. The script will save the model on every single epoch as it's not clear which epoch will perform best on CIFAR10. To obtain final testing performance, it's necessary to test the kernel of  the saved model for every epoch of imagenet training using the following command:

python -m cifar10.cifar10test --cuda --kernel-reg={reg} --test-kernel --load-model-path="/path/to/model.th" --save-dir ./output/ --batch-size 32 --test-batch-size 32

Also note that doing the transfer tests with the infinite π-Net will require at least 32gb of GPU VRAM and RAM.

# Hyperparameter Commands

Here we provide all of the commands and hyperparameters to reproduce any of the results from the best-performing networks of each variety on each task (Table 1). Each link in this table goes to a command which, when run, should reproduce the results (though may be very slightly off due to seeding/machine specific issues).

|  | NNGP   | NTK | Finite μ-Net  | Finite π-Net | Infinite π-Net | 
| ------------- |---------| --| --| --| --|
| CIFAR10     | [58.92](#cifar10-nngp)  | [ 59.63](#cifar10-ntk) | [61.31](#cifar10-munet) | [60.64](#cifar10-finpinet) |  [61.50](#cifar10-infpinet)| 
| MAML      | [ 43.80](#maml-nngp)  | [ 51.72](#maml-ntk) | [91.22](#maml-munet) | [92.21](#maml-finpinet) |  [ 91.46](#maml-infpinet) | 


This table contains all of the imagenet transfer numbers with links to their respective hyperparameters and commands.

|  | Finite μ-Net  | Finite π-Net, r=200 | Finite π-Net, r=400 | Infinite π-Net, r=200 | 
| ------------- |---------| --| --| --| 
| Transfer      |[61.84](#imagenet-munet) | [58.02](#imagenet-finpinet-r-200) | [59.36](#imagenet-finpinet-r-400) |  [64.39](#imagenet-infpinet-r-200) |
 
# CIFAR10

## CIFAR10 NNGP

```
python -m cifar10.cifar10test --varb 0 --depth 2   --kernel-reg 1e-4 --gp --float --batch-size 2000 --test-batch-size 2000  --save-dir ./output/
```

## CIFAR10 NTK

```
python -m cifar10.cifar10test --varb 0 --depth 2 --first-layer-lr-mult 1 --last-layer-lr-mult 1 --bias-lr-mult 1 --kernel-reg 1e-4 --ntk --float --batch-size 2000 --test-batch-size 2000  --save-dir ./output/
```

## CIFAR10 MuNet

```
python -m cifar10.cifar10infmlp --lr 1.00 --gclip-per-param --gclip 0.10 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00016 --batch-size 32 --epochs 50 --width 2048 --cuda --seed 0  --depth 2 --bias-alpha 4.000 --first-layer-lr-mult 0.2 --last-layer-lr-mult 16.0 --first-layer-alpha 2.000 --last-layer-alpha 0.250 --no-apply-lr-mult-to-wd --save-dir ./output/ --gaussian-init --no-Gproj
```

## CIFAR10 FinPiNet

```
python -m cifar10.cifar10infmlp --lr 0.5 --gclip-per-param --gclip 3.20 --lr-drop-ratio 0.15 --lr-drop-milestones 35 --scheduler multistep --wd 0.00016 --r 400 --batch-size 32 --epochs 50 --width 2048 --cuda --seed 0  --depth 2 --bias-alpha 4.000 --first-layer-lr-mult 0.20 --last-layer-lr-mult 8.0 --first-layer-alpha 0.125 --last-layer-alpha 0.500 --no-apply-lr-mult-to-wd --save-dir ./output/
```

## CIFAR10 InfPiNet

```
python -m cifar10.cifar10infmlp --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 400 --batch-size 8 --epochs 50 --width 0 --cuda --seed 0  --depth 2 --bias-alpha 0.5 --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir ./output/
```


# MAML


## MAML NNGP

```
python -m meta.train dataset --batch-size 8 --num-epochs 1 --scheduler multistep --varb 1 --depth 2 --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 8 --num-shots-test 1 --first-order --verbose --validate-only --overwrite-existing --test-dataset-split val --num-test-batches 500 --step-size 0.5 --hidden-size -1 --gp --output-folder  ./output/
```

## MAML NTK

```
python -m meta.train dataset --batch-size 8 --num-epochs 1 --scheduler multistep --varb 1 --depth 2 --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 8 --num-shots-test 1 --first-order --verbose --validate-only --overwrite-existing --test-dataset-split val --num-test-batches 500 --step-size 0.5 --hidden-size -1 --ntk --output-folder  ./output/
```

## MAML MuNet

```
python -m meta.train dataset --num-epochs 50 --meta-lr  4.0 --step-size 0.594604 --batch-size 8 --grad-clip 0.15 --meta-momentum 0 --bias-alpha  0.5 --first-layer-alpha  0.594604 --first-layer-lr-mult 0.4 --first-layer-init-alpha  0.840896 --second-layer-init-alpha 0.594604 --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 2 --num-shots-test 1  --normalize None --hidden-size -1 --depth 2 --dtype float16 --num-batches 1000  --num-test-batches 500 --adapt-readout-only --mu-init  --output-folder  ./output/
```

## MAML FinPiNet

```
python -m meta.train dataset --num-epochs 50 --meta-lr 5.656854 --step-size 0.25 --batch-size 8 --grad-clip 0.282843 --meta-momentum 0 --bias-alpha  2.828427  --first-layer-alpha 1.0 --first-layer-lr-mult 0.4 --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 2 --num-shots-test 1  --normalize None --hidden-size -1 --depth 2 --dtype float16 --num-batches 1000  --num-test-batches 500 --adapt-readout-only --Gproj-inner  --Gproj-outer  --infnet_r 400  --output-folder  ./output/
```

## MAML InfPiNet

```
python -m meta.train dataset --num-epochs 50 --meta-lr 32.0 --step-size 0.353553 --batch-size 8 --grad-clip  0.1   --meta-momentum 0 --bias-alpha 1.414214 --first-layer-alpha 1.0 --first-layer-lr-mult 0.400000 --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --num-workers 2 --num-shots-test 1  --normalize None --hidden-size -1 --depth 2 --dtype float16 --num-batches 1000  --num-test-batches 500 --adapt-readout-only --Gproj-inner  --Gproj-outer  --infnet_r 400  --output-folder  ./output/
```

# Imagenet 


## Imagenet MuNet

```
python -m imagenet.transfer_imagenet --save-dir=./output/ --gaussian-init --save-model --cuda --width=2048 --lr=0.005 --batch-size=16 --gclip=0.0 --epochs=40 --human --wd=0.008 --bias-alpha=4.0  --first-layer-lr-mult=0.553341 --last-layer-lr-mult=5.656854 --gclip-per-param 
```

Test with: epoch 32 reg 1e-4

## Imagenet FinPiNet r 200

```
python -m imagenet.transfer_imagenet  --save-dir=./output/ --save-model --cuda --width=2048 --r 200 --lr=0.028284 --batch-size=16 --gclip=0.4  --epochs=40 --human --wd=0.000177 --bias-alpha=0.353553  --first-layer-lr-mult= 1.524828 --last-layer-lr-mult=1.0 --gclip-per-param 
```

Test with: epoch 27, reg 1e-4

## Imagenet FinPiNet r 400

```
python -m imagenet.transfer_imagenet  --save-dir=./output/ --save-model --cuda --width=2048 --r 400 --lr=0.05 --batch-size=16 --gclip=0.8  --epochs=40 --human --wd=0.0005 --bias-alpha=0.707107 --first-layer-lr-mult=0.612372 --last-layer-lr-mult=1.0 --gclip-per-param 
```

Test with: epoch 31 reg 1e-3

## Imagenet InfPiNet r 200

```
python -m imagenet.transfer_imagenet  --save-dir=./output/ --save-model --cuda --r 200 --lr=0.01 --batch-size=16 --gclip=0  --epochs=40 --human --wd=0.0001 --bias-alpha=0.5 --first-layer-lr-mult=1.0 --last-layer-lr-mult=1.0 --gclip-per-param 
```

Test with: epoch 26 reg 1e-4

Note: this will require a very large amount of GPU memory, 32GB, and a very large amount of disk memory (probably around 100GB because it saves every epoch, though only one epoch is really needed to keep around) to run. 