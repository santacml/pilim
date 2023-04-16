# pilim

Welcome to the repo for [Efficient Computation of Deep, Nonlinear, Infinite-Width Neural Networks that Learn Features](https://www.microsoft.com/en-us/research/publication/efficient-computation-of-deep-nonlinear-infinite-width-neural-networks-that-learn-features/)!


This repo is a work in progress, we will continue to update the repo over time. Please feel free to add an issue if you find a bug or have a question.

There are two independent subfolders: [pilimit_orig](pilimit_orig) and [pilimit_lib](pilimit_lib). pilimit_orig contains the original code for the paper for reproducibility, while pilimit_lib contains code for easy pi-net creation. **pilimit_lib is the library you should use if you want to create your own pi-nets.** Each folder has a separate readme with instructions for use.

After writing the paper, we found the code in pilimit_orig difficult for re-use. Given its highly specific nature, here are many "gotcha's" that are hard to work around. We include this library so if someone else wants to double-check our paper or reproduce it exactly, that can be done. pilimit_lib is a rewrite of the library with an emphasis on reusing as many torch classes and functionality as possible. 

It's worth noting pilimit_lib does *not* reproduce the main paper results when using the same hyperparameters due to various floating point and rounding issues, but the results are essentially identical.

# Fast Introduction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dS3raZBv87yB2MNXyyAvvvcEP9s2KsRt?usp=sharing)

Introducing easy-to-use, extendable π-networks with torch-like syntax!

For an introduction to the library, take a look at [this Colab notebook](https://colab.research.google.com/drive/1dS3raZBv87yB2MNXyyAvvvcEP9s2KsRt?usp=sharing) for a quick walkthrough on different features available.

For a fast, DIY introduction:

1. Take a peek at [networks.py](experiments/networks/networks.py) which defines an infinite and finite MLP to gain a general understanding
2. Install required dependencies using the conda commands below
3. Quickly run through [minimal_test_mlp.py](tests/minimal_test_mlp.py) which will give you the absolute minimal amount of code for a pi-net, called simply via ```python -m tests.minimal_test_mlp```
4. Run your first experiment on cifar10 using [cifar10mlp.py](experiments/cifar10/cifar10mlp.py) via this command (this will require CUDA be enabled, to run on CPU, please remove --cuda and include --float):
```
python -m experiments.cifar10.cifar10mlp --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 400 --batch-size 8 --epochs 50 --width 0 --cuda --seed 0  --depth 2 --bias-alpha 0.5 --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir ./output/
```

Our intention is for users to create their own networks using pi-net primitives defined in [layers.py](pilimit_lib/inf/layers.py), however, the example InfMLP is a fully featured network in its own right which can be used for experiments (as we did for our paper).


# Conda Environment

To create a conda environment with proper dependencies for both libraries, run the following commands:

Create a new environment if desired:
```
conda create --name pilimit
conda activate pilimit
```

Dependencies:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge 
conda install -c conda-forge opt_einsum
conda install matplotlib psutil seaborn tqdm matplotlib
pip install ipywidgets 
```

MAML specific dependencies:
```
pip install cox torchmeta
conda install tensorboardX pytables
```

# Testing Files

We have a few different files for running pi-nets, summarized in this table:

| File | Purpose |
| ------------- |---------  |
| [cifar10mlp.py](experiments/cifar10/cifar10mlp.py) | main file for finite or inf-width pi-nets running on CIFAR10 | 
| [cifar10test.py](experiments/cifar10/cifar10test.py) | quickly test kernels, finite, or inf-width pi-nets on CIFAR10 without the training code | 
| [transfer_imagenet.py](experiments/imagenet/transfer_imagenet.py) | main file for running transfer learning experiments from imagenet -> CIFAR10 | 
| [minimal_test_mlp.py](experiments/tests/minimal_test_mlp.py)|  extremely minimal dummy test for pi-nets debugging|
| [test_suite_mlp.py](experiments/tests/test_suite_mlp.py) |  compare the inf pi-net to finite-pinet performance on dummy data - results should exactly match for any hparam|
| [compare_mlp.py](experiments/tests/compare_mlp.py) |  compare results of this library with the original pilimit library for accuracy |
| [compare_mlp_cifar10.py](experiments/tests/compare_mlp_cifar10.py) | compare results of this library with the original pilimit library for accuracy on cifar10|


# Experiment Details

Please note that for all experiments, Pi-Nets inherently increase in size for each optimization step, increasing both memory and runtime. To run for a significant number of steps, it will be necessary to run the networks on a CUDA-enabled environment and using float16. It is useful to test things with CPU, and in these cases, note that it is necessary to use float32 for compatibility.

In order to run experiments, create a downloadable library, and also have a copy of the old code, experiments have been placed in their own folder at the root level of the repo. It is necessary to run the files using the commands below (e.g. "python -m experiments.imagenet.transfer_imagenet ...).


# Running Saved Checkpoints

We provide saved versions of the [imagenet network here](https://1drv.ms/u/s!Aqm-bcw66kwDnSfBsbwKG05CxiRK?e=VUahyQ) and the [cifar10 network here](https://1drv.ms/u/s!Aqm-bcw66kwDnSYUPdFw-km20Hta?e=Wu3wd3). Please note these files are quite large (14gb and 9gb), and so require a large gpu/ram to run effectively.

The CIFAR10 checkpoint was created using pilimit_lib, and as such it can be tested with this command:

```
python -m experiments.cifar10.cifar10test --load-model-path PATH_TO_FILE --r 400 --cuda
```

The Imagenet checkpoint was created using pilimit_orig, and therefore needs to be tested with this command
```
python -m experiments.cifar10.cifar10test --load-model-path PATH_TO_FILE --r 200 --load-from-pilimit-orig --cuda --test-kernel
```

## CIFAR10

Cifar10 training can be done with the command above.

## Imagenet

For Imagenet experiments, we use the 32x32 downsampled data that is available for download here: https://image-net.org/download-images.php

By default, the script expects the data to be under experiments/imagenet/data, but feel free to place where you'd like (and adjust the command accordingly).

```
python -m experiments.imagenet.transfer_imagenet  --save-dir=./output/ --save-model --cuda --r 200 --lr=0.01 --batch-size=16 --gclip=0  --epochs=40 --human --wd=0.0001 --bias-alpha=0.5 --first-layer-lr-mult=1.0 --last-layer-lr-mult=1.0 --gclip-per-param 
```

You may test imagenet transfer performance during training by adding these flags, where the transfer milestones represent epochs of training:

```
--transfer --transfer-milestones=0,5,10
```

Otherwise, it is recommended to load a trained network and test the transfer performance at --transfer-milestone=0 (before training).

## MAML
Run your first MAML experiment using:

```
python -m experiments.meta.train dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.5 --batch-size 8 --num-workers 2 --num-epochs 2 --output-folder results --meta-lr .1  --grad-clip 0.1 --meta-momentum 0 --num-shots-test 1 --normalize None --hidden-size -1 --bias-alpha 4 --infnet_r 200 --first-layer-alpha 2  --verbose --first-layer-lr-mult 0.2 --dtype float16 --num-batches 1000 --num-test-batches 500 --adapt-readout-only --Gproj-inner --Gproj-outer --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init --depth 1
```

**Note: there is currently a bug with maml - the pilimit_lib results do not match pilimit_orig. We are investigating this discrepancy. Cifar10 tests are unaffected.**

# Converting from pilimit_orig to pilimit_lib

If one wishes to use a net trained in pilimit_orig in the library pilimit_lib, here are the steps:

1. Convert the network using the script [convert_net.py](pilimit_orig/convert_net.py)
2. Load the network in the new library with the `load_pilimit_orig_net` function in our example [networks.py](experiments/networks/networks.py)

# Roadmap

Here are the things we are planning to add to the repo, in order of priority:

High priority:
- implement loading/testing imagenet and cifar10 saved files
- fix minimal_test for both cpu and gpu
- pilimit_lib maml implementation fix
- gradient accumulation?


Post release:
- Create pip package for pilimit_lib
- pilimit_lib extensions / refactors
  - refactor finite net to use activation inside layer?
  - Fix compare_mlp and _cifar10 (they are currently broken due to dependencies)
  - Separate layernorm layer in pilimit_lib
  - Multiple activation functions
  - refactor gradient clipping
  - refactor storing pi in inf pi parameters, proj in fin pi params?
  - better cifar10/test_suite documentation