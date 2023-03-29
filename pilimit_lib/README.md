# π-Limit Lib

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dS3raZBv87yB2MNXyyAvvvcEP9s2KsRt?usp=sharing)

Introducing easy-to-use, extendable π-networks with torch-like syntax!

For a fast, DIY introduction:

1. Take a peek at [networks.py](experiments/networks/networks.py) which defines an infinite and finite MLP to gain a general understanding
2. Quickly run through [minimal_test_mlp.py](tests/minimal_test_mlp.py) which will give you the absolute minimal amount of code for a pi-net, called simply via ```python -m tests.minimal_test_mlp```
3. Run your first experiment on cifar10 using [cifar10mlp.py](cifar10mlp.py) via this command (this will require CUDA be enabled, to run on CPU, please remove --cuda and include --float):
```
python -m experiments.cifar10.cifar10mlp --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 400 --batch-size 8 --epochs 50 --width 0 --cuda --seed 0  --depth 2 --bias-alpha 0.5 --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir ./output/
```

After you've verified everything is working, take a look at [this Colab notebook](https://colab.research.google.com/drive/1dS3raZBv87yB2MNXyyAvvvcEP9s2KsRt?usp=sharing) for a quick walkthrough on different features of the library.

Our intention is for users to create their own networks using pi-net primitives defined in [layers.py](inf/layers.py), however, the example InfMLP is a fully feature network in its own right which can be used for experiments (as we did for our paper).

## Summary

Infinite π-networks require different functionality than traditional neural networks as a natural result of the infinite-width limit. In the infinite width, projected gradient updates exactly correspond to *concatenation* instead of accumulation.

In plain terms, the gradient update for each input sample will increase the size of A and B (and Amult) matrices in the π-nets and leave the previous weights in the matrices untouched. Unfortunately torch doesn't like this concept. To keep torch-like functions and especially ```loss.backward()```, certain workarounds are necessary.

There are a few main caveats to this library. Many torch functions work natively, i.e. ```model.parameters()```, ```torch.save(...); model.load_state_dict(...)```, and ```loss.backward()```.

However, many functions will not work and instead have drop-in replacements. We collect the common drop-in replacements below, but note that we cannot guarantee all other torch functions natively work with the π-net (in fact, many probably don't). If you find a useful torch function that fails, please file a bug report.


| Original Torch Implementation |   New π Function |
| ------------- |---------  |
| SGD |  inf.optim.InfSGD | 
| clip_grad_norm_ |  inf.utils.store_pi_grad_norm_ + inf.utils.clip_grad_norm_|
| net.apply(i.e. kaiming init) |  net.apply(utils.pi_init)|


Note that only vanilla SGD is implemented right now.

# Testing Files

We have a few different files for running pi-nets, summarized in this table:



| File | Purpose |
| ------------- |---------  |
| [cifar10mlp.py](experiments/cifar10/cifar10mlp.py) | main file for finite or inf-width pi-nets running on CIFAR10 | 
| [transfer_imagenet.py](experiments/imagenet/transfer_imagenet.py) | main file for running transfer learning experiments from imagenet -> CIFAR10 | 
| [minimal_test_mlp.py](pilimit_lib/tests/minimal_test_mlp.py)|  extremely minimal dummy test for pi-nets debugging|
| [test_suite_mlp.py](pilimit_lib/tests/test_suite_mlp.py) |  compare the inf pi-net to finite-pinet performance on dummy data - results should exactly match for any hparam|
| [compare_mlp.py](pilimit_lib/compare_mlp.py) |  compare results of this library with the original pilimit library for accuracy |
| [compare_mlp_cifar10.py](pilimit_lib/compare_mlp_cifar10.py) | compare results of this library with the original pilimit library for accuracy on cifar10|


# Experiment Details

Please note that for all experiments, Pi-Nets inherently increase in size for each optimization step, increasing both memory and runtime. To run for a significant number of steps, it will be necessary to run the networks on a CUDA-enabled environment and using float16. It is useful to test things with CPU, and in these cases, note that it is necessary to use float32 for compatibility.


## Imagenet

For Imagenet experiments, we use the 32x32 downsampled data that is available for download here: https://image-net.org/download-images.php

By default, the script expects the data to be under imagenet/data, but feel free to place where you'd like (and adjust the command accordingly).

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


# Implementation Details

## Inf-Width
The main modules one would use are in [layers.py](inf/layers.py), specifically InfPiInputLinearReLU and InfPiLinearReLU. It is necessary to have a special input layer for π-nets, and the activation function is baked into these layers (in the future we may add more activation functions, but they will still be intra-layer due to the limit formulation).

This is a very important concept to understand: the outputs of these inf-width layers are **pre-activations** and the next layer has the activation function built into it. See the paper Figures 2 and 3 to visualize this process.

Custom autograd functions are defined in [functional.py](inf/functional.py) in order to highjack backpropogation to perform projected inf-width backprop. The custom autograd functions themselves build on [math.py](inf/math.py) which contains primitives for inf-width operations (i.e. V-transforms).


## Finite-Width

Finite-width pi-networks are extremely similar to regular MLPs by design. Forward propogation is identical. The differences are:

- initialization (especially sampling from an inf-width network)
- gradient updating (which is projected into r-space)

## The Hacky Side

Here we detail the hacks used to make torch integration happen


InfPiLinearRelu uses a special paramter subclass called InfPiParameter, from [tensors.py](inf/tensors.py), which handles a few things things:
- concatenating incoming gradients
- setting the "pi size" for a given input sample
- storing pi_grad_norm for gradient clipping
- whether to apply the lr to this parameter or not (lr/wd only used for Amult)

FinPiLinearReLU also uses a special parameter subclass called FinPiParameter, from [tensors.py](inf/tensors.py), which handles:
- storing omegas, gcovinv, and the pi projection matrix


For both finite and infinite width pi-networks, certain things need to happen in the gradient update that are abnormal. For infinite-width networks this means appending the gradient instead of accumulating it, and for finite-width networks this means projecting the gradient before accumulating it.

In both the finite or infinite-width cases, special variables are stored as attributes in the parameters themselves. This is not desirable for a variety of reasons, and we plan to refactor (especially gradient clipping, where temporary norms for A*B are stored in the A parameter). But for now, they are functional.

The special logic for updating finite and infinite parameters are handled inside [optim.py](inf/optim.py), where a custom PiSGD optimizer detects if a parameters is one of these two classes for appropriate action. In the future, we'd like to refactor this so other optimizers can be easily used.