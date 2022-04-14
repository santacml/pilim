# π-Limit Lib

Introducing easy-to-use, extendable π-networks with torch-like syntax!

For a fast, DIY introduction:

1. Take a peek at [networks.py](examples/networks.py) which defines an infinite and finite MLP to gain a general understanding
2. Quickly run through [minimal_test_mlp.py](tests/minimal_test_mlp.py) which will give you the absolute minimal amount of code for a pi-net, called simply via ```python -m tests.minimal_test_mlp```
3. Run your first experiment on cifar10 using [cifar10mlp.py](cifar10mlp.py) via this command:
```
python cifar10mlp.py --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 400 --batch-size 8 --epochs 50 --width 0 --cuda --seed 0  --depth 2 --bias-alpha 0.5 --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir ./output/
```

## Summary

Infinite π-networks require different functionality than traditional neural networks as a natural result of the infinite-width limit. In the infinite width, projected gradient updates exactly correspond to *concatenation* instead of accumulation.

In plain terms, the gradient update for each input sample will increase the size of A and B (and Amult) matrices in the π-nets and leave the previous weights in the matrices untouched.

Unfortunately torch doesn't like this concept. To keep torch-like functions and especially ```loss.backward()```, certain workarounds are necessary.

All of the library's main code to define the layers is in [inf](inf), with torch-like file naming for components.

There are a few main caveats to this library. Many torch functions work natively, i.e. ```model.parameters()```, ```torch.save(...); model.load_state_dict(...)```, and ```loss.backward()```.

However, many functions will not work and instead have drop-in replacements. We collect the common drop-in replacements below, but note that we cannot guarantee all other torch functions natively work with the π-net (in fact, many probably don't). If you find a useful torch function that fails, please file a bug report.


| Original Torch Implementation |   New π Function |
| ------------- |---------  |
| SGD |  inf.optim.InfSGD |
| clip_grad_norm_ |  inf.utils.store_pi_grad_norm_ + inf.utils.clip_grad_norm_|
| net.apply(i.e. kaiming init) |  net.apply(utils.pi_init)|



Note that only vanilla SGD is implemented right now.

For example usage of π-nets and these custom functions, please refer to [cifar10mlp.py](cifar10mlp.py). 


# Implementation Details

## Inf-Width
The main modules one would use are in [layers.py](inf/layers.py), specifically InfPiInputLinearReLU and InfPiLinearReLU. It is necessary to have a special input layer for π-nets, and the activation function is baked into these layers (in the future we may add more activation functions, but they will still be intra-layer due to the limit formulation).

This is a very important concept to understand: the outputs of these inf-width layers are **pre-activations** and the next layer has the activation function built into it. See the paper for figures on visualizing this process.

Custom autograd functions are defined in [functional.py](inf/functional.py) in order to highjack backpropogation to perform projected inf-width backprop.

## Finite-Width

Finite-width pi-networks are extremely similar to regular MLPs by design. Forward propogation is identical. The differences are:

- initialization (especially sampling from an inf-width network)
- gradient updating (which is projected into r-space)

## The Hacky Side

Here we detail the hacks used to make torch integration happen


InfPiLinearRelu uses a special paramter subclass called InfPiParameter, from [tensors.py](inf/tensors.py), which handles 2 things:
- concatenating incoming gradients
- setting the "pi size" for a given input sample

FinPiLinearReLU also uses a special parameter subclass called FinPiParameter, from [tensors.py](inf/tensors.py), which handles:
- storing omegas, gcovinv, and the pi projection matrix

