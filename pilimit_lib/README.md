# π-Limit Lib

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