# π-Limit Lib

Introducing easy-to-use, extendable π-networks with torch-like syntax!

For a fast introduction, take a peek at [networks.py](examples/networks.py) which defines an infinite and finite MLP, take a look at [minimal_test_mlp.py](tests/minimal_test_mlp.py) which will give you the absolute minimal amount of code for a pi-net, then run the network on [cifar10mlp.py](cifar10mlp.py) for a fuller example of running an example MLP on cifar10.

All of the library's main code to define the layers is in [inf](inf), with torch-like file naming for components.

There are a few main caveats to this library. Many torch functions work natively, i.e. ```model.parameters()```, ```torch.save(...); model.load_state_dict(...)```, and ```loss.backward()```.

However, many functions will not work and instead have drop-in replacements. We collect the common drop-in replacements below, but note that we cannot guarantee all other torch functions natively work with the π-net (in fact, many probably don't). If you find a useful torch function that fails, please file a bug report.


| Original Torch Implementation |   New π Function |
| ------------- |---------  |
| SGD |  inf.optim.InfSGD |
| clip_grad_norm_ |  inf.utils.store_pi_grad_norm_ + inf.utils.clip_grad_norm_|

Note that only vanilla SGD is implemented right now.

For example usage of π-nets and these custom functions, please refer to [cifar10mlp.py](cifar10mlp.py). 

## Explanation

Infinite π-networks require different functionality than traditional neural networks as a natural result of the infinite-width limit. In the infinite width, projected gradient updates exactly correspond to *concatenation* instead of accumulation.

In plain terms, the gradient update for each input sample will increase the size of A and B (and Amult) matrices in the π-nets and leave the previous weights in the matrices untouched.

Unfortunately torch doesn't like this concept. To keep torch-like functions and especially ```loss.backward()```, certain workarounds are necessary.

## Implementation Details

The main modules one would use are in [layers.py](inf/layers.py), specifically InfPiInputLinearReLU and InfPiLinearReLU. It is necessary to have a special input layer for π-nets, and the activation function is baked into these layers (in the future we may add more activation functions, but they will still be intra-layer due to the limit formulation).

InfPiLinearRelu uses a special paramter subclass called InfPiParameter, from [tensors.py](inf/tensors.py), which handles 2 things:
- concatenating incoming gradients
- setting the "pi size" for a given input sample

