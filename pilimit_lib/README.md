# π-Limit Lib

Introducing easy-to-use, extendable π-networks with torch-like syntax!

For a fast introduction, take a peek at [networks.py](examples/networks.py) which defines an infinite and finite MLP, then run the network on [cifar10mlp.py](cifar10mlp.py) to run the MLP on cifar10.

All of the library's main code to define the layers is in [inf](inf), with torch-like file naming for components.

There are a few main caveats to this library. Many torch functions work natively, i.e. ```model.parameters()```, ```torch.save(...); model.load_state_dict(...)```, and ```loss.backward()```.

However, many functions will not work and instead have drop-in replacements. We collect the common drop-in replacements below, but note that we cannot guarantee all other torch functions natively work with the pi-net (in fact, many probably don't). If you find a useful torch function that fails, please file a bug report.


| Original Torch Implementation |   New Pi Function |
| ------------- |---------  |
| SGD |  inf.optim.InfSGD |
| clip_grad_norm_ |  inf.utils.store_pi_grad_norm_ + inf.utils.clip_grad_norm_|

Note that only vanilla SGD is implemented right now.

For example usage of pi-nets and these custom functions, please refer to[cifar10mlp.py](cifar10mlp.py). 

## Explanation

Infinite π-networks function differently from traditional neural networks as a natural result of the infinite-width limit. In the infinite width, projected gradient updates can be represented through *concatenation* instead of accumulation.

