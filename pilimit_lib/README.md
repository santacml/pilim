# π-Limit Lib

Introducing easy-to-use, extendable π-networks with torch-like syntax!

For a fast introduction, take a peek at [networks.py](examples/networks.py) which defines an infinite and finite MLP, then run the network on [cifar10mlp.py](cifar10mlp.py) to run the MLP on cifar10.

All of the library's main code to define the layers is in [inf](inf), with torch-like file naming for components.

There are a few main caveats to this library. Many torch functions work natively, i.e. ```model.parameters()``` and ```loss.backward()```.

However, many functions will not work and instead have drop-in replacements, such as:

- SGD -> InfSGD
- clip


## Explanation

Infinite π-networks function differently from traditional neural networks as a natural result of the infinite-width limit. In the infinite width, projected gradient updates can be represented through *concatenation* instead of accumulation.

