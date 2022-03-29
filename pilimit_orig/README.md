# NOTE: THIS REPO IS NO LONGER MAINTAINED

Please instead go to https://github.com/santacml/pilim, which has both this repo's code and a new codebase which can be used for your own experiments.




# Pi and Other Infinite-Width Limits

This repo contains code to run the Pi limit (as well as some other infinte-width limits) of neural networks on CIFAR10 and MAML.
For example usage, see the tests in `tests/`. 

NOTE: `test_cifar.py` and `test_maml.py` depend crucially on the random number generating setting, so they are not expected to pass on any machine other than Greg's at the moment.