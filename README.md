# pilim

Welcome to the repo for [Efficient Computation of Deep, Nonlinear, Infinite-Width Neural Networks that Learn Features](https://www.microsoft.com/en-us/research/publication/efficient-computation-of-deep-nonlinear-infinite-width-neural-networks-that-learn-features/)!


This repo is a work in progress, we will continue to update the repo over time. Please feel free to add an issue if you find a bug or have a question.


# Overview


There are two independent subfolders: [pilimit_orig](pilimit_orig) and [pilimit_lib](pilimit_lib). pilimit_orig contains the original code for the paper for reproducibility, while pilimit_lib contains code for easy pi-net creation. **pilimit_lib is the library you should use if you want to create your own pi-nets.** Each folder has a separate readme with instructions for use.

After writing the paper, we found the code in pilimit_orig difficult for re-use. Given its highly specific nature, here are many "gotcha's" that are hard to work around. We include this library so if someone else wants to double-check our paper or reproduce it exactly, that can be done. pilimit_lib is a rewrite of the library with an emphasis on reusing as many torch classes and functionality as possible. 

It's worth noting pilimit_lib does *not* reproduce the main paper results when using the same hyperparameters due to various floating point and rounding issues, but the results are essentially identical.

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


# Roadmap

Here are the things we are planning to add to the repo, in order of priority:

- double-check command rerun for accuracy / upload saved .pkl version of each best model
- double-check pilimit_lib tests still work after refactor
- review all docmentation and add details where appropriate (esp functional.py)
- Create pip package for pilimit_lib
- Colab notebook for easy reproducibility
- nicer dependency / conda env creation file
- pilimit_lib extensions / refactors
  - pilimit_lib maml implementation
  - refactor finite net to use activation inside layer?
  - Fix compare_mlp and _cifar10 (they are currently broken due to dependencies)
  - Separate layernorm layer in pilimit_lib
  - Multiple activation functions
  - refactor gradient clipping
  - refactor storing pi in inf pi parameters, proj in fin pi params
  - better cifar10/test_suite documentation