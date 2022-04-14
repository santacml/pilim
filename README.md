# pilim

Welcome to the repo for [Efficient Computation of Deep, Nonlinear, Infinite-Width Neural Networks that Learn Features](https://www.microsoft.com/en-us/research/publication/efficient-computation-of-deep-nonlinear-infinite-width-neural-networks-that-learn-features/)!


This repo is a work in progress, we will continue to update the repo over time. Please feel free to add an issue if you find a bug or have a question.


# Overview


There are two independent subfolders: [pilimit_orig](pilimit_orig) and [pilimit_lib](pilimit_lib). pilimit_orig contains the original code for the paper for reproducibility, while pilimit_lib contains code for easy pi-net creation. **pilimit_lib is the library you should use if you want to create your own pi-nets.** Each folder has a separate readme with instructions for use.

After writing the paper, we found the code in pilimit_orig difficult for re-use. Given its highly specific nature, here are many "gotcha's" that are hard to work around. We include this library so if someone else wants to double-check our paper or reproduce it exactly, that can be done. pilimit_lib is a rewrite of the library with an emphasis on reusing as many torch classes and functionality as possible. 

It's worth noting pilimit_lib does *not* reproduce the main paper results when using the same hyperparameters due to various floating point and rounding issues, but the results are essentially identical.

# Roadmap

Here are the things we are planning to add to the repo, in order of priority:

- Document pilimit_orig
  - Updated FKR  results
- Document pilimit_lib
  - in-depth class file comments, docstrings: utils, functional, math, tensors, optim
  - in-depth main testing/training file comments: cifar10, test_suite
  - refactor finite net to use activation inside layer?
- dependency lists for both libraries
- Double-check command rerun for accuracy / upload saved .pkl version of each best model
- Create easily usable pip package for pilimit_lib
- Colab notebook for easy reproducibility
- pilimit_lib extensions
  - Fix compare_mlp and _cifar10
  - Separate layernorm layer in pilimit_lib
  - Multiple activation functions
  - refactor output_layer / finite projection