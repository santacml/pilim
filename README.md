# pilim
Pi-Limit Torch-Style Library and Original Paper Code

This repo is a work in progress, we will continue to update the repo over time. Please feel free to add an Issue if you find a bug or have a question.


# Structure

There are two main folders, [pilimit_orig](pilimit_orig) and [pilimit_lib](pilimit_lib). pilimit_orig contains the original code for the paper for reproducibility, with very light documentation. pilimit_lib is the library you should use if you want to create your own pi-nets. Each folder has a separate readme with instructions for use.

After writing the paper, we found the code in pilimit_orig difficult for re-use. Given its highly specific nature, here are many "gotcha's" that are hard to work around. We include this library so if someone else wants to double-check our paper or reproduce it exactly, that can be done. 

Extending on pilimit_orig, pilimit_lib is a rewrite of the library with an emphasis on reusing as many torch classes and functions as possible, resulting in an easy-to-use library for creating infinite width pi-nets. 

It's worth noting pilimit_lib does *not* reproduce the main paper results when using the same hyperparameters due to various floating point and rounding issues, but the results are essentially identical.

# Roadmap

Here are the things we are planning to add to the repo, in order of priority:

- Document original repo
  - Updated FKR  results
- Document torch-style repo
  - write readme, caveats with .parameters, sgd, gclip, backward explanation, etc and example usage writeup
  - in-depth class file comments, docstrings
  - in-depth main testing/training file comments
  - Fix compare_mlp and _cifar10
  - refactor tensors
- dependency lists for both libraries
- Double-check command rerun for accuracy / upload saved .pkl version of each best model
- Create easily usable pip package for pilimit_lib
- Colab notebook for easy reproducibility