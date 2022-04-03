# pilim
Pi-Limit Torch-Style Library and Original Paper Code

This repo is a work in progress, we will continue to update the repo over time. Please feel free to add an Issue if you find a bug or have a question.


# Structure

There are two main folders, [pilimit_orig](pilimit_orig) and [pilimit_lib](pilimit_lib). pilimit_orig contains the original code for the paper for reproducibility, with light documentation. Each folder has a separate readme with instructions for use.

After writing the paper, we found the code difficult for re-use. Given its highly specific nature, here are many "gotcha's" that are hard to work around. We include this library so if someone else wants to double-check our paper or reproduce it exactly, that can be done. 

Therefore, pilimit_lib is a rewrite of the library with an emphasis on reusing as many torch functions as possible, with heavier documentation. It's worth noting this library does *not* reproduce the main paper results exactly, mostly due to various floating point issues, but the results are essentially identical.

# Roadmap

Here are the things we are planning to add to the repo, in order of priority:

- Document original repo
  - MAML hparams
  - transfer imagenet hparams, numbers
  - add comments and explanations to main testing/training/class files
  - Updated FKR and transfer results
  - Add caption to each graphing file to describe which figure it is
  - Remove unnecessary files and any useless comments, add proper formatting in important files
- Document torch-style repo
  - main testing/training file comments 
  - caveats with .parameters, sgd, gclip, backward explanation, etc
- Upload saved .pkl version of each best model
- Colab notebook for easy reproducibility