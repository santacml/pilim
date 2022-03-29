# pilim
Pi-Limit Torch-Style Library and Original Paper Code


## Structure

There are two main folders, pilimit_orig and pilimit_lib. pilimit_orig contains the original code for the paper for reproducibility, with light documentation. 

After writing the paper, we found that it would be difficult for someone interested in creating a pi-net to reuse this code. Given they have a custom method for gradient updating, performing the backwards calculation, etc., there are many "gotcha's" that are hard to work around. We include this library so if someone else wants to double-check our paper or reproduce it exactly, that can be done.

Therefore, pilimit_lib is a rewrite of the library with an emphasis on reusing as many torch functions as possible, with heavier documentation. It's worth noting this library does *not* reproduce the main paper results exactly, mostly due to various floating point issues, but the results are essentially identical.