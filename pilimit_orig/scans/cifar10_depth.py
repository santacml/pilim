#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph comparing performance vs depth of pinet, munet, and pilimit.
"""


import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import os
sns.set_theme()
import matplotlib.pylab as pl
import glob
import pandas as pd

def load_all_pkl(folder):
    pkl_paths = glob.glob(os.path.join(folder, "*.df"), recursive=True)
    print(len(pkl_paths))
    list_of_dfs = []
    for path in pkl_paths:
        try:
            df = pd.read_pickle(path)
        except:
            continue
        list_of_dfs.append(df)
    return pd.concat(list_of_dfs)


''' get depth scans
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/more-piranha/**/')
bigdf_1['exp'] = "more-piranha"
bigdf_2 = load_all_pkl(f'/home/misantac/amlt/driven-akita/**/')
bigdf_2['exp'] = "driven-akita"
bigdf_1 = bigdf_1.append(bigdf_2)
bigdf_1.to_pickle("cifar10-depth-mu-pi.df")
# df = pd.read_pickle("cifar10-depth-mu-pi.df")
# bigdf_2 = load_all_pkl(f'/home/misantac/amlt/renewing-alien/**/')
# bigdf_2['exp'] = "renewing-alien"
# df = df.append(bigdf_2)
# df.to_pickle("cifar10-depth-mu-pi.df")
# 0/0
# df = pd.read_pickle("cifar10-depth-mu-pi.df")
# bigdf_2 = load_all_pkl(f'/home/misantac/amlt/handy-bass/**/')
# bigdf_2['exp'] = "handy-bass"
# df = df.append(bigdf_2)
# df.to_pickle("cifar10-depth-mu-pi-inf.df")
# 0/0
# '''


df = pd.read_pickle("cifar10-depth-mu-pi-inf.df")
df = df[df["exp"] != "renewing-alien"]           # accidental run only had depth = 2
df = df.groupby(["exp", "depth"], as_index=False)["test_acc"].max()
print(df)
# 0/0

df.exp.replace("more-piranha", "PiNet",inplace=True)
df.exp.replace("driven-akita", "MuNet",inplace=True)
# df.exp.replace("renewing-alien", "InfNet",inplace=True)
df.exp.replace("handy-bass", "InfNet",inplace=True)

df['depth'] = df['depth'].astype(int)

# pi_depth = bigdf_1.groupby()
g = sns.lineplot(x='depth', y='test_acc', hue="exp", data=df)

plt.ylabel('Test Accuracy')
leg = g.legend()

plt.show()