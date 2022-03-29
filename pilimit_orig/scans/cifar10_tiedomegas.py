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
# bigdf_1 = load_all_pkl(f'/home/azureuser/amlt/talented-doe/**/')
# bigdf_1['exp'] = "talented-doe"
# bigdf_2 = load_all_pkl(f'/home/azureuser/amlt/ready-mongoose/**/')
# bigdf_2['exp'] = "ready-mongoose"
# bigdf_1 = bigdf_1.append(bigdf_2)
# bigdf_1.to_pickle("cifar10-tiedomegas.df")
# 0/0


bigdf_1 = load_all_pkl(f'/home/azureuser/amlt/close-sloth/**/')
bigdf_1['exp'] = "close-sloth"
bigdf_2 = load_all_pkl(f'/home/azureuser/amlt/endless-macaw/**/')
bigdf_2['exp'] = "endless-macaw"
bigdf_1 = bigdf_1.append(bigdf_2)
bigdf_1.to_pickle("cifar10-tiedomegas-2.df")
0/0
# '''


df = pd.read_pickle("cifar10-tiedomegas-2.df")
tied = df[df["exp"] ==  "close-sloth"]
# tied = df[df["exp"] ==  "talented-doe"]
# tied = tied[tied["test_acc"] == tied["test_acc"].max()]
# print(tied["seed"].unique())
# 0/0
# tied = tied[tied["seed"] == 4]
untied = df[df["exp"] ==  "endless-macaw"]
# untied = df[df["exp"] ==  "ready-mongoose"]
# untied = untied[untied["seed"] == 15]
# df = df.groupby(["exp", "tie_omegas"], as_index=False)["test_acc"].max()

df = tied.append(untied).reset_index()
# print(df)
# 0/0

# df.exp.replace("ready-mongoose", "Untied Omegas",inplace=True)
# df.exp.replace("talented-doe", "Tied Omegas",inplace=True)
df.exp.replace("endless-macaw", "Untied Omegas",inplace=True)
df.exp.replace("close-sloth", "Tied Omegas",inplace=True)

# pi_depth = bigdf_1.groupby()
# g = sns.lineplot(x='epoch', y='train_loss', hue="exp", data=df)
g = sns.lineplot(x='epoch', y='test_loss', hue="exp", data=df)

# plt.ylabel('Test Accuracy')
# leg = g.legend()

plt.legend()
plt.ylabel('Test Loss')
plt.xlabel('Train Epochs')

plt.savefig("omegas tied untied test loss.pdf", bbox_inches='tight')
plt.show()