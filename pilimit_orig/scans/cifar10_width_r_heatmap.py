#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a heatmap comparing width vs r vs performance for pinet on cifar10.
"""

import pandas as pd
import glob
import os
import seaborn as sns
sns.set_theme()
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np

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

''' collect results - run on sandbox after downloading amlt results
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/safe-bass/**/')    #munet  
bigdf_1.to_pickle("cifar10-mu-all.df")
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/valued-werewolf/**/')   #finpinet 
# bigdf_1.to_pickle("cifar10-pifinnet-all.df")
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/evident-kingfish/**/')   #infnet
# bigdf_1.to_pickle("cifar10-inf-all.df")
'''

mu_df = pd.read_pickle("cifar10-mu-all.df")
new_df = []
for width in [128, 256, 512, 1024, 2048]:
    df_row = mu_df[mu_df['width'] == width]
    df_row = df_row[df_row.test_acc == df_row.test_acc.max()]
    new_df.append(df_row)
df =  pd.concat(new_df)

inf_df = pd.read_pickle("cifar10-inf-all.df")
new_df = []
for r in [50, 100, 200, 400]:
    df_row = inf_df[inf_df['r'] == r]
    df_row = df_row[df_row.test_acc == df_row.test_acc.max()]
    new_df.append(df_row)
inf_df =  pd.concat(new_df)

fin_df = pd.read_pickle("cifar10-pifinnet-all.df")
new_df = []
for width in [128, 256, 512, 1024, 2048]:
    for r in [50, 100, 200, 400]:
        df_row = fin_df[((fin_df['r'] == r) & (fin_df['width'] == width))]
        df_row = df_row[df_row.test_acc == df_row.test_acc.max()]
        new_df.append(df_row)
new_df.append(inf_df)
new_df =  pd.concat(new_df)






new_df['r'] = new_df['r'].astype(int)
df["r"] = 'MuNet'
new_df = new_df.append(df)[['width', 'r', 'test_acc']].drop_duplicates()

new_df['width'].fillna(0, inplace=True)
new_df['width'] = new_df['width'].astype(int)
new_df['width'].replace(0, "Infinite", inplace=True)
new_df['test_acc']  *= 100
print(new_df[['width', 'r', 'test_acc']])
new_df = new_df.pivot("r", "width", "test_acc")
ax = sns.heatmap(new_df, annot=True, fmt=".2f", cmap='magma')
ax.invert_yaxis()
plt.ylabel('r')
plt.xlabel('Width')

yticks = ax.get_yticks()
yticks[-1] = 4.75
ylabels = ax.get_yticklabels()
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)

# https://stackoverflow.com/questions/62773944/insert-line-in-heatmap-after-every-7th-column
b, t = plt.xlim()
ax.hlines(y = 4, xmin = b-1, xmax = t, colors = 'white', lw = 5)
plt.show()