#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph of test accuracy vs width, across all HP combos, per r value for pinet.

    Under heavy testing.
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


# df = pd.read_pickle("maml-infnet-all.df")
# print(df.nlargest(5, 'val_accuracies_after')[['train_accuracies_after', 'hidden_size', 'infnet_r', 'val_accuracies_after', 'step_size', 'meta_lr', 'grad_clip', 'bias_alpha', 'first_layer_alpha', 'first_layer_lr_mult']])
# 0/0



df = pd.read_pickle("maml-pifinnet-all.df")
scan_vals = [ 'step_size', 'meta_lr', 'grad_clip', 'bias_alpha', 'first_layer_alpha', 'first_layer_lr_mult']

''' find best per r, width
seeds = list(range(5))
bigdf_1 = pd.read_pickle("maml-pifinnet-all.df").reset_index(drop=True)
new_df = []
for width in [128, 256, 512, 1024, 2048]:
    for r in [50, 100, 200, 400]:
        df_row = bigdf_1[((bigdf_1['infnet_r'] == r) & (bigdf_1['hidden_size'] == width))]
        df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]
        new_df.append(df_row)
        print(width, r, df_row.train_accuracies_after.max(), df_row.val_accuracies_after.max())
new_df =  pd.concat(new_df)
0/0
# '''

# df = df[df['infnet_r'] == 400]
# df = df[df['hidden_size'] == 2048]
# print(df.nlargest(5, 'val_accuracies_after')[['exp', 'train_accuracies_after', 'hidden_size', 'infnet_r', 'val_accuracies_after', 'step_size', 'meta_lr', 'grad_clip', 'bias_alpha', 'first_layer_alpha', 'first_layer_lr_mult']])
# 0/0


# grouped = df.groupby(scan_vals)
# largest = grouped['val_accuracies_after'].max().nlargest(50)
# df = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
# df = df.groupby(scan_vals + ['infnet_r', 'hidden_size'], as_index=False)['val_accuracies_after'].max()


# print(df.nlargest(5, 'val_accuracies_after')[['train_accuracies_after', 'hidden_size', 'infnet_r', 'val_accuracies_after', 'step_size', 'meta_lr', 'grad_clip', 'bias_alpha', 'first_layer_alpha', 'first_layer_lr_mult']])
# 0/0


grouped = df.groupby(scan_vals)
# largest = grouped['test_acc'].max().nlargest(20)
# re-retrieve each full group 
# df = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
# print(df.groupby(scan_vals).ngroups, "groups found")


# grouped = df.groupby(scan_vals + ['r', 'width'])['test_acc'].max()
# print(grouped)

grouped = df.groupby(scan_vals + ['infnet_r', 'hidden_size'], as_index=False)['val_accuracies_after'].max()
# grouped = grouped[grouped['test_acc'] > .3]

# test = df.loc[df.groupby(scan_vals + ['r', 'width'])['test_acc'].idxmax()]


g = sns.lineplot(x = "hidden_size", y = "val_accuracies_after", hue='infnet_r', data=grouped,  ci='sd' )
# g.set(xscale='log')
# g.set(yscale='log')

# largest2 = grouped['test_acc'].max()
# grouped2 = df.groupby(scan_vals)
# largest2 = grouped2['test_acc'].max().nlargest(1)
# df2 = pd.concat( [ grouped2.get_group(group) for i,group in enumerate( largest2.keys())])
# grouped2 = df2.groupby(scan_vals + ['r', 'width'], as_index=False)['test_acc'].max()
# print(grouped2[['test_acc']])
# 0/0
# g = sns.lineplot(x = "width", y = "test_acc", hue='r', data=grouped2, palette='crest')
plt.show() # to show graph

# sns.lineplot(x = "r", y = "test_acc", hue='width', data=grouped,)
# plt.show() # to show graph



''' tsne plot
grouped = df.groupby(scan_vals)
largest = grouped['val_accuracies_after'].max().nlargest(50)
# re-retrieve each full group 
df = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
df = df.groupby(scan_vals + ['infnet_r', 'hidden_size'], as_index=False)['val_accuracies_after'].max()
# df = df[df['r'] == 400]
# df = df[(df['width'] == 2048) &(df['r'] == 400)]
# df = df[(df['width'] == 2048) &(df['r'] == 400) & (df['test_acc'] > .57)]


# grouped = df.groupby("r")
# print(grouped)
# 0/0

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(df[scan_vals + ['val_accuracies_after']]) 
z = tsne.fit_transform(df[scan_vals]) 
tsne_df = pd.DataFrame()
tsne_df["y"] = df['infnet_r']
tsne_df["comp-1"] = z[:,0]
tsne_df["comp-2"] = z[:,1]
# tsne_df.to_pickle("tsne_df.df")
# tsne_df = pd.read_pickle("tsne_df.df")
# tsne_df["y"] = df['val_accuracies_after']

sns.scatterplot(x="comp-1", y="comp-2", hue=tsne_df.y.tolist(),
                # palette=sns.color_palette("hls", 3),
                data=tsne_df).set(title="T-SNE for top 50 hyperparam combinations") 
plt.show()
0/0
# '''
