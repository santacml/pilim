#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph of test accuracy vs width, across all HP combos, per r value for pinet.
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





'''  get top 20 hyperparam combos
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/valued-werewolf/**/')  
scan_vals = ['gclip', 'lr', 'wd', 'bias_alpha', 'lr_drop_milestones', 'first_layer_lr_mult', 'last_layer_lr_mult', 'first_layer_alpha', 'last_layer_alpha', 'batch_size']
grouped = bigdf_1.groupby(scan_vals)
# get 20 largest groups by test_acc
largest = grouped['test_acc'].max().nlargest(50)
# re-retrieve each full group 
test = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
# print(test.groupby(scan_vals).size(), "group size")
print(test.groupby(scan_vals).ngroups, "groups found")
# bigdf_1.to_pickle("valued-werewolf-best-50-groups.df")
bigdf_1.to_pickle("valued-werewolf.df")
0/0
# '''

df = pd.read_pickle("cifar10-pifinnet-all.df")
scan_vals = ['gclip', 'lr', 'wd', 'bias_alpha', 'lr_drop_milestones', 'first_layer_lr_mult', 'last_layer_lr_mult', 'first_layer_alpha', 'last_layer_alpha', 'batch_size']



# grouped = df.groupby(scan_vals + ['r', 'width'], as_index=False)['test_acc'].max()
# print(grouped.nlargest(10, "test_acc"))
# 0/0

''' bar graph
new_df = []
# grouped = df[df['width'] == 2048].groupby(scan_vals + ['r'], as_index=False)['test_acc'].max() # best epoch for each r
grouped = df.groupby(scan_vals + ['width', 'r'], as_index=False)['test_acc'].max() # best epoch for each r
for r in [50, 100, 200, 400]:
    test = grouped[grouped['r'] == r]
    over_50 = test[test["test_acc"] > .50].shape[0]
    new_df.append({"r": r, "over_50": over_50 / test.shape[0]})
    print(test.shape[0])
new_df = pd.DataFrame(new_df)
ax = sns.barplot(x="r", y="over_50", data=new_df)
plt.show()
0/0
# '''


''' tsne plot
grouped = df.groupby(scan_vals)
largest = grouped['test_acc'].max().nlargest(50)
# re-retrieve each full group 
df = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
df = df.groupby(scan_vals + ['r', 'width'], as_index=False)['test_acc'].max()
# df = df[df['r'] == 400]
df = df[(df['width'] == 2048) &(df['r'] == 400)]
# df = df[(df['width'] == 2048) &(df['r'] == 400) & (df['test_acc'] > .57)]


# grouped = df.groupby("r")
# print(grouped)
# 0/0

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(df[scan_vals + ['test_acc']]) 
tsne_df = pd.DataFrame()
tsne_df["y"] = df['r']
tsne_df["comp-1"] = z[:,0]
tsne_df["comp-2"] = z[:,1]
tsne_df.to_pickle("tsne_df.df")
tsne_df = pd.read_pickle("tsne_df.df")
tsne_df["y"] = df['test_acc']

sns.scatterplot(x="comp-1", y="comp-2", hue=tsne_df.y.tolist(),
                # palette=sns.color_palette("hls", 3),
                data=tsne_df).set(title="T-SNE for top 50 hyperparam combinations") 
plt.show()
0/0
# '''

grouped = df.groupby(scan_vals)
# largest = grouped['test_acc'].max().nlargest(20)
# re-retrieve each full group 
# df = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
# print(df.groupby(scan_vals).ngroups, "groups found")


# grouped = df.groupby(scan_vals + ['r', 'width'])['test_acc'].max()
# print(grouped)

grouped = df.groupby(scan_vals + ['r', 'width'], as_index=False)['test_acc'].max()
# grouped = grouped[grouped['test_acc'] > .3]

# test = df.loc[df.groupby(scan_vals + ['r', 'width'])['test_acc'].idxmax()]


g = sns.lineplot(x = "width", y = "test_acc", hue='r', data=grouped,  ci='sd' )
# g.set(xscale='log')
# g.set(yscale='log')

# largest2 = grouped['test_acc'].max()
grouped2 = df.groupby(scan_vals)
largest2 = grouped2['test_acc'].max().nlargest(1)
df2 = pd.concat( [ grouped2.get_group(group) for i,group in enumerate( largest2.keys())])
grouped2 = df2.groupby(scan_vals + ['r', 'width'], as_index=False)['test_acc'].max()
# print(grouped2[['test_acc']])
# 0/0
# g = sns.lineplot(x = "width", y = "test_acc", hue='r', data=grouped2, palette='crest')
plt.show() # to show graph

# sns.lineplot(x = "r", y = "test_acc", hue='width', data=grouped,)
# plt.show() # to show graph




0/0

'''  add ntk/gp to stats
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/promoted-puma/**/')   #ntk fine
bigdf_1 = bigdf_1[bigdf_1['depth'] == 2]
max_ntk = bigdf_1.ker_acc.max()

bigdf_1 = load_all_pkl(f'/home/misantac/amlt/sensible-wahoo/**/')   #gp fine
bigdf_1 = bigdf_1[bigdf_1['depth'] == 2]
max_gp = bigdf_1.ker_acc.max()

with open('all_cifar10_kernel_stats.pickle', "rb") as f:
    ALL_STATS = pickle.load(f)
ALL_STATS['ntk'] = max_ntk
ALL_STATS['gp'] = max_gp
with open('all_cifar10_kernel_stats.pickle', 'wb') as handle:
    pickle.dump(ALL_STATS, handle, protocol=pickle.HIGHEST_PROTOCOL)
0/0
# '''


n = 4
colors = pl.cm.viridis(np.linspace(0, .9, n))

'''
dictionary for each key is ALL_STATS[(width, epoch, kernel_reg)]

each value is raw acc
'''

width = 0
epochs = range(0,51)


with open('all_cifar10_kernel_stats.pickle', "rb") as f:
    ALL_STATS = pickle.load(f)

ntk_stats = [ALL_STATS['ntk']] * len(list(epochs))
gp_stats = [ALL_STATS['gp']] * len(list(epochs))

width_stats = []
x = []
for epoch in epochs:
    kernel_stats = ALL_STATS[width][epoch]

    best_acc = 0
    for kernel_reg in [10**(-n) for n in range(1,8)]:
        if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
            best_acc = kernel_stats[(width, epoch, kernel_reg)]
    width_stats.append(best_acc)
    x.append(epoch)
    
inf_line = [max(width_stats)] * len(list(epochs))

plt.subplot(1,1,1)
plt.plot(x, width_stats, label="Pi-Limit Training Kernel Regression", linestyle="solid")
plt.plot(x, gp_stats, label="GP", linestyle=":")
plt.plot(x, ntk_stats, label="NTK", linestyle=":")
plt.plot(x, inf_line, label="Pi-Limit", linestyle="dashed")
plt.legend()
plt.ylabel('Test Kernel Accuracy')
plt.xlabel('Train Epochs')

# plt.savefig("kernel_test.png", bbox_inches='tight')


plt.show()