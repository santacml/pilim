#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph of kernel performance vs epoch for many widths of pinet, and infnet.

    This data uses a very old format I don't use any more, it's the only file that still uses it. 
    It's just using a random, decent infnet hyperparameter combo.

    I tried recreating this graph using the best overall infnet HP combo and the results were very messy.

    We decided to stick with the old graph, but I kept the new code for reference. The new code WILL NOT WORK.
    Just run this as-is.
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
import torch


''' new cifar10 kernel graph
full_df = pd.read_pickle("cifar10-fin-kernels.df").reset_index()
new_df = []
for width in  [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    df = full_df
    ker_accs = {}
    inf_accs = {}
    for index, row in df.iterrows():
        path = row["load_model_path"]
        # print(path)
        row_width = path.split("/")[-3]
        if row_width != str(width): continue
        file = path.split("/")[-1]
        df.at[index, "epoch"] = int(file[5:-3])

        if isinstance(df.at[index, "ker_acc"], torch.Tensor):
            epoch = int(file[5:-3])
            if epoch in ker_accs and df.at[index, "ker_acc"].item() < ker_accs[epoch]: continue
            ker_accs[epoch] = df.at[index, "ker_acc"].item()
        if not np.isnan(df.at[index, "inf_test_acc"]):
            epoch = int(file[5:-3])
            inf_accs[epoch] = df.at[index, "inf_test_acc"]
    for epoch in range(0, 51):
        if not (epoch in ker_accs) or not (epoch in inf_accs): continue
        new_df.append({"epoch": epoch, "width": width, "ker_acc": ker_accs[epoch], "inf_test_acc": inf_accs[epoch]})
df = pd.DataFrame(new_df)
df.to_pickle("cifar10-fin-kernels-processed.df")
print(df)
0/0
'''

''' new kernel graph - very messy, keep old
df = pd.read_pickle("cifar10-inf-kernels-processed.df").reset_index()
df["width"] = "inf"
df = df.append(pd.read_pickle("cifar10-fin-kernels-processed.df")).reset_index()
# df = df[df["width"] > 500]


g = sns.lineplot(x='epoch', y='ker_acc', hue='width', data=df)
plt.show()
0/0
'''

n = 8
colors = pl.cm.plasma(np.linspace(.1, .7, n))

'''
dictionary for each key is ALL_STATS[(width, epoch, kernel_reg)]

each value is raw acc
'''

widths = [500, 1000, 5000, 10000, 20000, 30000, 40000, 0]

'''   without the stats saved locally, process all the results and create total results pkl
ALL_STATS = {}
for width in widths:
    ALL_STATS[width] = [0] * 51
for n, width in enumerate(widths):
    width_stats = []
    x = []
    for epoch in range(0, 51):
        if not os.path.exists(f'/home/misantac/teamdrive/inflimitmsr/kerneltest_full/kernel_stats_{10}cls_{7}rdg_{width}width_{epoch}epochs.pickle'):
            print("SKIPPING", width, epoch)
            continue

        with open(f'/home/misantac/teamdrive/inflimitmsr/kerneltest_full/kernel_stats_{10}cls_{7}rdg_{width}width_{epoch}epochs.pickle', 'rb') as handle:
            kernel_stats = pickle.load(handle)

        ALL_STATS[width][epoch] = kernel_stats

        best_acc = 0
        for kernel_reg in [10**(-n) for n in range(1,8)]:
            if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
                best_acc = kernel_stats[(width, epoch, kernel_reg)]
        width_stats.append(best_acc)
        x.append(epoch)
    
    plt.subplot(1,1,1)
    plt.plot(x, width_stats, label=str(width) if width != 0 else "Infinite", linestyle="dashed" if width == 0 else "solid", color=colors[n])
    plt.legend()
    plt.ylabel('Test Kernel Accuracy')
    plt.xlabel('Train Epochs')

# plt.savefig("kernel_test.png", bbox_inches='tight')

with open('all_cifar10_kernel_stats.pickle', 'wb') as handle:
    pickle.dump(ALL_STATS, handle, protocol=pickle.HIGHEST_PROTOCOL)
plt.show()
# '''



with open('all_cifar10_kernel_stats.pickle', "rb") as f:
    ALL_STATS = pickle.load(f)

for n, width in enumerate(widths):
    width_stats = []
    x = []
    for epoch in range(0, 51):
        kernel_stats = ALL_STATS[width][epoch]

        best_acc = 0
        for kernel_reg in [10**(-n) for n in range(1,8)]:
            if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
                best_acc = kernel_stats[(width, epoch, kernel_reg)]
        width_stats.append(best_acc)
        x.append(epoch)
    
    plt.subplot(1,1,1)
    plt.plot(x, width_stats, label=str(width) if width != 0 else "Infinite", linestyle="dashed" if width == 0 else "solid", color=colors[n])
    plt.legend()
    plt.ylabel('Test Kernel Accuracy')
    plt.xlabel('Train Epochs')

# plt.savefig("kernel_test.png", bbox_inches='tight')


# plt.show()