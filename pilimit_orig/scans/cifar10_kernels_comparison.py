#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph comparing kernel performance vs epoch of GP, NTK, infnet, munet, and pinet.
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


''' get ntk/gp stats
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/decent-urchin/**/')   #gp reg 
bigdf_1.to_pickle("cifar10-gp-all.df")
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/polite-prawn/**/')   #ntkreg 
bigdf_1.to_pickle("cifar10-ntk-all.df")
'''



'''get fin kernel stats
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/proper-sheep/**/')   # mu, pinet best param kernels
bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/generous-seasnail/**/'))
bigdf_1.to_pickle("cifar10-mu-fin-best-kernels.df")
0/0
'''



''' mu/fin kernels in very broken format
full_df = pd.read_pickle("cifar10-mu-fin-best-kernels.df").reset_index()
new_df = []
for model in ["PiNet", "MuNet"]:
    df = full_df
    ker_accs = {}
    inf_accs = {}
    for index, row in df.iterrows():
        path = row["load_model_path"]
        row_model = "PiNet"
        if path.split("/")[-4] in ["7368690537.25205-af6b1843-8f1d-41a7-9015-e13fb6df3b1a", "7367558822.88152-23dd1c42-6980-4f7f-b084-919859a3947c"]:
            row_model = "MuNet"

        if row_model != model: continue
        # print(path.split("/")[-4])
        # 0/0
        row_width = path.split("/")[-3]
        file = path.split("/")[-1]
        df.at[index, "epoch"] = int(file[5:-3])
        # print(int(file[5:-3]))

        if isinstance(df.at[index, "ker_acc"], torch.Tensor):
            epoch = int(file[5:-3])
            if epoch in ker_accs and df.at[index, "ker_acc"].item() < ker_accs[epoch]: continue
            ker_accs[epoch] = df.at[index, "ker_acc"].item()
        if not np.isnan(df.at[index, "inf_test_acc"]):
            epoch = int(file[5:-3])
            inf_accs[epoch] = df.at[index, "inf_test_acc"]
    for epoch in range(0, 51):
        if not (epoch in ker_accs) or not (epoch in inf_accs): continue
        new_df.append({"epoch": epoch, "Model": model, "ker_acc": ker_accs[epoch], "inf_test_acc": inf_accs[epoch]})
df = pd.DataFrame(new_df)
df.to_pickle("cifar10-mu-fin-best-kernels-processed.df")
# print(df[df["epoch"] == 0])
0/0
# '''






''' cifar10 kernels in very broken format
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/legible-lobster/**/')   # inf kernel
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/next-meerkat/**/'))
# bigdf_1.to_pickle("cifar10-inf-kernels.df")
# 0/0


df = pd.read_pickle("cifar10-inf-kernels.df").reset_index()
new_df = []
ker_accs = {}
inf_accs = {}
for index, row in df.iterrows():
    path = row["load_model_path"]
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
    new_df.append({"epoch": epoch, "ker_acc": ker_accs[epoch], "inf_test_acc": inf_accs[epoch]})
df = pd.DataFrame(new_df)
df.to_pickle("cifar10-inf-kernels-processed.df")
0/0
df = pd.read_pickle("cifar10-inf-kernels-processed.df").reset_index()
# print(list(df.sort_values(by=['epoch'])["ker_acc"]))
# 0/0
# '''

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

'''
dictionary for each key is ALL_STATS[(width, epoch, kernel_reg)]

each value is raw acc
'''

width = 0
epochs = range(0,51)

df = pd.read_pickle("cifar10-inf-kernels-processed.df").reset_index()

gp_df = pd.read_pickle("cifar10-gp-all.df")
gp_df = gp_df[gp_df["depth"] == 2]  # only want depth 2
ntk_df = pd.read_pickle("cifar10-ntk-all.df")
ntk_df = ntk_df[ntk_df["depth"] == 2]  # only want depth 2

#old way of nt
# with open('all_cifar10_kernel_stats.pickle', "rb") as f:
#     ALL_STATS = pickle.load(f)

# ntk_stats = [ALL_STATS['ntk']] * len(list(epochs))
# gp_stats = [ALL_STATS['gp']] * len(list(epochs))

ntk_stats = [float(ntk_df.ker_acc.max())] * len(list(epochs))
gp_stats = [float(gp_df.ker_acc.max())] * len(list(epochs))

inf_stats = list(df.sort_values(by=['epoch'])["inf_test_acc"])
inf_ker_stats = list(df.sort_values(by=['epoch'])["ker_acc"])
x = list(epochs)
inf_line = [max(inf_ker_stats)] * len(list(epochs))

df_2 = pd.read_pickle("cifar10-mu-fin-best-kernels-processed.df")
mu_ker_stats = list(df_2[df_2["Model"] == "MuNet"].sort_values(by=['epoch'])["ker_acc"])
pi_ker_stats = list(df_2[df_2["Model"] == "PiNet"].sort_values(by=['epoch'])["ker_acc"])

plt.subplot(1,1,1)
plt.plot(x, inf_ker_stats, label="Pi-Limit Kernel Accuracy", linestyle="solid")
plt.plot(x, mu_ker_stats, label="Mu-Net", linestyle="solid")
plt.plot(x, pi_ker_stats, label="Finite Pi-Net", linestyle="solid")
# plt.plot(x, inf_stats, label="Pi-Limit Test Accuracy", linestyle="solid")
plt.plot(x, gp_stats, label="GP", linestyle=":")
plt.plot(x, ntk_stats, label="NTK", linestyle=":")
plt.plot(x, inf_line, label="Pi-Limit", linestyle="dashed")
plt.legend()
plt.ylabel('Test Kernel Accuracy')
plt.xlabel('Train Epochs')

# plt.savefig("kernel_test.png", bbox_inches='tight')


plt.show()






















''' old

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
'''