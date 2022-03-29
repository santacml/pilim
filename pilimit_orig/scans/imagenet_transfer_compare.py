#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph comparing imagenet performance of munet vs pinet.

    THIS IS NO LONGER NECESSARY - KEEPING FOR REFERENCE.
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

n = 8
colors = pl.cm.plasma(np.linspace(.1, .7, n))

'''
dictionary for each key is ALL_STATS[(width, epoch, kernel_reg)]

each value is raw acc
'''

# widths = [10000, 20000]
widths = [2000]

'''   without the stats saved locally
ALL_STATS_MU = {}
for width in widths:
    ALL_STATS_MU[width] = [0] * 31
for n, width in enumerate(widths):
    width_stats = []
    for epoch in range(1, 31):
        if not os.path.exists(f'/home/misantac/teamdrive/inflimitmsr/imagenet250transfer_gauss_randomcls_2/0.01/kernel_stats_{10}cls_{6}rdg_{width}width_{epoch}epochs.pickle'):
            print("SKIPPING", width, epoch)
            continue

        with open(f'teamdrive/inflimitmsr/imagenet250transfer_gauss_randomcls_2/0.01/kernel_stats_{10}cls_{6}rdg_{width}width_{epoch}epochs.pickle', 'rb') as handle:
            kernel_stats = pickle.load(handle)

        ALL_STATS_MU[width][epoch] = kernel_stats

        best_acc = 0
        for kernel_reg in [10**(-n) for n in range(6,0,-1)]:
            if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
                best_acc = kernel_stats[(width, epoch, kernel_reg)]
        width_stats.append(best_acc)

        
ALL_STATS_PI = {}
for width in widths:
    ALL_STATS_PI[width] = [0] * 31
for n, width in enumerate(widths):
    width_stats = []
    for epoch in range(1, 31):
        if not os.path.exists(f'/home/misantac/teamdrive/inflimitmsr/imagenet250transfer_pifinnet_randomcls_2/0.005/kernel_stats_{10}cls_{6}rdg_{width}width_{epoch}epochs.pickle'):
            print("SKIPPING", width, epoch)
            continue

        with open(f'teamdrive/inflimitmsr/imagenet250transfer_pifinnet_randomcls_2/0.005/kernel_stats_{10}cls_{6}rdg_{width}width_{epoch}epochs.pickle', 'rb') as handle:
            kernel_stats = pickle.load(handle)

        ALL_STATS_PI[width][epoch] = kernel_stats

        best_acc = 0
        for kernel_reg in [10**(-n) for n in range(6,0,-1)]:
            if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
                best_acc = kernel_stats[(width, epoch, kernel_reg)]
        width_stats.append(best_acc)

ALL_STATS = {"mu": ALL_STATS_MU, "pi": ALL_STATS_PI}

with open('all_mu_pi_imagenet_transfer_kernel_stats_2.pickle', 'wb') as handle:
    pickle.dump(ALL_STATS, handle, protocol=pickle.HIGHEST_PROTOCOL)
# plt.show()
0/0
# '''



with open('all_mu_pi_imagenet_transfer_kernel_stats_2.pickle', "rb") as f:
# with open('all_mu_pi_imagenet_transfer_kernel_stats.pickle', "rb") as f:
# with open('all_mu_pi_imagenet_transfer_kernel_stats005.pickle', "rb") as f:
    ALL_STATS = pickle.load(f)

ALL_STATS_MU = ALL_STATS['mu']
ALL_STATS_PI = ALL_STATS['pi']

for n, width in enumerate(widths):
    width_stats = []
    x = []
    for epoch in range(1, 31):
        kernel_stats = ALL_STATS_MU[width][epoch]

        best_acc = 0
        for kernel_reg in [10**(-n) for n in range(6,0,-1)]:
            if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
                best_acc = kernel_stats[(width, epoch, kernel_reg)]
        width_stats.append(best_acc)
        x.append(epoch)
    
    plt.subplot(1,1,1)
    plt.plot(x, width_stats, label=str(width) + " mu")
    plt.legend()
    plt.ylabel('Test Kernel Accuracy')
    plt.xlabel('Train Epochs')


for n, width in enumerate(widths):
    width_stats = []
    x = []
    for epoch in range(1, 31):
        kernel_stats = ALL_STATS_PI[width][epoch]

        best_acc = 0
        for kernel_reg in [10**(-n) for n in range(6,0,-1)]:
            if kernel_stats[(width, epoch, kernel_reg)] > best_acc:
                best_acc = kernel_stats[(width, epoch, kernel_reg)]
        width_stats.append(best_acc)
        x.append(epoch)
    
    plt.subplot(1,1,1)
    plt.plot(x, width_stats, label=str(width) + " pi")
    plt.legend()
    plt.ylabel('Test Kernel Accuracy')
    plt.xlabel('Train Epochs')
# plt.savefig("kernel_test.png", bbox_inches='tight')


plt.show()