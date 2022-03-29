#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a heatmap comparing width vs r vs performance for pinet on MAML.
"""
import os
from cox.store import Store
import shutil
import subprocess
from cox.readers import CollectionReader
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


new_df = pd.read_pickle("maml-test-all.df").reset_index(drop=True)

new_df = new_df.groupby(["Net", "R", "Width"])['test_accuracies_after'].mean().reset_index()

new_df.R.replace(-1, "MuNet", inplace=True)
new_df['Width'].replace(-1, "Infinite", inplace=True)

new_df['test_accuracies_after']  *= 100
print(new_df[['R', 'Width', 'test_accuracies_after']])
new_df = new_df.pivot("R", "Width", "test_accuracies_after")
ax = sns.heatmap(new_df, annot=True, fmt=".2f", cmap='magma')
ax.invert_yaxis()
plt.ylabel('r')

yticks = ax.get_yticks()
yticks[-1] = 4.75
ylabels = ax.get_yticklabels()
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
# plt.xlabel('Width')

# https://stackoverflow.com/questions/62773944/insert-line-in-heatmap-after-every-7th-column
b, t = plt.xlim()
ax.hlines(y = 4, xmin = b-1, xmax = t, colors = 'white', lw = 5)
plt.show()


