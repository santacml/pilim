#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create a graph comparing MAML test accuracy vs width for pinet, munet, pilimit, NTK, and GP.

    The GP and NTK results are still in their cox stores and not a df.
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


widths = [128, 256, 512, 1024, 2048]
df = pd.read_pickle("maml-gp-all.df")
gp_perf = df.val_accuracies_after.max()
df = pd.read_pickle("maml-ntk-all.df")
ntk_perf = df.val_accuracies_after.max()
mu_perf = []

df = pd.read_pickle("maml-test-all.df")
infpi_perf = list(df[((df["Net"] == "InfPiNet") & (df["R"] == 400))]["test_accuracies_after"])         # get best infpinet perf
df = df[df["Width"] > 0]
df = df.drop(df[((df["Net"] == "PiNet") & (df["R"] != 400))].index)          # only keep pinet r=400

for width in widths:
    for perf in infpi_perf:
        df = df.append({'Net': 'InfPiNet', 'Width': width, 'test_accuracies_after': perf}, ignore_index=True)
    df = df.append({'Net': 'NTK', 'Width': width, 'test_accuracies_after': ntk_perf}, ignore_index=True)
    df = df.append({'Net': 'GP', 'Width': width, 'test_accuracies_after': gp_perf}, ignore_index=True)

g = sns.lineplot(x='Width', y='test_accuracies_after', hue='Net', data=df, ci="sd")


# https://stackoverflow.com/questions/52278350/seaborn-dashed-line-not-dashed-in-the-legend
# g.lines[1].set_linestyle(":")
g.lines[2].set_linestyle("--")
g.lines[3].set_linestyle("--")
g.lines[4].set_linestyle("--")
leg = g.legend(loc="center left")
leg_lines = leg.get_lines()
# leg_lines[1].set_linestyle(":")
leg_lines[2].set_linestyle("--")
leg_lines[3].set_linestyle("--")
leg_lines[4].set_linestyle("--")
plt.ylabel('Test Accuracy')
plt.xlabel('Width')
plt.subplots_adjust(wspace=0, hspace=0)



# df_orig = pd.read_pickle("maml-pinet-munet-best-width.df")
df_orig = pd.read_pickle("maml-test-all.df")
df_orig = df_orig[df_orig["Width"] > 0]
df_orig = df_orig.drop(df_orig[((df_orig["Net"] == "PiNet") & (df_orig["R"] != 400))].index)
for width in widths:
    for perf in infpi_perf:
        df_orig = df_orig.append({'Net': 'InfPiNet', 'Width': width, 'test_accuracies_after': perf}, ignore_index=True)
# axins = g.inset_axes([1500, 0.6, 1800, 0.85])
axins = g.inset_axes([0.45, 0.3, 0.5, 0.5])
g2 = sns.lineplot(x='Width', y='test_accuracies_after', hue='Net', data=df_orig, ax=axins)
g2.spines['bottom'].set_color('0')
g2.spines['top'].set_color('0')
g2.spines['right'].set_color('0')
g2.spines['left'].set_color('0')
g2.lines[2].set_linestyle("--")
# print(g2.get_lines())
# 0/0
# g2.set(ylabel=None)
g2.set(ylabel=None)
# g2.get_yaxis().set_visible(False)
g2.get_xaxis().set_visible(False)
g2.legend().set_visible(False)
# g.indicate_inset_zoom(axins, edgecolor="black")
plt.show()

