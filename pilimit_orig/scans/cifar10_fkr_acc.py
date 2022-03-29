
import os
from numpy.core.arrayprint import printoptions

from numpy.core.numeric import full
from cox.store import Store
import shutil
import subprocess
from cox.readers import CollectionReader
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
sns.set()


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
    df = pd.concat(list_of_dfs)
    return pd.concat(list_of_dfs)


# ''' fkr testing
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/rational-dove/**/') 
# print(bigdf_1)
# bigdf_1.to_pickle("cifar10-rational-dove.df")

full_df = pd.read_pickle("cifar10-rational-dove.df").reset_index()
full_df = full_df.drop_duplicates()
full_df['ker_acc'] = full_df['ker_acc'].astype(np.float32)
full_df['ker_acc'] = full_df['ker_acc'] * 100



inf_df = full_df
# inf_df = pd.read_pickle("cifar10-quick-penguin.df").reset_index()
# inf_df = inf_df.drop_duplicates()
# inf_df = inf_df[inf_df["test_subset_size"] == 500]
# inf_df = inf_df[inf_df["test_subset_size"] == 100]
inf_df = inf_df[inf_df["test_subset_size"] == 400]
inf_df = inf_df[inf_df["width"] == 0]
inf_df['ker_acc'] = inf_df['ker_acc'].astype(np.float32)
inf_mean = inf_df['ker_acc'].mean()
x = full_df.width.unique()
inf_line = [inf_mean]*len(x)


# full_df = full_df[full_df["test_subset_size"] == 100]
full_df = full_df[full_df["test_subset_size"] == 400]
full_df = full_df[full_df["width"] > 900]
full_df.loc[full_df['test_subset_size'] ==400, 'test_subset_size'] = "empirical"
# full_df.loc[full_df['width'] ==0, 'width'] = 50000
g2 = sns.lineplot(x='width', y='ker_acc', hue='test_subset_size', data=full_df, palette=['b'])


# g2 = sns.lineplot(x='width', y='ker_acc', hue='test_subset_size', data=full_df)
sns.lineplot(x, inf_line, label="$\pi$-Limit", linestyle="dashed")
plt.xlabel("Width")
plt.ylabel("Accuracy")
plt.show()
# plt.savefig("fkr plot.pdf", bbox_inches='tight')
0/0
# '''