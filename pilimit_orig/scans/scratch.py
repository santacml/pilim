#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" For random testing.
"""
import os
from matplotlib import colors
from numpy.core.arrayprint import printoptions

from numpy.core.numeric import full
# from cox.store import Store
import shutil
import subprocess
# from cox.readers import CollectionReader
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
sns.set()

def load_all_pkl(folder):
    # pkl_paths = glob.glob(os.path.join(folder, "*transfer_log.df"), recursive=True)
    pkl_paths = glob.glob(os.path.join(folder, "*log.df"), recursive=True)
    print(len(pkl_paths))
    list_of_dfs = []
    for path in pkl_paths:
        try:
            df = pd.read_pickle(path)
        except:
            continue
        list_of_dfs.append(df)
    return pd.concat(list_of_dfs)

bigdf_1 = load_all_pkl(f'/home/azureuser/amlt/endless-macaw/**/')
bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/azureuser/amlt/close-sloth/**/'))
bigdf_1 = bigdf_1.drop_duplicates()



# bigdf_1 = load_all_pkl(f'/home/azureuser/amlt/talented-doe/**/')
bigdf_1 = load_all_pkl(f'/home/azureuser/amlt/close-sloth/**/')
bigdf_1 = bigdf_1.drop_duplicates()
bigdf_1["test_acc"] *= 100
test = bigdf_1.groupby(["seed"])['test_acc'].max().reset_index()
acclist = test['test_acc'].tolist()
print("var", test['test_acc'].var())
print("mean", test['test_acc'].mean())
print("max", test['test_acc'].max())


bigdf_1 = load_all_pkl(f'/home/azureuser/amlt/endless-macaw/**/')
bigdf_1 = bigdf_1.drop_duplicates()
bigdf_1["test_acc"] *= 100
test = bigdf_1.groupby(["seed"])['test_acc'].max().reset_index()
acclist = test['test_acc'].tolist()
print("var", test['test_acc'].var())
print("mean", test['test_acc'].mean())
print("max", test['test_acc'].max())
0/0



# df = load_all_pkl(f'/home/misantac/amlt/sincere-firefly/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/moved-deer/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/curious-pug/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/nearby-hare/**/') 
# print(df["ker_acc"].max())

# pd.set_option('display.max_colwidth', None)
# df = load_all_pkl(f'/home/misantac/amlt/sincere-firefly/**/') 
# df["ker_acc"] = df["ker_acc"].astype(float)
# df = df[df["ker_acc"] == df["ker_acc"].max()]
# print(df[["ker_acc", "load_model_path"]])

# pd.set_option('display.max_colwidth', None)
# df = load_all_pkl(f'/home/misantac/amlt/optimal-dinosaur/**/') 
# df["ker_acc"] = df["ker_acc"].astype(float)
# df = df[df["ker_acc"] == df["ker_acc"].max()]
# print(df[["ker_acc", "load_model_path"]])

# df = load_all_pkl(f'/home/misantac/amlt/crisp-dassie/**/') 
# df["ker_acc"] = df["ker_acc"].astype(float)
# df = df[df["ker_acc"] == df["ker_acc"].max()]
# print(df[["ker_acc", "load_model_path"]])

# df = load_all_pkl(f'/home/misantac/amlt/subtle-silkworm/**/') 
# df["ker_acc"] = df["ker_acc"].astype(float)
# df = df[df["ker_acc"] == df["ker_acc"].max()]
# print(df[["ker_acc", "load_model_path"]])
# 0/0


# df = load_all_pkl(f'/home/misantac/amlt/closing-minnow/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/working-badger/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/firm-dingo/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/proper-wolf/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/meet-wallaby/**/') 
# df = load_all_pkl(f'/home/misantac/amlt/relevant-sawfly/**/') 
df = load_all_pkl(f'/home/misantac/amlt/handy-worm/**/') 
# print(df.keys())
# print(df["train_acc"].max())
# print(df["test_acc"].max())
# print(df["kernel_acc"].max())
# 0/0

df["kernel_acc"] = df["kernel_acc"].astype(float)
print(df["kernel_acc"].max())
df = df.sort_values("kernel_acc", ascending=False)
# df = df.sort_values("kernel_acc", ascending=False).groupby(["width", "lr", "first_layer_lr_mult", "last_layer_lr_mult", "gclip", "wd", "bias_alpha"], as_index=False).first()
print(df[["kernel_acc", "test_acc", "epoch", "width", "lr", "batch_size", "first_layer_lr_mult", "last_layer_lr_mult", "gclip", "wd", "bias_alpha"]].head(15))
# print(df[["kernel_acc", "test_acc", "epoch", "width", "lr", "first_layer_lr_mult", "last_layer_lr_mult", "gclip", "wd", "bias_alpha"]].head(15))
0/0

'''  ntk/gp Simple stats 
ntk_df = pd.read_pickle("cifar10-ntk-all.df").reset_index()
gp_df = pd.read_pickle("cifar10-gp-all.df").reset_index()

ntk_df = ntk_df[ntk_df["varb"] == 1]
ntk_df = ntk_df[ntk_df["depth"] == 2]
ntk_df = ntk_df[ntk_df["bias_lr_mult"] == 1]
ntk_df = ntk_df[ntk_df["first_layer_lr_mult"] == 1]
ntk_df = ntk_df[ntk_df["last_layer_lr_mult"] == 1]
print(ntk_df["ker_acc"].max())
print(ntk_df)

gp_df = gp_df[gp_df["varb"] == 1]
gp_df = gp_df[gp_df["depth"] == 2]
print(gp_df["ker_acc"].max())
print(gp_df)
0/0
# '''



''' fkr testing locally
    # for reg in [10**(-n) for n in range(7,2,-1)]:
# with open('test_kernels.sh', 'w') as f:
    # for reg in [1e-4]:
    #     for epoch in range(0, 51, 2):
    #         for width in [500, 1000, 5000, 10000, 20000, 30000, 40000, 0]:
    #                 f.write(f"python -m cifar10.cifar10test --cuda --kernel-reg={reg} --width {width} --train-subset-size 500 --test-subset-size 500 --test-kernel --load-model-path='/home/misantac/teamdrive/inflimitmsr/kerneltest_full/{width}/checkpoints/epoch{epoch}.th' --save-dir './temp_test/{width}/{epoch}/{reg}' --batch-size 32 --test-batch-size 32\n")
#                   0/0



with open('test_kernels_init.sh', 'w') as f:
    # f.write(f"rm -rf temp_test/")
    # for reg in [1e-7]:
    # for reg in [1e-6]:
    for dataset in [100, 500, 700, 1000, 5000, 10000]:
        for seed in range(10):
            # for reg in [1e-4]:
            for reg in [1e-6]:
                for width in [50, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 0]:
                        f.write(f"echo {width} {reg}\n")
                        f.write(f"python -m cifar10.cifar10test --float --cuda --seed={seed} --kernel-reg={reg} --width {width} --train-subset-size {dataset} --test-subset-size {dataset} --test-kernel --save-dir './temp_test/init/{width}/{reg}' --batch-size 32 --test-batch-size 32\n")
                        # f.write(f"python -m cifar10.cifar10test --cuda --kernel-reg={reg} --width {width} --train-subset-size 1500 --test-subset-size 1500 --test-kernel --save-dir './temp_test/init/{width}/{reg}' --batch-size 32 --test-batch-size 32\n")
0/0

# '''


# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/next-lizard/**/') 
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/premium-giraffe/**/') 

# params = ["width", "r", "lr", "batch_size", "lr_drop_milestones", "first_layer_lr_mult", "last_layer_lr_mult", "wd", "gclip", "bias_alpha", "first_layer_alpha", "last_layer_alpha"]
# print(bigdf_1.groupby(["width", "r"])["test_acc"].max())
# grouped = bigdf_1.groupby(params)
# largest = grouped['test_acc'].max().nlargest(5)
# print(largest.reset_index()[["test_acc"] + params])
# 0/0

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/evolving-gelding/**/') 
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/humble-cardinal/**/') 
# print(bigdf_1["r"].unique())
# bigdf_1 = bigdf_1[bigdf_1["r"] == 2049]
# bigdf_1 = bigdf_1.groupby(["seed"])["test_acc"].max().reset_index()
# bigdf_1["test_acc"] *= 100
# print(bigdf_1["test_acc"].std())
# print(bigdf_1["test_acc"].mean())
# print(bigdf_1["test_acc"])
# 0/0


# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/workable-reptile/**/') 
# print(bigdf_1)
# bigdf_1.to_pickle("cifar10-workable-reptile.df")
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/quick-penguin/**/') 
# print(bigdf_1)
# bigdf_1.to_pickle("cifar10-quick-penguin.df")
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/rational-dove/**/') 
# print(bigdf_1)
# bigdf_1.to_pickle("cifar10-rational-dove.df")
# 0/0


# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/oriented-amoeba/**/') 
# print(bigdf_1["kernel_acc"].max())
# 0/0
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/sweet-cattle/**/') 
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/moral-loon/**/') )   #orig
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/holy-monkey/**/') )  #refactored
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/guiding-redbird/**/') )  #refactored x2
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/pleased-seasnail/**/') )
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/casual-gannet/**/') )

# bigdf_1.to_pickle("cifar10-prune-tests.df")
# 0/0

df = pd.read_pickle("cifar10-prune-tests.df").reset_index()
# print(df)
# lsh = df[df["folder"] == "/home/misantac/amlt/moral-loon/**/"]
# lsh = df[df["folder"] == "/home/misantac/amlt/holy-monkey/**/"]
# lsh = df[df["folder"] == "/home/misantac/amlt/guiding-redbird/**/"]
lsh = df[df["folder"] == "/home/misantac/amlt/casual-gannet/**/"]
# lsh["label"] = lsh["prune_tol"].astype(str) + lsh["prune_lsh_rows"].astype(str)
# lsh["label"] = lsh[['prune_tol', 'prune_lsh_rows']].apply(lambda x: ' '.join(x), axis=1)

# lsh = lsh[lsh["prune_tol"] == 0.85]
# lsh = lsh[lsh["prune_lsh_rows"] < 20]
# lsh = lsh[lsh["prune_tol"] > 0]
# print(lsh["prune_lsh_rows"].unique())
# 0/0


cos = df[df["folder"] == "/home/misantac/amlt/sweet-cattle/**/"]

# cos_once = df[df["folder"] == "/home/misantac/amlt/adapted-duckling/**/"]
benchmark = df[df["folder"] == "/home/misantac/amlt/pleased-seasnail/**/"]
benchmark["prune_tol"] = 0
pi_limit = benchmark["test_acc"].max()
print("best", pi_limit)
# lsh = lsh.append(benchmark)
# cos = cos.append(benchmark)

# print(cos["prune_tol"].unique())
# 0/0

# fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2)
fig, (ax1, ax2) = plt.subplots(1, 2)

# lsh = lsh.groupby(["prune_tol", "prune_lsh_rows"], as_index=False).agg({'test_acc':['max'],'epoch_time':['mean'],'pruned':['sum']}).reset_index()
lsh = lsh.groupby(["prune_tol", "prune_lsh_rows"], as_index=False).agg(test_acc=('test_acc', 'max'), 
                                                     epoch_time=('epoch_time', 'mean'), 
                                                     pruned=('pruned', 'mean'), 
                                                     prune_tol=('prune_tol', 'mean'), 
                                                     prune_lsh_rows=('prune_lsh_rows', 'mean'))
# lsh["label"] = lsh["prune_lsh_rows"].astype(str) + lsh["prune_tol"].astype(str)
printme = lsh[lsh["pruned"] > .15]
printme = printme[printme["test_acc"] > .59]
print(printme)
# cos = cos.groupby(["prune_tol"], as_index=False).agg({'test_acc':['max'],'epoch_time':['mean'],'pruned':['sum']}).reset_index()
cos = cos.groupby(["prune_tol"], as_index=False).agg(test_acc=('test_acc', 'max'), 
                                                     epoch_time=('epoch_time', 'mean'), 
                                                     pruned=('pruned', 'mean'))

# lsh.columns = ["index", "prune_tol", "prune_lsh_rows", "test_acc", "epoch_time", "pruned"]
# cos.columns = ["index", "prune_tol", "test_acc", "epoch_time", "pruned"]
# print(cos.columns)
# print(lsh)
# 0/0


b_x = [0, 1]
b_y = [pi_limit] * len(b_x)

t_x = [0, lsh["pruned"].max()]
t_y = [benchmark["epoch_time"].mean()] * len(b_x)

cos["exp"] = "cos"
lsh["exp"] = "lsh"

total = lsh.append(cos)

# ax1, ax2, ax3, ax4, ax5, ax6 = row1, row2
g = sns.lineplot(x='pruned', y='test_acc', data=total, hue="exp", ax=ax1, palette="pastel")
g = sns.lineplot(x=b_x, y=b_y, ax=ax1, linestyle="dashed")
ax1.title.set_text("exp pruned vs test acc")
g = sns.lineplot(x='pruned', y='epoch_time',  data=total, hue="exp", ax=ax2, palette="pastel")
g = sns.lineplot(x=t_x, y=t_y, ax=ax2, linestyle="dashed")
ax2.title.set_text("exp pruned vs epoch time")
# g.xlabel("Width")
# g.ylabel("Accuracy")

# g = sns.lineplot(x='pruned', y='test_acc',data=cos, ax=ax2)
# ax2.title.set_text("cos pruned vs test acc")

# g = sns.lineplot(x='pruned', y='test_acc',data=cos, ax=ax4)
# ax4.title.set_text("cos pruned vs test acc")
# g = sns.lineplot(x='pruned', y='epoch_time', data=cos, ax=ax5)
# ax5.title.set_text("cos pruned vs epoch time")

# g = sns.lineplot(x='epoch', y='test_acc', hue="prune_tol", data=cos_once, ax=ax3)
# g = sns.lineplot(x=b_x, y=b_y, ax=ax3, linestyle="dashed")
# g = sns.lineplot(x='epoch', y='pruned', hue="prune_tol", data=cos_once, ax=ax6)

plt.tight_layout()
plt.show()

0/0



'''
# fig, (ax1, ax2) = plt.subplots(1, 2)
# /$seed/$dataset/$width

# full_df = pd.read_pickle("cifar10-quick-penguin-dists.df").reset_index()
full_df = pd.read_pickle("cifar10-workable-reptile-dists.df").reset_index()
# full_df["dist"] = full_df["dist"] / full_df["dist"].max()
# widths = np.array([50, 100, 500, 1000, 5000, 10000, 15000, 20000, 30000, 40000])
# theory = widths**(-.5)

# full_df = full_df[full_df["seed"] == "0"]
full_df = full_df[full_df["dataset"] == "500"]
full_df.loc[full_df["dataset"] == "500", 'dataset'] = "empirical deviation"

full_df['width'] = full_df['width'].astype(int)
# full_df = full_df.sort_values(by=['width'], ascending=True)

# print(full_df[full_df["dataset"] == "500"])
# print(full_df[full_df["dataset"] == "1000"])
# 0/0



# f, ax = plt.subplots(figsize=(7, 7))
# ax.set(xscale="log", yscale="log")
# sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})

# g2 = sns.lineplot(x='width', y='dist', hue='dataset', data=full_df, ax=ax)
# print(full_df["width"])
# 0/0
# g = sns.lineplot(widths,theory, ax=ax)
# plt.xlabel("log_2 width")
# plt.ylabel("log_2 dist")
# g2.set(xscale="log", yscale="log")
# g2.set_xscale("log", base=2)
# g2.set_yscale("log", base=2)
# g.set(xscale="log", yscale="log")
# g.set_xscale("log", base=2)
# g.set_yscale("log", base=2)
g = sns.lineplot(data=full_df, x='width', y='dist', hue='dataset')
# widths = [64, 128, 256, 512, 1024, 2048, 4092, 8192, 16384, 32768]
widths = full_df.width.unique()
plt.plot(widths, np.array(widths, dtype='float')**-0.5, '--', label=u'$1/\sqrt{width}$')
# print(widths,  np.array(widths, dtype='float')**-0.5)

# xx = 2**np.linspace(1, 16); plt.plot(xx, xx**-0.5, '--', label=u'${width}^{-.5}$')
# print(xx, xx**-0.5)
# 0/0
# plt.plot(full_df["width"], full_df["dist"])
# plt.ylabel(u'Relative Frob. Norm')
# plt.title("Inf to Fin Kernel Frobenius Norm per Dataset Subset Size")
plt.xlabel("Width")
# plt.ylabel("Feature Kernel Deviation from Limit in Frob. Norm")
plt.ylabel("Frob. Norm Distance")
plt.loglog(base=2)
plt.legend()
# g.set_xscale("log", base=2)
# g.set_yscale("log", base=2)
# plt.set(xscale="log", yscale="log")

plt.show()
# plt.savefig("kernel reg dist.pdf", bbox_inches='tight')
0/0
# '''




'''  getting frob distance graph



inf_dirs = {}
dirs = os.listdir("/home/misantac/amlt/quick-penguin/")
for dir in dirs:
    seeddir =  os.listdir(os.path.join("/home/misantac/amlt/quick-penguin/", dir))[0]
    # print(os.path.join("/home/misantac/amlt/workable-reptile/", dir, seeddir))
    inf_dirs[seeddir] = os.path.join("/home/misantac/amlt/quick-penguin/", dir, seeddir)

new_df = []
# dirs = os.listdir("/home/misantac/amlt/quick-penguin/")
dirs = os.listdir("/home/misantac/amlt/workable-reptile/")
for dir in dirs:
    seeddir =  os.listdir(os.path.join("/home/misantac/amlt/workable-reptile/", dir))[0]
    print(os.path.join("/home/misantac/amlt/workable-reptile/", dir, seeddir))
    
    for dataset in [500, 1000]:
    # for dataset in [500, 1000]:
        dataset = str(dataset)
        # inf_path = f"/home/misantac/amlt/quick-penguin/{seed}/{dataset}/0/ker.th"
        # try:
        # inf_path = os.path.join("/home/misantac/amlt/workable-reptile/", dir, seeddir, dataset, "0", "ker.th")
        # inf_path = os.path.join("/home/misantac/amlt/quick-penguin/", dir, seeddir, dataset, "0", "ker.th")
        inf_path = os.path.join(inf_dirs[seeddir], dataset, "0", "ker.th")
        inf_kernel = torch.load(inf_path)
        # except:
        #     print(seeddir, dataset, 0, "failed")
        # for width in [50, 100, 500, 1000, 5000, 10000, 15000, 20000, 30000, 40000]:
        for width in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
            width = str(width)
            # try:
                # kernel_path =  f"/home/misantac/amlt/quick-penguin/{seed}/{dataset}/{width}/ker.th"
            kernel_path =  os.path.join("/home/misantac/amlt/workable-reptile/", dir, seeddir, dataset, width, "ker.th")
            kernel = torch.load(kernel_path)

            dist = np.linalg.norm(inf_kernel - kernel) / np.linalg.norm(inf_kernel)


            new_df.append({"seed": seeddir, "dataset": dataset, "dist": dist, "width": width})
            # except:
            #     print(seeddir, dataset, width, "failed")
# print(dirs)
new_df = pd.DataFrame(new_df)
new_df.to_pickle("cifar10-workable-reptile-dists.df")
print(new_df)
0/0
0/0

'''




''' fkr testing
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/sterling-aardvark/**/**/**/') 
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/giving-chigger/**/') 
# print(bigdf_1)
# bigdf_1.to_pickle("cifar10-giving-chigger.df")
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/live-hen/**/') 
# print(bigdf_1)
# bigdf_1.to_pickle("cifar10-live-hen.df")
# 0/0


# full_df = pd.read_pickle("cifar10-scan-kernels.df").reset_index()
# full_df = pd.read_pickle("cifar10-live-hen.df").reset_index()
# full_df = pd.read_pickle("cifar10-giving-chigger.df").reset_index()
# full_df = pd.read_pickle("cifar10-workable-reptile.df").reset_index()
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














''' fkr gbest plot for 500

full_df = pd.read_pickle("cifar10-workable-reptile.df").reset_index()
full_df = full_df.drop_duplicates()
full_df['ker_acc'] = full_df['ker_acc'].astype(np.float32)



inf_df = pd.read_pickle("cifar10-quick-penguin.df").reset_index()
inf_df = inf_df.drop_duplicates()
inf_df = inf_df[inf_df["test_subset_size"] == 500]
inf_df = inf_df[inf_df["width"] == 0]
inf_df['ker_acc'] = inf_df['ker_acc'].astype(np.float32)
inf_mean = inf_df['ker_acc'].mean()
x = full_df.width.unique()
inf_line = [inf_mean]*len(x)


# full_df = full_df[full_df["test_subset_size"] == 500]
# full_df.loc[full_df['test_subset_size'] ==500, 'test_subset_size'] = "empirical"
# full_df.loc[full_df['width'] ==0, 'width'] = 50000
# g2 = sns.lineplot(x='width', y='ker_acc', hue='test_subset_size', data=full_df, palette=['b'])


g2 = sns.lineplot(x='width', y='ker_acc', hue='test_subset_size', data=full_df)
sns.lineplot(x, inf_line, label="$\pi$-Limit", linestyle="dashed")
plt.xlabel("Width")
plt.ylabel("Feature Kernel Regression")
plt.show()
0/0
# '''













# full_df = full_df[full_df['test_subset_size'] == 500]

# print(len(full_df[full_df['width'] == 0][['test_subset_size', 'ker_acc']]))
# 0/0

# print(full_df.keys())
# full_df['Width'] = full_df['Width'].astype(str) 

# full_df['dist'] = np.linalg.norm(full_df['ker_acc'] - inf_ker_mean)


# full_df = full_df[full_df['test_subset_size'] == 1000]
# inf_ker_mean = full_df[full_df['width'] == 0]['ker_acc'].mean()
# full_df = full_df[full_df['width'] != 0]
# full_df['dist'] = (full_df['ker_acc']**2 - inf_ker_mean**2)**.5

# widths = np.array([50, 100, 500, 1000, 5000, 10000, 15000, 20000, 30000, 40000])
# theory = widths**(-.5)
# g2 = sns.lineplot(x='width', y='dist', hue='test_subset_size', data=full_df)
# g2.set(xscale="log", yscale="log")
# plt.plot(widths,theory)
# plt.xlabel("log_2 width")
# plt.ylabel("log_2 dist")
# g2.set_xscale("log", base=2)
# g2.set_yscale("log", base=2)


# plt.show()
# 0/0




# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/more-minnow/**/') 
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/star-thrush/**/') 
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/engaging-mudfish/**/') 
# bigdf_1 = bigdf_1[bigdf_1['depth'] == 2]
# max_ntk = bigdf_1.ker_acc.max()  
# print(max_ntk)
# bigdf_1['ker_acc'].nlargest(15)
# 0/0




# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/enough-bluejay/**/') 
# bigdf_1 = load_all_pkl(f'/home/misantac/pilimit/temp_test/init/**/')
# print(bigdf_1)
# bigdf_1 = bigdf_1.sort_values(by=['width'], ascending=True)
# print(bigdf_1[["width", "ker_acc"]])
# 0/0
# bigdf_1.to_pickle("cifar10-500-kernels.df")
# 0/0


# full_df = bigdf_1

''' mu/fin kernels in very broken format
# full_df = pd.read_pickle("cifar10-500-kernels.df").reset_index()
# print(full_df)
# 0/0
new_df = []
for width in [500, 1000, 5000, 10000, 20000, 30000, 40000, 0]:
    print("width", width)
    df = full_df
    ker_accs = {}
    inf_accs = {}
    for index, row in df.iterrows():
        path = row["load_model_path"]
        row_width = row["width"]
        if row_width != width: continue
        # if row["kernel_reg"] != 1e-4: continue
        # print(row["kernel_reg"])

        # if row_model != model: continue
        # print(path.split("/")[-4])
        # 0/0
        # row_width = path.split("/")[-3]
        # df.at[index, "epoch"] = 0
        # print(int(file[5:-3]))
        file = path.split("/")[-1]
        df.at[index, "epoch"] = int(file[5:-3])

        if isinstance(df.at[index, "ker_acc"], torch.Tensor):
            # epoch = 0
            epoch = int(file[5:-3])
            if epoch in ker_accs and df.at[index, "ker_acc"].item() < ker_accs[epoch]: continue
            ker_accs[epoch] = df.at[index, "ker_acc"].item()
        print(index)
        if not np.isnan(df.at[index, "inf_test_acc"]):
            epoch = int(file[5:-3])
            inf_accs[epoch] = df.at[index, "inf_test_acc"]
    for epoch in range(0, 1):
        if not (epoch in ker_accs) or not (epoch in inf_accs): continue
        new_df.append({"epoch": epoch, "Width": width, "ker_acc": ker_accs[epoch], "inf_test_acc": inf_accs[epoch]})
df = pd.DataFrame(new_df)
# df.to_pickle("cifar10-1k-500-processed.df")
print(df)
# print(df[df["epoch"] == 0])
0/0
# '''













# # '''get fin kernel stats
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/fast-polliwog/**/')   
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/fitting-walrus/**/'))
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/destined-porpoise/**/'))
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/winning-mole/**/'))


# bigdf_1.to_pickle("cifar10-11.11-kernels.df")
# 0/0
# '''



''' mu/fin kernels in very broken format
full_df = pd.read_pickle("cifar10-11.11-kernels.df").reset_index()
# print(full_df)
# 0/0
new_df = []
# for model in ["PiNet", "MuNet"]:
for width in [500, 1000, 5000, 10000, 20000, 30000, 40000, 0]:
    print("width", width)
    df = full_df
    ker_accs = {}
    inf_accs = {}
    for index, row in df.iterrows():
        path = row["load_model_path"]
        # row_model = "PiNet"
        # if path.split("/")[-4] in ["7368690537.25205-af6b1843-8f1d-41a7-9015-e13fb6df3b1a", "7367558822.88152-23dd1c42-6980-4f7f-b084-919859a3947c"]:
        #     row_model = "MuNet"
        row_width = row["width"]
        if row_width != width: continue

        # if row_model != model: continue
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
        new_df.append({"epoch": epoch, "Width": width, "ker_acc": ker_accs[epoch], "inf_test_acc": inf_accs[epoch]})
df = pd.DataFrame(new_df)
df.to_pickle("cifar10-11.11-kernels-processed.df")
print(df)
# print(df[df["epoch"] == 0])
0/0
# '''


# full_df = pd.read_pickle("cifar10-11.11-kernels-processed.df").reset_index()
# full_df['Width'] = full_df['Width'].astype(str)
# g2 = sns.lineplot(x='epoch', y='inf_test_acc', hue='Width', data=full_df)

# full_df = full_df[full_df["Width"] == 5000]
# print(full_df)

# x = range(0,51,1)
# for width in [500, 1000, 5000, 10000, 20000, 30000, 40000, 0]:
#     print(width)
#     stats = list(full_df[full_df["Width"] == width].sort_values(by=['epoch'])["ker_acc"])

#     plt.subplot(1,1,1)
#     plt.plot(x, stats, label=width, linestyle="solid")
# plt.legend()
# plt.ylabel('Test Kernel Accuracy')
# plt.xlabel('Train Epochs')

# plt.savefig("kernel_test.png", bbox_inches='tight')


# plt.show()


# pd.set_option('display.max_colwidth', None)



# df = pd.read_pickle("cifar10-inf-all.df")
# df = df[df["epoch"] > 70]





# '''    # testing 11/11
def load_all_pkl(folder):
    # pkl_paths = glob.glob(os.path.join(folder, "*transfer_log.df"), recursive=True)
    pkl_paths = glob.glob(os.path.join(folder, "*log.df"), recursive=True)
    print(len(pkl_paths))
    list_of_dfs = []
    for path in pkl_paths:
        try:
            df = pd.read_pickle(path)
        except:
            continue
        list_of_dfs.append(df)
    return pd.concat(list_of_dfs)



# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/noble-redfish/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/destined-marlin/**/')
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/glowing-cougar/**/')
bigdf_1 = bigdf_1.drop_duplicates()
bigdf_1["test_acc"] *= 100
test = bigdf_1.groupby(["seed"])['test_acc'].max().reset_index()
# print(test)
acclist = test['test_acc'].tolist()
print("var", test['test_acc'].var())
print("mean", test['test_acc'].mean())
0/0




params = [ "lr", "batch_size","wd", "lr_drop_milestones"]
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/modest-fowl/**/')   #55.84    infnet
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/creative-javelin/**/')   #53.74 pifinnet
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/huge-pelican/**/')   #54.24 munet

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/main-collie/**/') #infnet simple
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/beloved-doe/**/') #finnet simple
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/desired-asp/**/') #munet simple
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/settled-loon/**/') #infnet simpler
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/liberal-foxhound/**/') #finnet simpler
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/evident-sponge/**/') #munet simpler

bigdf_1 = bigdf_1[bigdf_1["depth"] == 2]
# bigdf_1 = bigdf_1[bigdf_1["lr_drop_milestones"] == 30]
# bigdf_1 = bigdf_1[bigdf_1["batch_size"] == 64]

print(bigdf_1['test_acc'].max())

grouped = bigdf_1.groupby(params)
largest = grouped['test_acc'].max().nlargest(5)
print(largest.reset_index()[["test_acc"] + params])

0/0
# '''





''' #timing scans


def load_all_pkl_maml(folder):
    pkl_paths = glob.glob(os.path.join(folder, "*"), recursive=False)
    print(len(pkl_paths))
    list_of_dfs = []
    for path in pkl_paths:
        try:
            # df = pd.read_pickle(path)
            head, tail = os.path.split(path)
            if tail in ['store.h5', 'save', 'tensorboard']: continue   # weird structure for finpinet
            s = Store(head, tail)
            result = s['result'].df
            metadata = s['metadata'].df
            result['tmp'] = 1
            metadata['tmp'] = 1
            df = pd.merge(result, metadata, on=['tmp'])
            df = df.drop('tmp', axis=1)
            df['head'] = head
            df['tail'] = tail
            list_of_dfs.append(df)
            # print(list_of_dfs[-1])
            s.close()
        except:
            print("failed reading", path)
            continue
        # list_of_dfs.append(df)
    return pd.concat(list_of_dfs)

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/still-boar/**/') #infnet
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/sound-bass/**/') #pinet
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/boss-primate/**/') #munet
# for depth in [1,2,3,4]:
#     df = bigdf_1[bigdf_1["depth"]==depth]
#     print("depth", depth, "time", df["epoch_time"].mean(), "has nan", df["train_loss"].isnull().values.any())

#     df = df.sort_values(by=['epoch'], ascending=True)
#     print(df[["epoch", "epoch_time"]])
#     print()

#     diff = df["epoch_time"][1:].to_numpy()- df["epoch_time"][:-1].to_numpy()
#     print(diff.mean())
#     0/0



# bigdf_1 = load_all_pkl_maml(f'/home/misantac/amlt/advanced-fly/**/**/**/')  #pinet
# bigdf_1 = load_all_pkl_maml(f'/home/misantac/amlt/probable-whippet/**/**/')  #munet
bigdf_1 = load_all_pkl_maml(f'/home/misantac/amlt/exciting-dove/**/**/**/')  #infnet 1,2
# bigdf_1 = load_all_pkl_maml(f'/home/misantac/amlt/outgoing-koala/**/**/**/')  #infnet 3,4
# bigdf_1 = bigdf_1[bigdf_1["infnet_r"]==400]
bigdf_1 = bigdf_1[bigdf_1["depth"]==1]    # 10.19 extra seconds per epoch, starting at 58.65, so 58.65 + 25*10.19 is Avg
# bigdf_1 = bigdf_1[bigdf_1["depth"]==2]    # 10.19 extra seconds per epoch, starting at 58.65, so 58.65 + 25*10.19 is Avg
# bigdf_1 = bigdf_1[bigdf_1["depth"]==3]  # 20.94 extra seconds per epoch, starting at 73.48
# bigdf_1 = bigdf_1[bigdf_1["depth"]==4]   # 31.44 extra seconds per epoch, starting at 86.83
print(bigdf_1[["epoch", "train_time"]])
0/0
# for depth in [1,2,3,4]:
# for depth in [3,4]:
for depth in [1]:
    df = bigdf_1[bigdf_1["depth"]==depth]
    # df = bigdf_1[bigdf_1["infnet_r"]==400]
    # df = df[:50] # weirdly seems to have run more than 50? not sure where the 26 comes from
    print("depth", depth, "time", df["train_time"].mean(), "has nan", df["train_mean_outer_loss"].isnull().values.any())
    # print("depth", depth, "time", df["train_time"].mean())

    # df = df.sort_values(by=['epoch'], ascending=True)
    # print(df[["epoch", "train_time"]])
    # print(len(df))
    # print(df["train_time"].to_list())
    # 0/0

    diff = df["train_time"][1:].to_numpy()- df["train_time"][:-1].to_numpy()
    print(diff.mean() , len(diff))
    # 0/0

0/0

# '''





'''

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/deciding-mantis/**/')
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/witty-malamute/**/'))

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/nice-swine/**/')

# bigdf_1["kernel_acc"] = bigdf_1["kernel_acc"].astype(float)

# params = ["epoch", "width", "lr", "batch_size", "first_layer_lr_mult", "last_layer_lr_mult", "wd", "gclip", "bias_alpha"]
params = ["width", "lr", "batch_size", "first_layer_lr_mult", "last_layer_lr_mult", "wd", "gclip", "bias_alpha"]

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/rare-dolphin/**/')
# bigdf_1 = bigdf_1.append(load_all_pkl(f'/home/misantac/amlt/cheerful-chamois/**/'))
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/cheerful-chamois/**/')


# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/adequate-robin/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/sterling-cougar/**/')


# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/learning-chipmunk/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/popular-mongoose/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/premium-terrier/**/')

# bigdf_1 = bigdf_1[bigdf_1["epoch"] > 70]

# print(bigdf_1.groupby(["width"], as_index=False)['kernel_acc'].max())
# print(bigdf_1.groupby(["width"], as_index=False)[params])

# bigdf_1["kernel_acc"] = bigdf_1["kernel_acc"].astype(float)
# grouped = bigdf_1.groupby(params, as_index=False)
# grouped = bigdf_1.groupby(params)
# largest = grouped['kernel_acc'].max().nlargest(50)
# print(largest.reset_index()[["kernel_acc"] + params])




# bigdf_1["kernel_acc"] = bigdf_1["kernel_acc"].astype(float)
# print(bigdf_1["kernel_acc"].max())
# best = bigdf_1[bigdf_1["kernel_acc"] == bigdf_1["kernel_acc"].max()]
# print(best[params])



# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/new-ewe/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/destined-wildcat/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/magnetic-silkworm/**/')

# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/fine-ostrich/**/')
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/massive-toucan/**/')
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/ruling-frog/**/')

# print(bigdf_1["load_model_path"])

print(bigdf_1["ker_acc"].max())
best = bigdf_1[bigdf_1["ker_acc"] == bigdf_1["ker_acc"].max()]
# print(best["load_model_path"])
0/0
# '''




''' # create width vs r vs perf variance heatmap - under heavy testing.

# new_df = pd.read_pickle("maml-infnet-all.df").reset_index(drop=True)
new_df = pd.read_pickle("maml-pifinnet-all.df").reset_index(drop=True)
new_df = new_df.append(pd.read_pickle("maml-pifinnet-test-bignets-all.df")).reset_index(drop=True)
# new_df = pd.read_pickle("maml-test-all.df").reset_index(drop=True)

new_df = new_df[new_df["epoch"] == 50]
new_df = new_df[new_df["infnet_r"] != 2400]
print(new_df['val_accuracies_after'])
# new_df = new_df.groupby(["hidden_size", "infnet_r"])['val_accuracies_after'].count().reset_index()

# new_df['val_accuracies_after']  *= 100
# print(new_df["val_accuracies_after"])
# new_df = new_df.groupby(["hidden_size", "infnet_r"])['train_accuracies_after'].max().reset_index()
# new_df = new_df.groupby(["hidden_size", "infnet_r"])['val_accuracies_after'].max().reset_index()
new_df = new_df.groupby(["hidden_size", "infnet_r"])['val_accuracies_after'].var().reset_index()
# new_df = new_df.groupby(["hidden_size", "infnet_r"])['val_accuracies_after'].median().reset_index()
# new_df = new_df.groupby(["Net", "R", "Width"])['test_accuracies_after'].mean().reset_index()

# new_df['train_accuracies_after']  *= 100
new_df['val_accuracies_after']  *= 100
# new_df.R.replace(-1, "MuNet", inplace=True)
# new_df['Width'].replace(-1, "Infinite", inplace=True)

# print(new_df[['R', 'Width', 'test_accuracies_after']])
# new_df = new_df.pivot("infnet_r", "hidden_size", "train_accuracies_after")
new_df = new_df.pivot("infnet_r", "hidden_size", "val_accuracies_after")
ax = sns.heatmap(new_df, annot=True, fmt=".2f", cmap='magma')
ax.invert_yaxis()
plt.ylabel('r')

# yticks = ax.get_yticks()
# yticks[-1] = 4.75
# ylabels = ax.get_yticklabels()
# ax.set_yticks(yticks)
# ax.set_yticklabels(ylabels)
# plt.xlabel('Width')

# https://stackoverflow.com/questions/62773944/insert-line-in-heatmap-after-every-7th-column
b, t = plt.xlim()
# ax.hlines(y = 4, xmin = b-1, xmax = t, colors = 'white', lw = 5)
plt.show()


0/0
# '''





''' random testing
# df = pd.read_pickle("maml-infnet-all.df")
df = pd.read_pickle("maml-pifinnet-all.df")
# df = pd.read_pickle("maml-mu-all.df")

# df = df[df["infnet_r"] == 400]

grouped = df.groupby(["infnet_r", "hidden_size"], as_index=False)
largest = grouped.apply(lambda x:x.nlargest(10,'val_accuracies_after')).reset_index(drop=True)
# print(largest.groupby(["hidden_size"])['val_accuracies_after'].median())
print(largest["infnet_r"])
0/0
# re-retrieve each full group 
# df = pd.concat( [ grouped.get_group(group) for i,group in enumerate( largest.keys())])
# print(largest)
# 0/0


# print(df.groupby(["infnet_r"])['val_accuracies_after'].max())
# print(df.groupby(["hidden_size"])['val_accuracies_after'].max())
# cmd = "zip -r mu-nets.zip "
# for width in [128, 256, 512, 1024, 2048]:
#     df_t = df[df["hidden_size"] == width]
#     df_row = df_t[df_t.val_accuracies_after == df_t.val_accuracies_after.max()]
#     for ind in df_row.index:
#         cmd = cmd + df_row["head"][ind] + " "
# print(cmd)

# df_row = df[df.val_accuracies_after == df.val_accuracies_after.max()]
# for ind in df_row.index:
#     print(df_row["head"][ind])
#     print(df_row["tail"][ind])
# python -m meta.test --config '/home/misantac/amlt/master-dingo/pinet-sweep-maml_186/1024/400/5b24bfa8f11ad9e56e9d3b608d53eac33766d1f6/config.json' --use-cuda --seed 1
# python -m meta.test --config '/home/misantac/amlt/inf_maml_sweep_4real/pinet-sweep-maml_74/200/d189f89fe0d7901136f0e3c43714ddc5c8cdd154/config.json' --use-cuda --seed 1
# 0/0

# df = pd.read_pickle("maml-pinet-munet-best-width.df")

# print(df.nlargest(30, 'val_accuracies_after')[["val_accuracies_after", "exp", "hidden_size"]])
# print(df.nlargest(30, 'test_accuracies_after'))
# print(df[df['Net'] == 'MuNet'].groupby(['Width']).max())  # already filtered best per width
0/0
# '''







