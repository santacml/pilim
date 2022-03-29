

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


'''
    it's necessary to find the saved kernel for each width/dataset combo
    in the below code, inf kernel was saved separately from the fin kernels
    but this code can be modified to save a df of the dists from inf kernel to fin kernels
'''



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




# create the inf frob distance graph




# '''
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

full_df['width'] = full_df['width'].astype(int)                                   # make sure to do this otherwise it interprets strings as ints
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