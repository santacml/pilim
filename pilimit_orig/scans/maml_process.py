#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file is for processing MAML validation results, choosing the top HP combos, and creating a bash script to test them for X seeds.
    It's not necessary to rerun unless MAML results are rerun for some reason.
    Each code block has a comment describing what it does.
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

def load_all_pkl(folder):
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

seeds = list(range(20))

''' gather mu-net results
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/bold-man/**/**/')
bigdf_1['exp'] = "bold-man"
bigdf_2 = load_all_pkl(f'/home/misantac/amlt/suited-hamster/**/**/')
bigdf_2['exp'] = "suited-hamster"
bigdf_3 = load_all_pkl(f'/home/misantac/amlt/possible-aardvark/**/**/')
bigdf_3['exp'] = "possible-aardvark"
bigdf_1 = bigdf_1.append(bigdf_2).append(bigdf_3)
bigdf_1.to_pickle("maml-mu-all.df")
print(bigdf_1.groupby(['hidden_size'])['val_accuracies_after'].max())
0/0
# '''

# bigdf_1 = pd.read_pickle("maml-mu-all.df").reset_index(drop=True)
# bigdf_2 = load_all_pkl(f'/home/misantac/amlt/dynamic-magpie/**/**/')
# bigdf_2['exp'] = "dynamic-magpie"
# bigdf_1 = bigdf_1.append(bigdf_2)
# bigdf_1.to_pickle("maml-mu-all.df")
# 0/0

''' create file for munet to run
seeds = list(range(5))
bigdf_1 = pd.read_pickle("maml-mu-all.df").reset_index(drop=True)
new_df = []
for width in [128, 256, 512, 1024, 2048]:
    df_row = bigdf_1[bigdf_1['hidden_size'] == width]
    df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]
    new_df.append(df_row)
    print(width,  df_row.val_accuracies_after.max())
new_df =  pd.concat(new_df)
with open('eval_maml.sh', 'w') as f:
    for ind in new_df.index:
        store = Store(new_df['head'][ind], new_df['tail'][ind])
        for seed in seeds:
            # print(f"python -m meta.test --config '{new_df['head'][ind]}/{new_df['tail'][ind]}/config.json' --use-cuda --seed {seed}")
            f.write(f"python -m meta.test --config '{new_df['head'][ind]}/{new_df['tail'][ind]}/config.json' --use-cuda --seed {seed}\n")
0/0
# '''

''' gather all infnet results
# bigdf_1 = load_all_pkl(f'/home/misantac/amlt/intimate-porpoise/**/**/') 
# bigdf_1['exp'] = "intimate-porpoise"
# bigdf_2 = load_all_pkl(f'/home/misantac/amlt/exciting-boxer/**/**/') 
# bigdf_2['exp'] = "exciting-boxer"
# bigdf_3 = load_all_pkl(f'/home/misantac/amlt/open-mammoth/**/**/') 
# bigdf_3['exp'] = "open-mammoth"

# bigdf_4 = load_all_pkl(f'/home/misantac/amlt/inf_maml_sweep_4real/**/**/') 
# bigdf_4['exp'] = "inf_maml_sweep_4real"
# bigdf_5 = load_all_pkl(f'/home/misantac/amlt/inf_maml_sweep_4real_rr1/**/**/') 
# bigdf_5['exp'] = "inf_maml_sweep_4real_rr1"
# bigdf_6 = load_all_pkl(f'/home/misantac/amlt/inf_maml_sweep_4real_eu/**/**/') 
# bigdf_6['exp'] = "inf_maml_sweep_4real_eu"

# bigdf_1 = bigdf_1.append(bigdf_2).append(bigdf_3).append(bigdf_4).append(bigdf_5).append(bigdf_6)
# bigdf_1.to_pickle("maml-infnet-all.df")

# bigdf_1 = pd.read_pickle("maml-infnet-all.df")
# bigdf_2 = load_all_pkl(f'/home/misantac/amlt/finer-tarpon/**/**/') 
# bigdf_1['exp'] = "finer-tarpon"
# bigdf_1 = bigdf_1.append(bigdf_2)
# bigdf_1.to_pickle("maml-infnet-all.df")
0/0
# '''


''' create a bash file to be run that evaluates MAML for many seeds for each best inf net
seeds = list(range(5))
bigdf_1 = pd.read_pickle("maml-infnet-all.df").reset_index(drop=True)
new_df = []
for r in [50, 100, 200, 400]:
    df_row = bigdf_1[bigdf_1['infnet_r'] == r]
    df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]
    new_df.append(df_row)
new_df =  pd.concat(new_df)
for ind in new_df.index:
    job_name = os.path.split(os.path.split(new_df['head'][ind])[0])[1]
    print(f"amlt results {new_df['exp'][ind]} :{job_name} --no-md5")
with open('eval_maml.sh', 'w') as f:
    for ind in new_df.index:
        store = Store(new_df['head'][ind], new_df['tail'][ind])
        for seed in seeds:
            f.write(f"python -m meta.test --config '{new_df['head'][ind]}/{new_df['tail'][ind]}/config.json' --use-cuda --seed {seed}\n")
0/0
# '''

''' gather finnet results
df = pd.read_pickle("master-dingo.df")
df['exp'] = "master-dingo"
bigdf_1 = load_all_pkl(f'/home/misantac/amlt/ample-lioness/**/**/**/')  #pifinnet fine 1
bigdf_1['exp'] = "ample-lioness"
bigdf_2 = load_all_pkl(f'/home/misantac/amlt/glowing-squirrel/**/**/**/')  #pifinnet fine 1
bigdf_2['exp'] = "glowing-squirrel"
bigdf_3 = load_all_pkl(f'/home/misantac/amlt/tops-macaque/**/**/**/') #pifinnet fine 2
bigdf_3['exp'] = "tops-macaque"
bigdf_1 = bigdf_1.append(bigdf_2).append(bigdf_3).append(df)
bigdf_1.to_pickle("maml-pifinnet-all.df")
# '''


''' eval pifinnet
seeds = list(range(5, 15))
bigdf_1 = pd.read_pickle("maml-pifinnet-all.df").reset_index(drop=True)
new_df = []
for width in [128, 256, 512, 1024, 2048]:
    for r in [50, 100, 200, 400]:
        df_row = bigdf_1[((bigdf_1['infnet_r'] == r) & (bigdf_1['hidden_size'] == width))]
        df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]
        print(width, r, df_row.val_accuracies_after.max())
        # if df_row["exp"].item() == "master-dingo": continue  # if master-dingo already done
        new_df.append(df_row)
new_df =  pd.concat(new_df)
# print(new_df)
with open('eval_maml.sh', 'w') as f:
    for ind in new_df.index:
        store = Store(new_df['head'][ind], new_df['tail'][ind])
        for seed in seeds:
            f.write(f"python -m meta.test --config '{new_df['head'][ind]}/{new_df['tail'][ind]}/config.json' --use-cuda --seed {seed}\n")
0/0
# '''

all_test_df = []
bigdf_1 = pd.read_pickle("maml-pifinnet-all.df").reset_index(drop=True)
for width in [128, 256, 512, 1024, 2048]:
    for r in [50, 100, 200, 400]:
        df_row = bigdf_1[((bigdf_1['infnet_r'] == r) & (bigdf_1['hidden_size'] == width))]
        df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]

        head = df_row['head'].iloc[0]
        tail = df_row['tail'].iloc[0]
        s = Store(head, tail)
        result_df = s['test_result_batch-1_step-1'].df
        for ind2 in result_df.index:
            all_test_df.append({'Net': 'PiNet', 'Width': width, "R": r, "val_acc": df_row.val_accuracies_after.max(), 'test_accuracies_after': result_df['accuracies_after'][ind2]})
        s.close()
bigdf_1 = pd.read_pickle("maml-infnet-all.df").reset_index(drop=True)
for r in [50, 100, 200, 400]:
    df_row = bigdf_1[bigdf_1['infnet_r'] == r]
    df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]

    head = df_row['head'].iloc[0]
    tail = df_row['tail'].iloc[0]
    s = Store(head, tail)
    result_df = s['test_result_batch-1_step-1'].df
    for ind2 in result_df.index:
        all_test_df.append({'Net': 'InfPiNet', 'Width': -1, "R": r, "val_acc": df_row.val_accuracies_after.max(), 'test_accuracies_after': result_df['accuracies_after'][ind2]})
    s.close()
bigdf_1 = pd.read_pickle("maml-mu-all.df").reset_index(drop=True)
bigdf_1 = bigdf_1[bigdf_1["batch_size"] < 10]                          # ignore munet where batch size > 10, incorrectly scanned 
for width in [128, 256, 512, 1024, 2048]:
    df_row = bigdf_1[bigdf_1['hidden_size'] == width]
    df_row = df_row[df_row.val_accuracies_after == df_row.val_accuracies_after.max()]

    head = df_row['head'].iloc[0]
    tail = df_row['tail'].iloc[0]
    s = Store(head, tail)
    result_df = s['test_result_batch-1_step-1'].df
    for ind2 in result_df.index:
        all_test_df.append({'Net': 'MuNet', 'Width': width, "R": -1, "val_acc": df_row.val_accuracies_after.max(), 'test_accuracies_after': result_df['accuracies_after'][ind2]})
    s.close()
all_test_df =  pd.DataFrame(all_test_df)

print(all_test_df)
all_test_df.to_pickle("maml-test-all.df")
0/0