import torch
from torch import nn, optim
from inf.utils import *
from inf.optim import *
from inf.pimlp import FinPiMLP, InfPiMLP
from tqdm.notebook import tqdm
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import numpy as np
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import SGD
from datetime import datetime
import os
import psutil
sns.set()
import time
import itertools
from torch.utils.data import Subset
import torch.utils.data as data_utils

def get_data(train_sample_limit=None, test_sample_limit=None):
  torch.manual_seed(3)
  np.random.seed(3)

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])

  trainset = datasets.CIFAR10(root='./dataset', train=True,
                                          download=True, transform=transform)
                                          
  testset = datasets.CIFAR10(root='./dataset', train=False,
                                        download=True, transform=transform)
                                          
  if train_sample_limit is not None:
    indices = torch.arange(train_sample_limit)
    trainset = data_utils.Subset(trainset, indices)
    print("Using only", len(trainset), "training samples")
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)

  if test_sample_limit is not None:
    indices = torch.arange(test_sample_limit)
    testset = data_utils.Subset(testset, indices)
    print("Using only", len(testset), "testing samples")
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
    #                                         shuffle=False, num_workers=2)

  return trainset, testset

def to_one_hot(target, num_cls=10, center=True):
  # TRANSFER
  oh_target = target.new_zeros(target.shape[0], num_cls).type(torch.get_default_dtype())
  oh_target.scatter_(1, target.unsqueeze(-1), 1)
  if center:
    # oh_target -= 0.1
    oh_target -= 0.5
  return oh_target

def remove_extra_cls_cifar10(dataset, keep_cls):
  map = {}
  i = 0
  for n in range(10):
    if n in keep_cls:
      map[n] = i
      i += 1
    else:
      map[n] = -1
  
  targets =  np.array(dataset.targets)
  idx = np.array(range(len(targets)))
  idx_to_keep = np.isin(targets, keep_cls)
  idx = idx[idx_to_keep]
  
  dataset.targets = torch.from_numpy(np.array(dataset.targets)).apply_(map.get).tolist()

  return Subset(dataset, idx)

def remove_extra_cls_imagenet(dataset, keep_cls):
  map = {}
  i = 0
  for n in range(1001):
    if n in keep_cls:
      map[n] = i
      i += 1
    else:
      map[n] = -1
  
  
  if dataset.train:
    targets =  np.array(dataset.train_labels)
    idx = np.array(range(len(targets)))
    idx_to_keep = np.isin(targets, keep_cls)
    idx = idx[idx_to_keep]
    
    dataset.train_labels = torch.from_numpy(np.array(dataset.train_labels)).apply_(map.get).tolist()
  else:
    targets =  np.array(dataset.val_labels)
    idx = np.array(range(len(targets)))
    idx_to_keep = np.isin(targets, keep_cls)
    idx = idx[idx_to_keep]

    dataset.val_labels = torch.from_numpy(np.array(dataset.val_labels)).apply_(map.get).tolist()
  
  return Subset(dataset, idx)
