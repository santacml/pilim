import torch
from torch import nn, optim
from inf.utils import *
from inf.optim import *
from inf.pimlp import FinPiMLP, InfPiMLP
from utils.data import *
from utils.resnet import ResNet
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

def get_feature_kernel_and_labels(net, dataloader, num_cls=10, normalize=False):
  net.eval()
  g = []
  g2 = []
  labels = []
  dtype = torch.get_default_dtype()
  with torch.no_grad():
    for data, target in dataloader:
      # data, target = data.to(net.device), target.to(net.device)
      data, target = data.to("cuda"), target.to("cuda")

      data = data.to(dtype)
      if isinstance(net, InfPiMLP):
        _ = net(data)
        if normalize:
          g.append(net.gbars[net.L].cpu().double())
        else:
          g.append(net.gs[net.L].cpu().double())

        labels.append(to_one_hot(target, num_cls=num_cls).cpu().double())
      elif isinstance(net, FinPiMLP):
        data = data.reshape(data.shape[0], -1)
        _, kernel_output = net(data, save_kernel_output=True)
        # g.append(net.kernel_output.cpu().double())

        g.append(kernel_output.double())
        labels.append(to_one_hot(target, num_cls=num_cls).double())
      elif isinstance(net, ResNet):
        _ = net(data, save_kernel_output=True)
        # g.append(net.kernel_output.cpu().double())

        g.append(net.kernel_output.double().cpu())
        labels.append(to_one_hot(target, num_cls=num_cls).double().cpu())
      else:
        raise "Undetermined model type for kernel creation."



  # shape (dataset_size, r)
  feats = torch.cat(g)
  del g
  # shape (dataset_size, 10)
  labels = torch.cat(labels)
  ker = feats @ feats.T
  del feats
  # del infnet
  torch.cuda.empty_cache()
  if isinstance(net, InfPiMLP):
    batch_size = 1000
    m = ker.shape[0]
    for n in range(int(m / batch_size) + 1):
        idx1 = int(n*batch_size)
        idx2 = int((n+1)*batch_size)
        if (idx2 > m): idx2 = m
        if idx1 == idx2: break

        ker[idx1:idx2, :] = 0.5 * J1(ker[idx1:idx2, :].cpu().double()) 
        
    # ker = 0.5 * J1(ker.cpu().double()) 

  # return ker.double(), labels.double()
  return ker, labels

def kernel_predict(ker_inv_labels, ker_test_train):
  # shape (batch, 10)
  out = ker_test_train @ ker_inv_labels
  prediction = torch.argmax(out, dim=1)
  return out, prediction

def test_kernel_cifar10(net, train_loader, test_loader, dataset_len, num_cls=10, kernel_reg=1e-8, solve=False, inv_cuda=False, normalize=False, save_dir=None, save_name=None):
  dataloader = itertools.chain(iter(train_loader), iter(test_loader))
  print('making kernel')
  
  if save_dir != None and save_name != None and os.path.isfile(os.path.join(save_dir, f'{save_name}_ker.th') and os.path.isfile(os.path.join(save_dir, f'{save_name}_labels.th'))):
    print("loading existing kernel from", save_dir, "with name", save_name)
    ker = torch.load(os.path.join(save_dir, f'{save_name}_ker.th'))
    labels = torch.load(os.path.join(save_dir, f'{save_name}_labels.th'))
  else:
    if inv_cuda:
      net.cuda() # net was previously moved to cpu to save space
    ker, labels = get_feature_kernel_and_labels(net, dataloader, num_cls=num_cls, normalize=normalize)

  if save_dir != None and save_name != None and not os.path.isfile(os.path.join(save_dir, f'{save_name}_ker.th') and not os.path.isfile(os.path.join(save_dir, f'{save_name}_labels.th'))):
    print("saving kernel and labels to", save_dir, "with name", save_name)
    torch.save(ker, os.path.join(save_dir, f'{save_name}_ker.th'))
    torch.save(labels, os.path.join(save_dir, f'{save_name}_labels.th'))

  # reg = kernel_reg
  # for n in range(ker.shape[0]):
  #   ker[n,n] += reg
  # ker += reg * torch.eye(ker.shape[0], dtype=torch.float64)
  idx = list(range(ker.shape[0]))
  ker[idx, idx] += kernel_reg

  N = dataset_len

  if inv_cuda:
    # net.cpu() # move net to cpu to save cuda space
    ker = ker.cuda()
    labels = labels.cuda()

  if solve:
    print('inverting kernel using SOLVE - cannot save inverse')
    ker_inv_labels = torch.linalg.solve(ker[:N, :N], labels[:N])
  else:
    print('inverting kernel')
    ker_inv = torch.inverse(ker[:N, :N])

    if save_dir != None and save_name != None:
      print("saving inv kernel to", save_dir, "with name", save_name)
      torch.save(ker_inv, os.path.join(save_dir, f'{save_name}_ker_inv.th'))

    ker_inv_labels = ker_inv @ labels[:N]
  ker_test_train = ker[N:, :N]
  print('making prediction')
  out, pred = kernel_predict(ker_inv_labels, ker_test_train)
  ker_acc = (torch.argmax(labels[N:], dim=1) == pred).float().mean()
  print(f'kernel acc: {ker_acc}')

  return ker_acc
