import torch
from inf.utils import *
from inf.optim import *
import numpy as np
from torch.utils.data import Subset
import torch.utils.data as data_utils

def to_one_hot(target, num_cls=10, center=True):
  '''
  one-hot encoding.
  '''
  oh_target = target.new_zeros(target.shape[0], num_cls).type(torch.get_default_dtype())
  oh_target.scatter_(1, target.unsqueeze(-1), 1)
  if center:
    # oh_target -= 0.1
    oh_target -= 0.5
  return oh_target

def remove_extra_cls_cifar10(dataset, keep_cls):
  '''
  function to remove excess classes from cifar10 (for testing).
  '''
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
  '''
  function to remove excess classes from a imagenet.
  '''
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
