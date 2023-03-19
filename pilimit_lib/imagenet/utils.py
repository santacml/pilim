
# shamelessly stolen and then modified from: https://github.com/ActiveVisionLab/BNInterpolation/blob/master/imagenet32_dataset.py

import sys, os
import numpy as np

import pickle
from PIL import Image
import torch
from torch.utils.data import Subset
import torch.utils.data as data_utils
from torch.utils.data import Dataset

_base_folder = ''
# _train_list = ['train_data_batch_1',
#                'train_data_batch_2',
#                'train_data_batch_3',
#                'train_data_batch_4',
#                'train_data_batch_5',
#                'train_data_batch_6',
#                'train_data_batch_7',
#                'train_data_batch_8',
#                'train_data_batch_9',
#                'train_data_batch_10']
_train_list = ['val_data']
_val_list = ['val_data']
_label_file = 'map_clsloc.txt'


class ImageNet32(Dataset):
    """`ImageNet32 <https://patrykchrabaszcz.github.io/Imagenet32/>`_ dataset.
    Warning: this will load the whole dataset into memory! Please ensure that
    4 GB of memory is available before loading.
    Refer to ``map_clsloc.txt`` for label information.
    The integer labels in this dataset are offset by -1 from ``map_clsloc.txt``
    to make indexing start from 0.
    Args:
        root (string): Root directory of dataset where directory
            ``imagenet-32-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        exclude (list, optional): List of class indices to omit from dataset.
        remap_labels (bool, optional): If True and exclude is not None, remaps
            remaining class labels so it is contiguous.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, exclude=None, remap_labels=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        # Now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for f in _train_list:
                file = os.path.join(self.root, _base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    self.train_data.append(entry['data'])
                    self.train_labels += entry['labels']
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.train_labels = np.array(self.train_labels) - 1
        else:
            f = _val_list[0]
            file = os.path.join(self.root, _base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.val_data = entry['data']
                self.val_labels = entry['labels']
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.val_labels = np.array(self.val_labels) - 1

        if exclude is not None:
            if self.train:
                include_idx = np.isin(self.train_labels, exclude, invert=True)
                self.train_data = self.train_data[include_idx]
                self.train_labels = self.train_labels[include_idx]

                if remap_labels:
                    mapping = {y: x for x, y in enumerate(np.unique(self.train_labels))}
                    self.train_labels = remap(self.train_labels, mapping)

            else:
                include_idx = np.isin(self.val_labels, exclude, invert=True)
                self.val_data = self.val_data[include_idx]
                self.val_labels = self.val_labels[include_idx]

                if remap_labels:
                    mapping = {y: x for x, y in enumerate(np.unique(self.val_labels))}
                    self.val_labels = remap(self.val_labels, mapping)

        if self.train:
            self.train_labels = self.train_labels.tolist()
        else:
            self.val_labels = self.val_labels.tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.val_data[index], self.val_labels[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.val_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'val'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

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

if __name__ == "__main__":

    test = ImageNet32(".")

    print(test)

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
