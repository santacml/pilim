'''
  A simple cifar10 testing file.

  This file allows for testing of a finite or inf pi-net, but also an ntk or nngp. 
  See example usage in the readme with optimal hyperparameters per network.

  In addition, this file allows for simple testing on the cifar10 test dataset,
  or, if desired, creating a feature kernel (finite or inf-width) from a network
  for feature kernel evaluation.
'''

import copy
from inf.kernel import relu_gp_fn, relu_ntk_fn

from inf.utils import J1
from inf.pimlp import *
from utils.data import *
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import time
import itertools
from tqdm import tqdm
import os
from opt_einsum import contract


def to_one_hot(target, center=True):
  oh_target = target.new_zeros(target.shape[0], 10).type(torch.get_default_dtype())
  oh_target.scatter_(1, target.unsqueeze(-1), 1)
  if center:
    # oh_target -= 0.1
    oh_target -= 0.5
  return oh_target

def get_kernel_and_labels_from_kerfn(kerfn, dataloader, datasize=60_000):
  dataloader = [(data.cuda(), target.cuda()) for (data, target) in dataloader]
  dataloader2 = copy.copy(dataloader)
  ker = torch.zeros(datasize, datasize, dtype=torch.double)
  labels = torch.zeros(datasize, 10, dtype=torch.double)

  for i, (data1, target1) in tqdm(list(enumerate(dataloader))):
    bsz = data1.shape[0]
    labels[i * bsz : (i+1) * bsz] = to_one_hot(target1)
    for j, (data2, _) in enumerate(dataloader2):
      data1 = data1.reshape(data1.shape[0], -1)
      data2 = data2.reshape(data2.shape[0], -1)
      ker[i * bsz : (i+1) * bsz, j * bsz : (j+1) * bsz] = kerfn(data1, data2).double()
      ker[j * bsz : (j+1) * bsz, i * bsz : (i+1) * bsz] = \
          ker[i * bsz : (i+1) * bsz, j * bsz : (j+1) * bsz].T

  return ker, labels

def kernel_predict(ker_inv_labels, ker_test_train):
  out = ker_test_train @ ker_inv_labels
  prediction = torch.argmax(out, dim=1)
  return out, prediction
  
def main(arglst=None):
  parser = argparse.ArgumentParser(description='PyTorch training ensemble models',
                                  conflict_handler='resolve')
  parser.add_argument('--verbose', action='store_true',
                      help='verbose')

  # Kernel arguments
  parser.add_argument('--gp', action='store_true',
      help='train the last layer of a relu MLP')
  parser.add_argument('--ntk', action='store_true',
      help='train via the NTK of a relu MLP')
  parser.add_argument('--varw', type=float, default=1,
      help='For GP model, all non-readout weights are initialized as N(0, varw/fanin). (default: 1)')
  parser.add_argument('--varb', type=float, default=0,
      help='For GP model, all non-readout biases are initialized as N(0, varb/fanin). (default: 0)')
  parser.add_argument('--first-layer-lr-mult', type=float, default=1,
                    help='learning rate multiplier for first layer weights. '
                         'Only applicable to NTK')
  parser.add_argument('--last-layer-lr-mult', type=float, default=1,
                    help='learning rate multiplier for last layer weights. '
                         'Only applicable to NTK')
  parser.add_argument('--bias-lr-mult', type=float, default=1,
                    help='learning rate multiplier for biases. '
                          'Only applicable to NTK')
  parser.add_argument('--depth', type=int, default=2,
                      help='depth')

  # Custom arguments
  parser.add_argument('--data', type=str, default='./dataset',
                      help='location of the data corpus')
  parser.add_argument('--load-model-path', type=str, default='',
                      help='where to save the trained model')
  parser.add_argument('--width', type=int, default=None,
                      help='if specified, sample finite width from infnet and test')
  parser.add_argument('--cuda', action='store_true',
                      help='Whether to use cuda')
  parser.add_argument('--float', action='store_true',
                      help='Whether to use fp32')
  parser.add_argument('--batch-size', type=int, default=64,
                      help='training bsz')
  parser.add_argument('--test-batch-size', type=int, default=64,
                      help='test bsz')
  parser.add_argument('--test-kernel', action='store_true',
                      help='test the kernel induced by the learned infnet embedding')
  parser.add_argument('--kernel-reg', type=float, default=1e-8,
                      help='diagonal regularization for the kernel')
  parser.add_argument('--multiple-regs', action='store_true',
                      help='use a predetermined range of kernel regs')
  parser.add_argument('--loss', type=str, default='xent', choices=['xent', 'mse'],
                      help='loss func')
  parser.add_argument('--human', action='store_true',
                      help='Whether to print huamn-friendly output')
  parser.add_argument('--quiet', action='store_true',
                      help='squash all prints')
  parser.add_argument('--seed', type=int, default=1,
                      help='random seed')
  parser.add_argument('--save-dir', type=str, default='.',
                      help='directory to save results')
  parser.add_argument('--save-kernel', action='store_true',
                      help='whether to save the kernel and the inverse kernel')
  parser.add_argument('--train-subset-size', type=int, default=None,
                      help='Set a training subset size of Cifar10 instead of the full dataset.')
  parser.add_argument('--test-subset-size', type=int, default=None,
                      help='Set a testing subset size of Cifar10 instead of the full dataset.')

  if arglst is None:
    args = parser.parse_args()
  else:
    args = parser.parse_args(arglst)

  if not args.float:
    torch.set_default_dtype(torch.float16)
    print('using half precision')
  else:
    print('using full precision')

  torch.manual_seed(args.seed)
  use_cuda = args.cuda
  test_batch_size = args.test_batch_size
  device = torch.device("cuda" if use_cuda else "cpu")
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])

  trainset = datasets.CIFAR10(root=args.data, train=True,
                                          download=True, transform=transform)

  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=8)

  testset = datasets.CIFAR10(root=args.data, train=False,
                                        download=True, transform=transform)

  test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                          shuffle=False, num_workers=8)

  def get_loss(output, target, reduction='mean'):
    if args.loss == 'xent':
      loss = F.cross_entropy(output, target, reduction=reduction)
    elif args.loss == 'mse':
      oh_target = to_one_hot(target)
      loss = F.mse_loss(output, oh_target, reduction=reduction)
    return loss
  
  @torch.no_grad()
  def get_feature_kernel_and_labels(infnet, dataloader):
    infnet.eval()
    g = []
    s = []
    labels = []
    dtype = torch.get_default_dtype()
    with torch.no_grad():
      for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        data = data.to(dtype)
        
        if isinstance(infnet, InfPiMLP):
          _ = infnet(data)
          g.append(infnet.gbars[infnet.L].cpu().double())
          s.append(infnet.ss[infnet.L].cpu().double())
        else:
          data = data.reshape(data.shape[0], -1)
          _, kernel_output = infnet(data, save_kernel_output=True)
          g.append(kernel_output.cpu().double())
        labels.append(to_one_hot(target).cpu().double())
    # shape (dataset_size, r)
    feats = torch.cat(g)
    del g
    # shape (dataset_size, 10)
    labels = torch.cat(labels)
    ker = feats @ feats.T
    del feats
    # del infnet
    torch.cuda.empty_cache()
    print("made it past kernel creation")

    
    # ker = 0.5 * J1(ker.cpu().double())
    if isinstance(net, InfPiMLP):
      ss = torch.flatten(torch.cat(s))
      batch_size = 1000
      m = ker.shape[0]
      # do this in batches because doing all at once usually overflows gpu
      for n in range(int(m / batch_size) + 1):
          idx1 = int(n*batch_size)
          idx2 = int((n+1)*batch_size)
          if (idx2 > m): idx2 = m
          if idx1 == idx2: break

          # normalize kernel
          # ker[idx1:idx2, :] = 0.5 * J1(ker[idx1:idx2, :].cpu().double())   * ss[idx1:idx2, None]  * ss[None, :]   # normalize

          # don't normalize kernel
          ker[idx1:idx2, :] = 0.5 * J1(ker[idx1:idx2, :].cpu().double())
      
      
      # ss = torch.flatten(torch.cat(s))
      # ker = ss[None, :] * ss[:, None] * ker
    else:
      D = torch.diag(ker)**(-1/2)
      # ker = .5 * D[:, None]**(-1/2) * D[None, :]**(-1/2) * ker

      # save memory hopefully
      ker = .5 * contract("kr, k, r -> kr", ker, D, D)

    print("kernel", ker)
    print("kernel diag", torch.diag(ker))

    return ker, labels


  def test_nn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device).type(torch.get_default_dtype()), target.to(device)
        data = data.reshape(data.shape[0], -1)
        output = model(data)
        test_loss += get_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if args.human:
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
    
    return test_loss, correct / len(test_loader.dataset)

  device = torch.device("cuda" if use_cuda else "cpu")

  os.makedirs(args.save_dir, exist_ok=True)
  
  log_df = []

  test_loss = -1
  test_acc = -1
  ker_acc = -1

  if not args.gp and not args.ntk:
    try:
      net = torch.load(args.load_model_path)
      if args.width and isinstance(net, InfPiMLP):
        print("sampling infnet with width", args.width )
        net = net.sample(args.width)
    except:
      print("loading error, trying old loading method")
      print("TEMP CODE")
      # rank of Z space
      din = 32**2 * 3

      num_cls = 10

      dout = num_cls
      #
      infnet = InfPiMLP(d=din, dout=dout, L=2, r=200,
                        first_layer_alpha=1,
                        last_layer_alpha=1,
                        initbuffersize=1000, device=device,
                        # bias_alpha=0.5,
                        # last_bias_alpha=0.5,
                        bias_alpha=1,
                        last_bias_alpha=1,
                        _last_layer_grad_no_alpha=True)

      if args.width:
        net = infnet.sample(args.width)
      else:
        net = infnet
      if args.load_model_path: net.load(args.load_model_path)


    if use_cuda:
      net.cuda()
    else:
      net.cpu()
    test_loss, test_acc = test_nn(net, device, test_loader)
    print(f'infnet: test loss: {test_loss}\ttest acc: {test_acc}')

  if args.test_kernel or args.gp or args.ntk:
    dataloader = itertools.chain(iter(train_loader), iter(test_loader))

    tic = time.time()
    print('making kernel')
    if args.gp:
      kerfn = relu_gp_fn([args.varw]*args.depth, [args.varb]*args.depth)
      ker, labels = get_kernel_and_labels_from_kerfn(kerfn, dataloader)
    elif args.ntk:
      kerfn = relu_ntk_fn(
        [args.varw]*(args.depth+1), [args.varb]*(args.depth+1),
        [args.first_layer_lr_mult] + [1]*(args.depth-1) + [args.last_layer_lr_mult],
        [args.bias_lr_mult] * (args.depth+1))
      ker, labels = get_kernel_and_labels_from_kerfn(kerfn, dataloader)
    else:
      ker, labels = get_feature_kernel_and_labels(net, dataloader)
    toc = time.time()
    print(f'{toc - tic} seconds')

    if args.save_kernel:
      print('saving kernel')
      tic = time.time()
      torch.save(ker, os.path.join(args.save_dir, 'ker.th'))
      torch.save(labels, os.path.join(args.save_dir, 'labels.th'))
      toc = time.time()
      print(f'{toc - tic} seconds')

    if args.multiple_regs:
      print("using multiple kernel reg values")
      regs = [10**(-n) for n in range(5,2,-1)]
    else:
      regs = [args.kernel_reg]



    for reg in regs:
      reg = args.kernel_reg
      idx = list(range(ker.shape[0]))
      ker[idx, idx] += reg
      N = 50000 

      if args.save_kernel:
        print('inverting kernel')
        tic = time.time()
        ker_inv = torch.inverse(ker[:N, :N])
        toc = time.time()
        print(f'{toc - tic} seconds')

        print('saving inverse kernel')
        tic = time.time()
        torch.save(ker_inv, os.path.join(args.save_dir, 'ker_inv.th'))
        toc = time.time()
        print(f'{toc - tic} seconds')

        ker_inv_labels = ker_inv @ labels[:N]
      else:
        print('inverting with solve - not saving kernel')
        ker_inv_labels = torch.linalg.solve(ker[:N, :N], labels[:N])

      print('making prediction')
      tic = time.time()
      ker_test_train = ker[N:, :N]
      out, pred = kernel_predict(ker_inv_labels, ker_test_train)
      ker_acc = (torch.argmax(labels[N:], dim=1) == pred).float().mean()
      print(f'kernel acc: {ker_acc}')
      toc = time.time()
      print(f'{toc - tic} seconds')

      ker[idx, idx] -= reg

      if isinstance(net, FinPiMLP):
        width = net.width
      else:
        width = None
      log_df.append(
        dict(
          inf_test_loss=test_loss,
          inf_test_acc=test_acc,
          ker_acc=ker_acc,
          ker_reg=reg,
          loaded_width=width,
          **vars(args)
      ))

  if args.save_dir:
    log_file = os.path.join(args.save_dir, 'log.df')
    os.makedirs(args.save_dir, exist_ok=True)
    pd.DataFrame(log_df).to_pickle(log_file)

if __name__ == '__main__':
  main()
