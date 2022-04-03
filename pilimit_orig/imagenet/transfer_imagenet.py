# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys, os
from utils.data import *
from utils.kernels import *
from utils.resnet import *
from inf.pimlp import *
from inf.optim import InfSGD, InfMultiStepLR, MultiStepGClip
from imagenet.imagenet import ImageNet32
import numpy as np
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import SGD
import pandas as pd
import time
from torch.utils.data import Subset
import psutil
import pickle
import random

num_cifar10_cls = 3
# num_cifar10_cls = 10

def main(arglst=None):
  parser = argparse.ArgumentParser(description='PyTorch training ensemble models',
                                  conflict_handler='resolve')
  parser.add_argument('--verbose', action='store_true',
                      help='verbose')

  # Custom arguments
  parser.add_argument('--imagenet-data', type=str, default='/mnt/output/',
                      help='location of the imagenet data corpus')
  parser.add_argument('--cifar-data', type=str, default='./dataset',
                      help='location of the cifar data corpus')
  parser.add_argument('--save-dir', type=str, default='',
                      help='directory to save results and checkpoints')
  parser.add_argument('--save-model', action='store_true',
                      help='whether to save checkpoints')
  parser.add_argument('--cuda', action='store_true',
                      help='Whether to use cuda')
  parser.add_argument('--float', action='store_true',
                      help='Whether to use fp32')
  parser.add_argument('--batch-size', type=int, default=64,
                      help='training bsz')
  parser.add_argument('--test-batch-size', type=int, default=64,
                      help='test bsz')
  parser.add_argument('--lr', type=float, default=2.0,
                      help='learning rate')
  parser.add_argument('--scheduler', type=str, default='None', choices=('None', 'cosine', 'multistep'),
      help='None | cosine | multistep. (default: None)')
  parser.add_argument('--lr-drop-ratio', type=float, default=0.5,
      help='if using multistep scheduler, lr is multiplied by this number at milestones')
  parser.add_argument('--gclip-drop-ratio', type=float, default=0.5,
      help='if using multistep gclip scheduler, gclip is multiplied by this number at milestones')
  parser.add_argument('--lr-drop-milestones', type=str, default='',
      help='comma-separated list of epoch numbers. If using multistep scheduler, lr is dropped at after these epochs*num-batches steps.')
  parser.add_argument('--gclip-drop-milestones', type=str, default='',
      help='comma-separated list of epoch numbers. If using multistep scheduler, gclip is dropped at after these epochs*num-batches steps.')
  parser.add_argument('--no-apply-lr-mult-to-wd', action='store_true',
                      help="Don't apply lr mult to weight decay. "
                          "This should only be used for debugging purposes.")
  parser.add_argument('--wd', type=float, default=1e-4,
                      help='weight decay')
  parser.add_argument('--gclip', type=float, default=0.2,
                      help='gradient clipping')
  parser.add_argument('--gclip-per-param', action='store_true',
                      help='do gradient clipping for every parameter tensor individually')
  parser.add_argument('--layernorm', action='store_true',
                      help='layernorm')
  parser.add_argument('--first-layer-lr-mult', type=float, default=0.1,
                      help='learning rate multiplier for first layer weights')
  parser.add_argument('--last-layer-lr-mult', type=float, default=1,
                      help='learning rate multiplier for last layer weights')
  parser.add_argument('--bias-lr-mult', type=float, default=0.5,
                      help='learning rate multiplier for biases')
  parser.add_argument('--first-layer-alpha', type=float, default=1,
                      help='first layer alpha')
  parser.add_argument('--bias-alpha', type=float, default=1,
                      help='bias alpha')
  parser.add_argument('--last-layer-alpha', type=float, default=1,
                      help='last layer alpha')
  parser.add_argument('--depth', type=int, default=2,
                      help='depth')
  parser.add_argument('--loss', type=str, default='xent', choices=['xent', 'mse'],
                      help='loss func')
  parser.add_argument('--r', type=int, default=400,
                      help='rank of Z space')
  parser.add_argument('--init-from-data', action='store_true',
                      help='initializing infnet from data')
  parser.add_argument('--init-A-B-direct', action='store_true',
                      help='initializing infnet using gradient descent on A and B directly')
  parser.add_argument('--gaussian-init', action='store_true',
                      help='initializing finnet with gaussians')
  parser.add_argument('--cycarr', action='store_true',
                      help='Whether to use CycArr; otherwise DynArr')
  parser.add_argument('--human', action='store_true',
                      help='Whether to print huamn-friendly output')
  parser.add_argument('--width', type=int, default=None,
                      help='width of the network; default is Inf')
  parser.add_argument('--epochs', type=int, default=24,
                      help='number of training epochs; default 25')
  parser.add_argument('--quiet', action='store_true',
                      help='squash all prints')
  parser.add_argument('--seed', type=int, default=1,
                      help='random seed')
  parser.add_argument('--transfer', action='store_true',
                      help='perform transfer learning')
  parser.add_argument('--solve-kernel', action='store_true',
                      help='use torch solve instead of inverse')
  parser.add_argument('--resnet', action='store_true',
                      help='use a resnet')
  parser.add_argument('--transfer-milestones', type=str, default='',
      help='comma-separated list of epoch numbers. Transfer kernel performance is evaluated at these steps.')
                      
  # parser.add_argument('--zero-transfer-init', action='store_true',
  #                     help='Initialize transfer net with 0 for output layer.')

  if arglst is None:
    args = parser.parse_args()
  else:
    args = parser.parse_args(arglst)
  
  if args.scheduler == 'None':
      args.scheduler = None


  if not args.float:
    torch.set_default_dtype(torch.float16)
    print('using half precision')
  else:
    print('using full precision')

  # %%
  torch.manual_seed(args.seed)
  use_cuda = args.cuda
  batch_size = args.batch_size
  test_batch_size = args.test_batch_size

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])

  transform_imgnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

  trainset = ImageNet32(args.imagenet_data, transform=transform_imgnet)

  # keep_cls = []
  # while len(keep_cls) < 250:
  #   n = random.randint(0,1000)
  #   if not (n in keep_cls): keep_cls.append(n)

  # best run
  keep_cls = [294, 828, 241, 320, 210, 561, 67, 706, 956, 996, 490, 166, 287, 337, 726, 305, 688, 314, 195, 107, 433, 802, 717, 868, 697, 335, 127, 359, 101, 236, 558, 120, 249, 982, 888, 987, 944, 885, 547, 588, 498, 20, 778, 74, 658, 489, 428, 821, 151, 955, 776, 979, 389, 256, 126, 77, 27, 580, 750, 557, 758, 829, 275, 649, 11, 850, 784, 894, 153, 651, 769, 51, 5, 252, 650, 156, 771, 701, 161, 831, 723, 233, 484, 974, 554, 447, 246, 176, 28, 946, 272, 783, 160, 603, 104, 510, 69, 13, 366, 924, 369, 152, 612, 158, 324, 203, 845, 388, 667, 38, 75, 782, 482, 730, 684, 132, 253, 220, 448, 313, 623, 360, 570, 886, 640, 440, 700, 240, 643, 131, 963, 116, 283, 239, 830, 197, 841, 933, 459, 398, 408, 105, 984, 574, 259, 437, 362, 507, 391, 922, 58, 983, 596, 288, 642, 869, 568, 450, 88, 442, 480, 670, 826, 225, 560, 215, 403, 426, 772, 672, 976, 932, 330, 853, 25, 164, 686, 137, 421, 235, 50, 668, 273, 501, 609, 49, 64, 361, 801, 78, 791, 96, 304, 261, 102, 654, 962, 266, 798, 226, 998, 443, 222, 781, 870, 780, 427, 710, 889, 368, 31, 599, 297, 915, 377, 214, 415, 600, 890, 355, 977, 319, 282, 971, 943, 436, 497, 24, 586, 202, 787, 444, 634, 815, 861, 289, 945, 518, 907, 988, 46, 595, 880, 941, 449, 953, 786, 262, 939, 390]

  labels_to_keep = len(keep_cls)
  labels_per_chunk = 1
  # labels_to_keep = 20
  print("Keeping only labels:", keep_cls)
  trainset = remove_extra_cls_imagenet(trainset, keep_cls)
  
  # trainset.train_labels = (targets / labels_per_chunk).astype(np.int).tolist()  # chunk labels
# 
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

  trainset_transfer = datasets.CIFAR10(root=args.cifar_data, train=True,
                                          download=True, transform=transform)
  # use 3 classes of cifar10 for speed
  trainset_transfer = remove_extra_cls_cifar10(trainset_transfer, [0,5,6])
  
  train_loader_transfer = torch.utils.data.DataLoader(trainset_transfer, batch_size=32,
                                            shuffle=True, num_workers=8)
  
  testset = ImageNet32(args.imagenet_data, train=False, transform=transform_imgnet)
  
  testset = remove_extra_cls_imagenet(testset, keep_cls)

  # testset.val_labels = (targets / labels_per_chunk).astype(np.int).tolist()  # chunk in groups 

  test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                          shuffle=False, num_workers=8)
  # '''

  testset_transfer = datasets.CIFAR10(root=args.cifar_data, train=False,
                                        download=True, transform=transform)
  
  testset_transfer = remove_extra_cls_cifar10(testset_transfer, [0,5,6])

  test_loader_transfer = torch.utils.data.DataLoader(testset_transfer, batch_size=test_batch_size,
                                          shuffle=False, num_workers=8)

  def get_loss(output, target, reduction='mean'):
    if args.loss == 'xent':
      loss = F.cross_entropy(output, target, reduction=reduction)
    elif args.loss == 'mse':
      oh_target = target.new_zeros(target.shape[0], 10).type(torch.get_default_dtype())
      oh_target.scatter_(1, target.unsqueeze(-1), 1)
      oh_target -= 0.1
      loss = F.mse_loss(output, oh_target, reduction=reduction)
    return loss

  def train_nn(model, device, train_loader, optimizer, epoch, Gproj=True,
              log_interval=100, gclip_sch=False, max_batch_idx=None,
              lr_gain=0, scheduler=None, transfer=False):
      model.train()
      train_loss = 0
      losses = []
      correct = 0
      nbatches = len(train_loader)
      tic = time.time()
      for batch_idx, (data, target) in enumerate(train_loader):
          if max_batch_idx is not None and batch_idx > max_batch_idx:
            break
          data, target = data.to(device).type(torch.get_default_dtype()), target.to(device)

          if not args.resnet: data = data.reshape(data.shape[0], -1)
          optimizer.zero_grad()
          output = model(data)
          if isinstance(model, InfPiMLP):
            output.requires_grad = True
            output.retain_grad()
            loss = get_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            model.backward(output.grad)
            if gclip_sch and gclip_sch.gclip > 0:
              model.gclip(gclip_sch.gclip, per_param=args.gclip_per_param)
          else:
            loss = get_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            if not args.resnet and Gproj:
              model.Gproj()
            if gclip_sch and gclip_sch.gclip > 0:
              if args.gclip_per_param:
                for param in model.parameters():
                  torch.nn.utils.clip_grad_norm_(param, gclip_sch.gclip)
              else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip_sch.gclip)
          optimizer.step()
          if lr_gain > 0:
            optimizer.param_groups[0]['lr'] += lr_gain / nbatches
          train_loss += get_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()

          if scheduler is not None:
            scheduler.step()
            if gclip_sch:
              gclip_sch.step()
          
          if args.human and batch_idx % log_interval == 0:
            toc = time.time()
            elapsed = toc - tic
            process = psutil.Process(os.getpid())
            cpu_memory = process.memory_info().rss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.2f}, cuda memory {:.8f}, memory {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), elapsed, torch.cuda.max_memory_reserved()/ 1e9, cpu_memory / 1e9  ))
            tic = toc
              
          
      train_loss /= len(train_loader.dataset)
      train_acc = correct / len(train_loader.dataset)
      if args.human:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
      
      return losses, train_acc
  
  def test_nn(model, device, test_loader, transfer=False):
      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device).type(torch.get_default_dtype()), target.to(device)

              if not args.resnet: data = data.reshape(data.shape[0], -1)
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



  # %%
  lr = args.lr
  wd = args.wd
  gclip = args.gclip
  width = args.width
  # depth
  L = args.depth
  # rank of Z space
  r = args.r
  din = 32**2 * 3
#   dout = 1000
  # dout = 100
  dout = int(labels_to_keep / labels_per_chunk)
  #
  infnet = InfPiMLP(d=din, dout=dout, L=L, r=r,
                    first_layer_alpha=args.first_layer_alpha,
                    last_layer_alpha=args.last_layer_alpha,
                    initbuffersize=1000, device=device,
                    bias_alpha=args.bias_alpha,
                    layernorm=args.layernorm,
                    resizemult=1.01,    # so that when this net gets very, very large, we only add .01*current size. will be slower but use max mem.
                    # readout_zero_init = True,
                    # cuda_batch_size=1000000,
                    # switch to DynArr if don't want cyclic
                    arrbackend=CycArr if args.cycarr else DynArr, maxsize=10**6)

  if args.init_from_data:
    # raise NotImplementedError()
    loader = torch.utils.data.DataLoader(trainset, batch_size=r,
                                          shuffle=True, num_workers=2)
    X = next(iter(loader))[0]
    X = X.reshape(X.shape[0], -1)
    infnet.initialize_from_data(X, dotest=False)


  width = args.width

  mynet = None
  if width is None:
    optimizer = InfSGD(infnet, lr, wd=wd,
      first_layer_lr_mult=args.first_layer_lr_mult,
      last_layer_lr_mult=args.last_layer_lr_mult,
      bias_lr_mult=args.bias_lr_mult,
      apply_lr_mult_to_wd=not args.no_apply_lr_mult_to_wd)
    mynet = infnet
  else:
    torch.set_default_dtype(torch.float32)
    mynet = infnet.sample(width)
    mynet = mynet.cuda()
    if args.gaussian_init:
      for _, lin in mynet.linears.items():
        lin.weight.data[:].normal_()
        lin.weight.data /= lin.weight.shape[1]**0.5 / 2**0.5
    if not args.float:
      torch.set_default_dtype(torch.float16)
      mynet = mynet.half()
    # if args.bias_lr_mult != 1 or args.first_layer_lr_mult != 1:
      # import pdb; pdb.set_trace()
      # raise NotImplementedError()
    paramgroups = []
    # first layer weights
    paramgroups.append({
      'params': [mynet._linears[0].weight],
      'lr': args.first_layer_lr_mult * lr,
      'weight_decay': wd / args.first_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    # biases
    if args.bias_alpha != 0:
      paramgroups.append({
        'params': [l.bias for l in mynet._linears],
        'lr': args.bias_lr_mult * lr
      })
    # all other weights
    paramgroups.append({
      'params': [l.weight for l in mynet._linears[1:]],
    })
    optimizer = SGD(paramgroups, lr, weight_decay=wd)
    # optimizer = SGD(mynet.parameters(), lr, weight_decay=wd)

  milestones = []
  if args.lr_drop_milestones:
    milestones = [int(float(e) * len(train_loader)) for e in args.lr_drop_milestones.split(',')]
  gclip_sch = None
  gclip_milestones = []
  if args.gclip_drop_milestones:
    gclip_milestones = [int(float(e) * len(train_loader)) for e in args.gclip_drop_milestones.split(',')]
  gclip_sch = MultiStepGClip(gclip, milestones=milestones, gamma=args.lr_drop_ratio)
  if args.verbose:
    print('gclip milestones', gclip_milestones)
  sch = None
  if width is None:
    if args.scheduler == 'cosine':
      raise NotImplementedError()
    elif args.scheduler == 'multistep':
      if args.verbose:
          print('multistep scheduler')
          print('milestones', milestones)
      sch = InfMultiStepLR(optimizer, milestones=milestones, gamma=args.lr_drop_ratio)
  else:
    if args.scheduler == 'cosine':
        if args.verbose:
            print('cosine scheduler')
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * len(train_loader))
    elif args.scheduler == 'multistep':
        if args.verbose:
            print('multistep scheduler')
            print('milestones', milestones)
        sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_drop_ratio)
        


  if args.resnet:
    print("TESTING USING RESNET38 - SOME ARGS MAY NOT WORK")
    infnet = None
    # mynet = resnet38(250)
    # mynet = resnet38(1000)
    mynet = resnet56(1000)
    if args.cuda:
      mynet.cuda()
    optimizer = SGD(mynet.parameters(), lr, weight_decay=wd)


  train_losses = []
  train_accs = []
  test_losses = []
  test_accs = []

  log_df = []
  transfer_df = []

  # kernel_regs = [10**(-n) for n in range(1,7)]
  kernel_regs = [0.001]
  
  if args.transfer_milestones:
    transfer_milestones = [int(e) for e in args.transfer_milestones.split(',')]
    
  if args.save_dir:
    # print(f'results saved to {args.save_dir}')
    print(f"logs will be saved to {os.path.join(args.save_dir, 'log.df')}")
    if args.save_model:
      print(f"checkpoints saved to {os.path.join(args.save_dir, 'checkpoints')}")
      # save initialization
      model_path = os.path.join(args.save_dir, 'checkpoints')
      os.makedirs(model_path, exist_ok=True)
      model_file = os.path.join(model_path, f'epoch0.th')
      with open(model_file, 'wb') as f:
        torch.save(mynet, f)


  if args.transfer and 0 in transfer_milestones:
    print("Evaluating transferred learning pre-training (no finetuning)")
    for kernel_reg in kernel_regs:
        print("Using ridge value", kernel_reg)
        kernel_acc = test_kernel_cifar10(mynet, train_loader_transfer, test_loader_transfer, len(trainset_transfer), num_cls=3, kernel_reg=kernel_reg, solve=args.solve_kernel, normalize=False)
    
        transfer_df.append(
          dict(
            epoch=0,
            kernel_acc=kernel_acc,
            kernel_reg=kernel_reg,
            **vars(args)
        ))

  # %%
  for epoch in range(1, args.epochs+1):
    epoch_start = time.time()
    losses, train_acc = train_nn(mynet, device, train_loader, optimizer, epoch, Gproj=not args.gaussian_init, gclip_sch=gclip_sch, scheduler=sch)
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    train_losses.append(losses)
    train_accs.append(train_acc)
    test_loss, test_acc = test_nn(mynet, device, test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    if args.save_model: # and test_acc == min(test_accs)
      model_path = os.path.join(args.save_dir, 'checkpoints')
      os.makedirs(model_path, exist_ok=True)
      model_file = os.path.join(model_path, f'epoch{epoch}.th')
      with open(model_file, 'wb') as f:
        torch.save(mynet, f)
    log_df.append(
      dict(
        epoch=epoch,
        train_loss=np.mean(train_losses[-1]),
        test_loss=test_losses[-1],
        train_acc=train_acc,
        test_acc=test_accs[-1],
        epoch_time=epoch_time,
        **vars(args)
    ))
    if not args.human and not args.quiet:
      if epoch == 1:
        header = f'epoch\ttr loss\tts loss\ttr acc\tts acc\ttime'
        print(header)
      stats = f'{epoch}\t{np.mean(train_losses[-1]):.3f}\t{test_losses[-1]:.3f}\t{train_acc}\t{test_accs[-1]}\t{epoch_time/60.:0.2f}'
      print(stats)

    if args.transfer and epoch in transfer_milestones:
      print("Evaluating transferred learning mid-training (no finetuning)")
      try:
        for kernel_reg in kernel_regs:
            print("Using ridge value", kernel_reg)
            kernel_acc = test_kernel_cifar10(mynet, train_loader_transfer, test_loader_transfer, len(trainset_transfer), num_cls=3, kernel_reg=kernel_reg, solve=args.solve_kernel)
        
            transfer_df.append(
              dict(
                epoch=epoch,
                kernel_acc=kernel_acc,
                kernel_reg=kernel_reg,
                **vars(args)
            ))
      except Exception as e:
        print("EXCEPTION CAUGHT - RESUMING TRAINING")
        print(e)


  if args.save_dir:
    pd.DataFrame(log_df).to_pickle(os.path.join(args.save_dir, 'log.df'))
    print(f'log dataframe saved at {args.save_dir}')

  if args.transfer:
    print("Evaluating transferred learning")

    for kernel_reg in kernel_regs:
      print("Using ridge value", kernel_reg)
      kernel_acc = test_kernel_cifar10(mynet, train_loader_transfer, test_loader_transfer, len(trainset_transfer), num_cls=3, kernel_reg=kernel_reg, solve=args.solve_kernel)

      transfer_df.append(
        dict(
          epoch=epoch,
          kernel_acc=kernel_acc,
          kernel_reg=kernel_reg,
          **vars(args)
      ))

  if args.save_dir:
    pd.DataFrame(transfer_df).to_pickle(os.path.join(args.save_dir, 'transfer_log.df'))
    print(f'log dataframe saved at {args.save_dir}')

  return min(test_accs)

# %%
if __name__ == '__main__':
  main()
