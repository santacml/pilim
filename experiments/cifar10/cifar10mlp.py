'''
  A simple cifar10 training and testing file with standard training arguments. 
  Usage is extremely identical to pilimit_orig besides the file name.
  Training arguments each have a short description.
  It is recommended to use fp16 for speed and storage efficiency.

  This file is able to obtain extremely close results to the original paper, but slightly different,
  differences being from floating point rounding.

  This version of training seems to be slightly faster, but less memory efficient than pilimit_lib.

  This will NOT work in a multi-gpu environment.
  For a full CIFAR10 InfPiMLP training run, 8-16gb of GPU VRAM is necessary.

'''

import sys, os
import numpy as np
import argparse

from numpy.core.fromnumeric import clip
import torch.nn.functional as nnF
from torchvision import datasets, transforms
import pandas as pd
import time
import os
import torch
from pilimit_lib.inf.layers import *
from pilimit_lib.inf.optim import *
from pilimit_lib.inf.utils import *
from experiments.networks.networks import FinPiMLPSample, InfMLP

def main(arglst=None):
  parser = argparse.ArgumentParser(description='PyTorch training models',
                                  conflict_handler='resolve')
  parser.add_argument('--verbose', action='store_true',
                      help='verbose')

  # Custom arguments
  parser.add_argument('--data', type=str, default='./examples/cifar10/dataset',
                      help='location of the data corpus')
  parser.add_argument('--save-dir', type=str, default='',
                      help='directory to save results and checkpoints')
  parser.add_argument('--save-model', action='store_true',
                      help='whether to save checkpoints')
  parser.add_argument('--model-path', type=str, default='',
                      help='location of model to load')
  parser.add_argument('--test', action='store_true',
                      help='Whether to only test the network')
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
  parser.add_argument('--momentum', type=float, default=0,
                      help='momentum')
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
                      help='last layer alpha (applies to bias as well). '
                      'This is multiplicative with --last-bias-alpha')
  parser.add_argument('--last-bias-alpha', type=float, default=None,
                      help='last layer bias alpha. Default to --bias-alpha. '
                      'This overrides --bias-alpha for the last layer. '
                      'This is multiplicative with --last-layer-alpha')
  parser.add_argument('--last-layer-grad-no-alpha', action='store_true',
                      help="Don't multipy last layer gradients by last_layer_alpha. "
                          "This should only be used for debugging purposes.")
  parser.add_argument('--no-apply-lr-mult-to-wd', action='store_true',
                      help="Don't apply lr mult to weight decay. "
                          "This should only be used for debugging purposes.")
  parser.add_argument('--depth', type=int, default=1,
                      help='depth')
  parser.add_argument('--loss', type=str, default='xent', choices=['xent', 'mse'],
                      help='loss func')
  parser.add_argument('--r', type=int, default=200,
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

  # these arguments unused in paper
  parser.add_argument('--data-augment', action='store_true',
                      help='augment the dataset by flipping and cropping')
  parser.add_argument('--cuda-batch-size', type=int, default=None,
                      help='batch to cuda')
  parser.add_argument('--accum-steps', type=int, default=1, 
                      help='grad accumulation steps for smaller batch size')
  parser.add_argument('--teacher-path', type=str, default='',
                      help='location of teacher model to distill from')
  parser.add_argument('--teacher-alpha', type=float, default=.5,
                      help='teacher alpha')
  parser.add_argument('--teacher-temp', type=float, default=1,
                      help='teacher temp')

  if arglst is None:
    args = parser.parse_args()
  else:
    args = parser.parse_args(arglst)
  
  print(args)

  if args.width == 0:
    print("Got 0 width, defaulting to inf")
    args.width = None
  
  if args.scheduler == 'None':
      args.scheduler = None


  if not args.float:
    torch.set_default_dtype(torch.float16)
    print('using half precision')
  else:
    print('using full precision')


  torch.manual_seed(args.seed)
  use_cuda = args.cuda
  batch_size = args.batch_size
  test_batch_size = args.test_batch_size

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  transform_list = []
  if args.data_augment:
    transform_list.extend([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4)])
  transform_list.extend([transforms.ToTensor()])

  transform_list.extend([transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])
  transform = transforms.Compose(transform_list)

  trainset = datasets.CIFAR10(root=args.data, train=True,
                                          download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

  testset = datasets.CIFAR10(root=args.data, train=False,
                                        download=True, transform=transform)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                          shuffle=False, num_workers=8)


  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  def get_loss(output, target, teacher_output=None, reduction='mean', teacher_alpha=.3, teacher_temp=1):
    # we only ever use xent
    if args.loss == 'xent':
      loss = nnF.cross_entropy(output, target, reduction=reduction)
    elif args.loss == 'mse':
      oh_target = target.new_zeros(target.shape[0], 10).type(torch.get_default_dtype())
      oh_target.scatter_(1, target.unsqueeze(-1), 1)
      oh_target -= 0.1
      loss = nnF.mse_loss(output, oh_target, reduction=reduction)

    if teacher_output is not None:
        ditillation_loss = nnF.kl_div(
            nnF.softmax(output / teacher_temp, dim=1),
            nnF.softmax(teacher_output / teacher_temp, dim=1),
            reduction="batchmean"
        )
        loss = teacher_alpha * loss + (1 - teacher_alpha) * ditillation_loss
    return loss

  def train_nn(
      model, 
      device, 
      train_loader, 
      optimizer, 
      epoch, 
      log_interval=100, 
      gclip_sch=False, 
      max_batch_idx=None,
      scheduler=None, 
      teacher=None):
      '''
      Main training loop for one epoch.

      Inputs;
        model: the model to use
        device: torch device to send data to
        train_loader: torch dataloader for training
        optimizer: the optimizer
        epoch: what epoch we are on (for scheduling)
        log_interval: how often to log if args.human
        gclip_sch: gradient clipping schedule
        max_batch_idx: hard stop batch idx,
        scheduler: lr scheduler, 
        teacher: teacher for distillation (not used in paper)
      '''
      model.train()
      train_loss = 0
      losses = []
      correct = 0
      nbatches = len(train_loader)
      tic = time.time()

      forward_time = 0
      backward_time = 0
      clip_time = 0
      step_time = 0

      for batch_idx, (data, target) in enumerate(train_loader):
        if max_batch_idx is not None and batch_idx > max_batch_idx:
            break
        n = batch_idx + 1
        data, target = data.to(device).type(torch.get_default_dtype()), target.to(device)
        data = data.reshape(data.shape[0], -1)

        # 0 grad if we reach accum steps 
        if batch_idx % args.accum_steps == 0:
          optimizer.zero_grad()
        
        start = time.time()
        output = model(data)
        forward_time = forward_time * (n-1)/n + (time.time() - start) / n

        teacher_output = None
        if teacher is not None:
          teacher_output = teacher(data)

        loss = get_loss(output, target, teacher_output=teacher_output, teacher_alpha=args.teacher_alpha, teacher_temp=args.teacher_temp)
        losses.append(loss.item())
        
        start = time.time()
        loss.backward()
        stage_grad(model)
        backward_time = backward_time * (n-1)/n + (time.time() - start) / n

        if batch_idx % args.accum_steps == 0:
          start = time.time()
          unstage_grad(model)
          if gclip_sch and gclip_sch.gclip > 0:
              store_pi_grad_norm_(model.modules())
              if args.gclip_per_param:
                  for param in model.parameters():
                      clip_grad_norm_(param, gclip)
                      # torch.nn.utils.clip_grad_norm_(param, gclip_sch.gclip)  # normal torch usage
              else:
                  clip_grad_norm_(model.parameters(), gclip)
                  # torch.nn.utils.clip_grad_norm_(model.parameters(), gclip_sch.gclip)  # normal torch usage
          clip_time = clip_time * (n-1)/n + (time.time() - start) / n

          start = time.time()
          optimizer.step()

          step_time = step_time * (n-1)/n + (time.time() - start) / n
        

        train_loss += get_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.accum_steps == 0:
          if scheduler is not None:
              scheduler.step()
              if gclip_sch:
                  gclip_sch.step()
          
        if args.human and batch_idx % log_interval == 0:
            toc = time.time()
            elapsed = toc - tic
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.2f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), elapsed))
            tic = toc

        torch.cuda.empty_cache()
              
          
      train_loss /= len(train_loader.dataset)
      train_acc = correct / len(train_loader.dataset)
      if args.human:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
      # print("forward", forward_time)
      # print("backward", backward_time)
      # print("clip", clip_time)
      # print("step", step_time)

      return losses, train_acc

  def test_nn(model, device, test_loader):
      '''
      Test a model on a dataset (for validation/testing).
      '''
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


  lr = args.lr
  wd = args.wd
  gclip = args.gclip
  # depth
  L = args.depth
  # rank of Z space
  r = args.r
  din = 32**2 * 3
  dout = 10

  infnet = InfMLP(din, dout, r, L, device=device,
                  first_layer_alpha=args.first_layer_alpha,
                  last_layer_alpha=args.last_layer_alpha,
                  bias_alpha=args.bias_alpha,
                  last_bias_alpha=args.last_bias_alpha,
                  layernorm=args.layernorm)
  
  teacher = None
  if args.teacher_path and len(args.teacher_path) > 0:
    teacher = InfMLP(din, dout, r, L, device=device,
                    first_layer_alpha=1,
                    last_layer_alpha=.5,
                    bias_alpha=.5,
                    last_bias_alpha=args.last_bias_alpha,
                    layernorm=args.layernorm)
    print("loading teacher model from", args.teacher_path)
    teacher.load_state_dict(torch.load(args.teacher_path))

  mynet = None
  if args.width is None:
    # note we collect parameters to use in optimizer for both inf and finite networks.
    paramgroups = []
    # first layer weights
    paramgroups.append({
      'params': [infnet.layers[0].A],
      'lr': args.first_layer_lr_mult * lr,
      'weight_decay': wd / args.first_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    # biases
    if infnet.layers[0].bias is not None:
      paramgroups.append({
        'params': [l.bias for l in infnet.layers],
        'lr': args.bias_lr_mult * lr,
        'weight_decay': wd / args.bias_lr_mult if args.no_apply_lr_mult_to_wd else wd
        
      })
    # all other weights
    paramgroups.append({
      'params': [l.Amult for l in infnet.layers[1:-1]],
    })
    paramgroups.append({
      # 'params': [l.Amult for l in infnet.layers[-1:-1]],
      'params': [infnet.layers[-1].Amult],
      'lr': args.last_layer_lr_mult * lr,
      'weight_decay': wd / args.last_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    paramgroups.append({
      'params': [l.A for l in infnet.layers[1:]],
    })
    paramgroups.append({
      'params': [l.B for l in infnet.layers[1:]],
    })
    optimizer = PiSGD(paramgroups, lr, weight_decay=wd, momentum=args.momentum)
    mynet = infnet
  else:
    if args.last_layer_grad_no_alpha:
      raise NotImplementedError()

    mynet = FinPiMLPSample(infnet, args.width)
    
    if args.cuda:
      mynet = mynet.cuda()
    if args.gaussian_init:
      for _, lin in mynet.layers.items():
        lin.weight.data[:].normal_()
        lin.weight.data /= lin.weight.shape[1]**0.5 / 2**0.5
    if not args.float:
      # torch.set_default_dtype(torch.float16)
      mynet = mynet.half()
    
    paramgroups = []
    # first layer weights
    paramgroups.append({
      'params': [mynet.layers[0].weight],
      'lr': args.first_layer_lr_mult * lr,
      'weight_decay': wd / args.first_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    # last layer weights
    paramgroups.append({
      'params': [mynet.layers[-1].weight],
      'lr': args.last_layer_lr_mult * lr,
      'weight_decay': wd / args.last_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    # all other weights
    paramgroups.append({
      'params': [l.weight for l in mynet.layers[1:-1]],
    })    
    # biases
    if mynet.layers[0].bias is not None:
      paramgroups.append({
        'params': [l.bias for l in mynet.layers],
        'lr': args.bias_lr_mult * lr
      })
    optimizer = PiSGD(paramgroups, lr, weight_decay=wd, momentum=args.momentum)  # nesterov not implemented


  if args.model_path:
    print("loading model from", args.model_path)
    mynet.load_state_dict(torch.load(args.model_path))
  
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
  if args.scheduler == 'cosine':
      if args.verbose:
          print('cosine scheduler')
      sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * len(train_loader))
  elif args.scheduler == 'multistep':
      if args.verbose:
          print('multistep scheduler')
          print('milestones', milestones)
      sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_drop_ratio)
        



  train_losses = []
  train_accs = []
  test_losses = []
  test_accs = []

  log_df = []
  if args.save_dir:
    # print(f'results saved to {args.save_dir}')
    print(f"logs will be saved to {os.path.join(args.save_dir, 'log.df')}")
    if args.save_model:
      print(f"checkpoints saved to {os.path.join(args.save_dir, 'checkpoints')}")

  if args.save_model: # and test_acc == min(test_accs)
    model_path = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, f'epoch0.th')
    torch.save(mynet.state_dict(), model_file)


  if args.test:
    print("Testing Network")
    test_loss, test_acc = test_nn(mynet, device, test_loader)
    print(test_loss, test_acc)
    0/0


  for epoch in range(1, args.epochs+1):
    epoch_start = time.time()
    losses, train_acc = train_nn(mynet, device, train_loader, optimizer, epoch, gclip_sch=gclip_sch, scheduler=sch, teacher=teacher)
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
      # with open(model_file, 'wb') as f:
      #   torch.save(mynet, f)
      # mynet.save(model_file)
      torch.save(mynet.state_dict(), model_file)
      print(f'model saved at {model_file}')
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
    if args.save_dir:
      log_file = os.path.join(args.save_dir, 'log.df')
      os.makedirs(args.save_dir, exist_ok=True)
      pd.DataFrame(log_df).to_pickle(log_file)
      # print(f'log dataframe saved at {log_file}')

  
  return min(test_accs)

if __name__ == '__main__':
  main()