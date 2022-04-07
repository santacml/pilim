'''
  A simple cifar10 training and testing file with standard training arguments. 

  See example usage in the readme with optimal hyperparameters per network.
  Training arguments each have a short description.

  It is recommended to use fp16 for speed and storage efficiency.
  
  This will NOT work in a multi-gpu environment.
  For a full CIFAR10 InfPiMLP training run, 8-16gb of GPU VRAM is necessary.
'''
from inf.pimlp import *
from inf.optim import InfSGD, InfMultiStepLR, MultiStepGClip
import numpy as np
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import SGD
import pandas as pd
import time
import os

def main(arglst=None):
  parser = argparse.ArgumentParser(description='PyTorch training ensemble models',
                                  conflict_handler='resolve')
  parser.add_argument('--verbose', action='store_true',
                      help='verbose')

  # Custom arguments
  parser.add_argument('--data', type=str, default='./dataset',
                      help='location of the data corpus')
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
  parser.add_argument('--no-Gproj', action='store_true',
                      help='do not do G projection')
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
  parser.add_argument('--depth', type=int, default=2,
                      help='depth')
  parser.add_argument('--loss', type=str, default='xent', choices=['xent', 'mse'],
                      help='loss func')
  parser.add_argument('--r', type=int, default=200,
                      help='rank of Z space')
  parser.add_argument('--init-from-data', action='store_true',
                      help='initializing infnet from data')
  parser.add_argument('--gaussian-init', action='store_true',
                      help='initializing finnet with gaussians')
  parser.add_argument('--cycarr', action='store_true',
                      help='Whether to use CycArr; otherwise DynArr')
  parser.add_argument('--human', action='store_true',
                      help='Whether to print huamn-friendly output')
  parser.add_argument('--width', type=int, default=0,
                      help='width of the network; default is Inf')
  parser.add_argument('--epochs', type=int, default=24,
                      help='number of training epochs; default 25')
  parser.add_argument('--quiet', action='store_true',
                      help='squash all prints')
  parser.add_argument('--seed', type=int, default=1,
                      help='random seed')
  parser.add_argument('--train-subset-size', type=int, default=None,
                      help='Set a training subset size of Cifar10 instead of the full dataset.')
  parser.add_argument('--test-subset-size', type=int, default=None,
                      help='Set a testing subset size of Cifar10 instead of the full dataset.')
  parser.add_argument('--tie-omegas', action='store_true',
                      help='tie omegas for finite network')

  if arglst is None:
    args = parser.parse_args()
  else:
    args = parser.parse_args(arglst)

  if args.width == 0:
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

  # standard transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])

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

  def get_loss(output, target, reduction='mean'):
    # we only ever use xent
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
              lr_gain=0, scheduler=None):
      '''
      Training loop for one epoch for the model.
      '''
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
          data = data.reshape(data.shape[0], -1)
          optimizer.zero_grad()
          output = model(data)
          if isinstance(model, InfPiMLP):
            # Custom backwards / gclip functions necessary
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
            if Gproj:
              model.Gproj()
            if gclip_sch and gclip_sch.gclip > 0:
              if args.gclip_per_param:
                for param in model.parameters():
                  torch.nn.utils.clip_grad_norm_(param, gclip_sch.gclip)
              else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip_sch.gclip)
          
          # for InfMLP, this calls step on the network
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), elapsed))
            tic = toc
              
          
      train_loss /= len(train_loader.dataset)
      train_acc = correct / len(train_loader.dataset)
      if args.human:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
      
      return losses, train_acc

  
  def test_nn(model, device, test_loader):
      '''
      Test the model on all samples in test_loader.
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
  L = args.depth
  r = args.r
  din = 32**2 * 3
  dout = 10
  
  # always create an InfPiMLP
  infnet = InfPiMLP(d=din, dout=dout, L=L, r=r,
                    first_layer_alpha=args.first_layer_alpha,
                    last_layer_alpha=args.last_layer_alpha,
                    initbuffersize=1000, device=device,
                    bias_alpha=args.bias_alpha,
                    last_bias_alpha=args.last_bias_alpha,
                    _last_layer_grad_no_alpha=args.last_layer_grad_no_alpha,
                    layernorm=args.layernorm,
                    arrbackend=CycArr if args.cycarr else DynArr, maxsize=10**6)

  mynet = None
  if args.width is None:
    optimizer = InfSGD(infnet, lr, wd=wd, momentum=args.momentum,
      first_layer_lr_mult=args.first_layer_lr_mult,
      last_layer_lr_mult=args.last_layer_lr_mult,
      apply_lr_mult_to_wd=not args.no_apply_lr_mult_to_wd,
      bias_lr_mult=args.bias_lr_mult)
    mynet = infnet
  else:
    # if using finnet, sample from infnet
    # this allows us to use the same infnet initialization for many widths
    # and we can therefore show finnet approaching infnet using the same initialization
    if args.last_layer_grad_no_alpha:
      raise NotImplementedError()
    torch.set_default_dtype(torch.float32)
    mynet = infnet.sample(args.width, tieomegas=args.tie_omegas)
    mynet = mynet.cuda()

    # gaussian_init gets rid of all pi-initialization and replaces it with mu-initialization
    # this should be used WITHOUT --Gproj
    if args.gaussian_init:
      for _, lin in mynet.linears.items():
        lin.weight.data[:].normal_()
        lin.weight.data /= lin.weight.shape[1]**0.5 / 2**0.5
    
    if not args.float:
      torch.set_default_dtype(torch.float16)
      mynet = mynet.half()
    paramgroups = []
    # first layer weights
    paramgroups.append({
      'params': [mynet._linears[0].weight],
      'lr': args.first_layer_lr_mult * lr,
      'weight_decay': wd / args.first_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    # last layer weights
    paramgroups.append({
      'params': [mynet._linears[-1].weight],
      'lr': args.last_layer_lr_mult * lr,
      'weight_decay': wd / args.last_layer_lr_mult if args.no_apply_lr_mult_to_wd else wd
    })
    # all other weights
    paramgroups.append({
      'params': [l.weight for l in mynet._linears[1:-1]]
    })    
    # biases
    if mynet._linears[0].bias is not None:
      paramgroups.append({
        'params': [l.bias for l in mynet._linears],
        'lr': args.bias_lr_mult * lr,
        'weight_decay': wd / args.bias_lr_mult if args.no_apply_lr_mult_to_wd else wd
      })
    optimizer = SGD(paramgroups, lr, weight_decay=wd)

  # create all schedulers, for lr and gclip
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
  if args.width is None:
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
        
  train_losses = []
  train_accs = []
  test_losses = []
  test_accs = []

  log_df = []
  if args.save_dir:
    print(f"logs will be saved to {os.path.join(args.save_dir, 'log.df')}")
    if args.save_model:
      print(f"checkpoints saved to {os.path.join(args.save_dir, 'checkpoints')}")
      # save initialization
      model_path = os.path.join(args.save_dir, 'checkpoints')
      os.makedirs(model_path, exist_ok=True)
      model_file = os.path.join(model_path, f'epoch0.th')
      with open(model_file, 'wb') as f:
        torch.save(mynet, f)

  for epoch in range(1, args.epochs+1):
    # main training loop
    epoch_start = time.time()
    losses, train_acc = train_nn(mynet, device, train_loader, optimizer, epoch, Gproj=not args.no_Gproj, gclip_sch=gclip_sch, scheduler=sch)
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
      print(f'model saved at {model_file}')
    
    # output is logged as a dataframe
    log_df.append(
      dict(
        epoch=epoch,
        train_loss=np.mean(train_losses[-1]),
        test_loss=test_losses[-1],
        min_test_loss=min(test_losses),
        min_test_loss_epoch=test_losses.index(min(test_losses)) + 1,
        train_acc=train_acc,
        test_acc=test_accs[-1],
        max_test_acc=max(test_accs),
        max_test_acc_epoch=test_accs.index(max(test_accs)) + 1,
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
  return min(test_accs)

if __name__ == '__main__':
  main()