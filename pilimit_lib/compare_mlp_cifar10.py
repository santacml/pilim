'''
A file specifically for comparing results between pilimit_lib and pilimit_orig.

Do not use unless you are debugging code in the repo.
'''

import torch
from torch import nn
from inf.layers import *
from inf.optim import *
# torch.set_default_dtype(torch.float16)
from torchvision import models
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from examples.networks import InfMLP
from inf.utils import *
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import ssl
from ..pilimit_orig.inf.pimlp import *
from ..pilimit_orig.inf.optim import InfSGD


def main():
  # ssl._create_default_https_context = ssl._create_unverified_context
  sns.set()
  def tight_layout(plt):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])

  class Timer():
    def __init__(self):
      self.lasttime = time.time()

    def log(self, msg):
      newtime = time.time()
      print(msg, newtime - self.lasttime)
      self.lasttime = newtime

    def track(self):
      newtime = time.time()
      diff = newtime - self.lasttime
      self.lasttime = newtime

      return diff

    def reset(self):
      newtime = time.time()
      self.lasttime = newtime

  # import os, sys
  # sys.path.insert(1, os.path.split(os.path.dirname(os.path.realpath(sys.argv[0])))[0])
  # sys.path.insert(1, "C:\repos\pilimit\inf")


  # 0/0


  torch.manual_seed(33)
  np.random.seed(33)
  device="cuda"


  batch_size = 1

  transform_list = []
  transform_list.extend([transforms.ToTensor()])

  transform_list.extend([transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])
  transform = transforms.Compose(transform_list)

  trainset = datasets.CIFAR10(root=".", train=True,
                                          download=True, transform=transform)

  np.random.seed(0) # reproducability of subset
  indices = np.random.choice(range(50000), size=batch_size, replace=False).tolist()
  trainset = data_utils.Subset(trainset, indices)
  print("Using subset of", len(trainset), "training samples")
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

  testset = datasets.CIFAR10(root=".", train=False,
                                        download=True, transform=transform)
  np.random.seed(0) # reproducability of subset
  indices = np.random.choice(range(50000), size=batch_size, replace=False).tolist()
  testset = data_utils.Subset(testset, indices)
  print("Using subset of", len(testset), "testing samples")
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)








  d_in = 32*32*3
  d_out = 10
  r = 200
  epoch = 50
  L = 1 
  layernorm = False

  first_layer_lr_mult = 1
  last_layer_lr_mult = 1
  bias_lr_mult = .1
  first_layer_alpha = 1
  bias_alpha = 1
  last_layer_alpha = 1
  
  # first_layer_alpha = float(5)
  # bias_alpha = 0
  # last_layer_alpha = 5
  # first_layer_lr_mult = .5
  # last_layer_lr_mult = 5
  # bias_lr_mult = 0
  no_apply_lr_mult_to_wd = True



  net = InfMLP(d_in, d_out, r, L, device=device, first_layer_alpha=first_layer_alpha, bias_alpha=bias_alpha, last_layer_alpha=last_layer_alpha, layernorm=layernorm)

  # orig_net = InfPiMLP(d=d_in, dout=d_out, L=L+1, r=r, bias_alpha=1, initbuffersize=1000, first_layer_alpha=1, last_layer_alpha=10, _last_layer_grad_no_alpha=True)
  orig_net = InfPiMLP(d=d_in, dout=d_out, L=L+1, r=r, first_layer_alpha=first_layer_alpha, bias_alpha=bias_alpha, initbuffersize=1000, last_layer_alpha=last_layer_alpha, layernorm=layernorm, device=device)

  with torch.no_grad():
      for l in range(1, L+3):
          if l == 1:
              orig_net.As[l] = net.layers[0].A.clone()
              if bias_alpha > 0: orig_net.biases[l] = net.layers[0].bias.clone()
          else:
              orig_net.As[l].a[:] = net.layers[l-1].A.detach().clone()
              orig_net.As[l].arr.requires_grad = False
              orig_net.Amult[l].a[:] = net.layers[l-1].Amult.detach().clone()
              orig_net.Amult[l].arr.requires_grad = False
              orig_net.Bs[l].a[:] = net.layers[l-1].B.detach().clone()
              orig_net.Bs[l].arr.requires_grad = False
              if bias_alpha > 0: orig_net.biases[l][:] = net.layers[l-1].bias.clone()

  # with torch.no_grad():
  #     for l in range(1, L+3):
  #         if l == 1:
  #             net.layers[0].A[:] = orig_net.As[l].clone()
  #             net.layers[0].bias[:] = orig_net.biases[l].clone()
  #         else:
  #             net.layers[l-1].A[:] = orig_net.As[l].a[:].detach().clone()
  #             net.layers[l-1].Amult[:] = orig_net.Amult[l].a[:].detach().clone()
  #             net.layers[l-1].B[:] = orig_net.Bs[l].a[:].detach().clone()
  #             net.layers[l-1].bias[:] = orig_net.biases[l][:].clone()

  # 0/0

  # net.apply(pi_init)

  # model_copy = type(net)(net.d_in, net.d_out, net.r, net.L, device=device)
  # model_copy.load_state_dict(net.state_dict()) # works if register buffer is used where appropriate

  # stores even nested params!
  # print(net.layers[1].bias.omega == model_copy.layers[1].bias.omega)

  # 0/0

  orig_losses = []
  lr = .001
  gclip = 0
  # wd=0
  gclip = 0.5
  wd=0.1
  gclip_per_param = True
  step = True


  orig_forwards = []
  orig_backwards = []
  orig_gclips = []
  orig_steps = []
  timer = Timer()
  optimizer = InfSGD(orig_net, lr, wd=wd,
    first_layer_lr_mult=first_layer_lr_mult,
    last_layer_lr_mult=last_layer_lr_mult,
    bias_lr_mult=bias_lr_mult,
    apply_lr_mult_to_wd=not no_apply_lr_mult_to_wd)
  for i in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device).type(torch.get_default_dtype()), target.to(device)
      data = data.reshape(data.shape[0], -1)
      X = data
      y = target

      timer.reset()
      orig_net.zero_grad()
      timer.reset()
      yhat = orig_net(X)
      # print(yhat.shape)
      oh_target = target.new_zeros(target.shape[0], 10).type(torch.get_default_dtype())
      oh_target.scatter_(1, target.unsqueeze(-1), 1)
      oh_target -= 0.1
      loss = F.mse_loss(yhat, oh_target, reduction="mean")

      orig_forwards.append(timer.track())
      # loss = 0.5 * ((yhat - y)**2).sum()

      # if i % 10 == 0: 
      print(i, loss.item())
      # 0/0
        # print("orig", torch.cuda.memory_reserved() / 1e9, torch.cuda.max_memory_reserved()  / 1e9)

      # dloss = yhat - y
      timer.reset()
      # orig_net.backward(dloss)
      loss.backward()
      orig_net.backward(yhat.grad)
      orig_backwards.append(timer.track())
      timer.reset()
      if gclip: 
        orig_net.gclip(gclip, per_param=gclip_per_param)
      orig_gclips.append(timer.track())
      
      # orig_losses.append(orig_net.dAs[2][0].detach().cpu().numpy())
      # orig_losses.append(orig_net.dbiases[2].detach().numpy())
      # print(orig_net.dbiases[2].detach().numpy().shape)

      timer.reset()
      # orig_net.step(lr, wd=wd, apply_lr_mult_to_wd=False)
      if step: optimizer.step()
      orig_steps.append(timer.track())
      orig_losses.append(loss.item())
      # orig_losses.append(orig_net.gs[3].detach().numpy())


  # print([type(n) for n in net.parameters()])
  # 0/0

  # del orig_net
  torch.cuda.empty_cache()

  forwards = []
  backwards = []
  gclips = []
  steps = []

  net.train()
  # gclip = 0
  # epoch = 3
  losses = []
  paramgroups = []
  # first layer weights
  paramgroups.append({
    'params': [net.layers[0].A],
    'lr': first_layer_lr_mult * lr,
    'weight_decay': wd / first_layer_lr_mult if no_apply_lr_mult_to_wd else wd
  })
  if net.layers[0].bias is not None:
    paramgroups.append({
      'params': [l.bias for l in net.layers],
      'lr': bias_lr_mult * lr,
      'weight_decay': wd / bias_lr_mult if no_apply_lr_mult_to_wd else wd
    })
  paramgroups.append({
    'params': [l.Amult for l in net.layers[1:-1]],
  })
  paramgroups.append({
    'params': [net.layers[-1].Amult],
    'lr': last_layer_lr_mult * lr,
    'weight_decay': wd / last_layer_lr_mult if no_apply_lr_mult_to_wd else wd
  })
  paramgroups.append({
    'params': [l.A for l in net.layers[1:]],
  })
  paramgroups.append({
    'params': [l.B for l in net.layers[1:]],
  })
  # optimizer = PiSGD(net.parameters(), lr = lr, weight_decay=wd)
  optimizer = PiSGD(paramgroups, lr = lr, weight_decay=wd)
  for epoch in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device).type(torch.get_default_dtype()), target.to(device)
      data = data.reshape(data.shape[0], -1)
      labels = target

      timer.reset()
      optimizer.zero_grad()
      net.zero_grad()
      
      timer.reset()
      prediction = net(data)
      forwards.append(timer.track())
      
      # loss = torch.sum((prediction - labels)**2)* .5
      oh_target = target.new_zeros(target.shape[0], 10).type(torch.get_default_dtype())
      oh_target.scatter_(1, target.unsqueeze(-1), 1)
      oh_target -= 0.1
      loss = F.mse_loss(prediction, oh_target, reduction="mean")
    
      if epoch % 10 == 0: 
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # print("new", torch.cuda.memory_reserved() / 1e9, torch.cuda.max_memory_reserved()  / 1e9)
      
      timer.reset()
      loss.backward()
      backwards.append(timer.track())

      # with torch.no_grad():
      #     net.layers[1].bias.grad*=100

      timer.reset()
      if gclip:
          store_pi_grad_norm_(net.modules())
          if gclip_per_param:
            for param in net.parameters():
                clip_grad_norm_(param, gclip)
          else:
            clip_grad_norm_(net.parameters(), gclip)
      gclips.append(timer.track())

      # losses.append(net.layers[1].A.pi.grad.detach().cpu().numpy())
      # losses.append(net.layers[1].bias.grad.detach().numpy())
      timer.reset()
      if step: optimizer.step()
      steps.append(timer.track())

      # print(net.layers[1])
      losses.append(loss.item())
      # losses.append(net.layers[2](net.layers[1](net.layers[0](data))).detach().numpy())
  # 0/0

  orig_losses = np.array(orig_losses)
  losses = np.array(losses)
  print(orig_losses - losses)
  print(np.mean(orig_losses - losses))
  # print(orig_losses[0])
  # print(losses[0])

  def mean_total_time(msg, arr1, arr2):
      arr1 = np.array(arr1)
      arr2 = np.array(arr2)
      print(msg, "total orig minus new", np.sum(arr1-arr2), "orig mean", np.mean(arr1), "new mean", np.mean(arr2), "new minus orig", np.mean(arr2-arr1))

  mean_total_time("forwards", orig_forwards, forwards)
  mean_total_time("backwards", orig_backwards, backwards)
  mean_total_time("gclips", orig_gclips, gclips)
  mean_total_time("steps", orig_steps, steps)



  with torch.no_grad():
      for l in range(1, L+3):
          if l == 1:
              print("A", (orig_net.As[l] - net.layers[0].A.clone()).abs().sum())
              print("Agrad", (orig_net.dAs[l] - net.layers[0].A.grad.clone()).abs().sum())
              # print(orig_net.dAs[l][0, :10])
              # print(net.layers[0].A.grad[0, :10])
              if bias_alpha > 0: 
                print(l, "bias", (orig_net.biases[l] - net.layers[0].bias.clone()).abs().sum())
          else:
              print("A", (orig_net.As[l].a[:] - net.layers[l-1].A.detach().clone()).abs().sum())
              # print(orig_net.As[l].a[:])
              # print(net.layers[l-1].A)
              # print(orig_net.dAs[l][0][0, :10])
              # print(net.layers[l-1].A.pi.grad[0, :10])
              print("Amult", (orig_net.Amult[l].a[:] - net.layers[l-1].Amult.detach().clone()).abs().sum())
              print("B", (orig_net.Bs[l].a[:] - net.layers[l-1].B.detach().clone()).abs().sum())
              if bias_alpha > 0: 
                  print(l, "bias", (orig_net.biases[l][:] - net.layers[l-1].bias.clone()).abs().sum())
                  # print(orig_net.biases[l][:])
                  # print(net.layers[l-1].bias.clone())
                  # print(l, "dbias", (orig_net.dbiases[l][:] - net.layers[l-1].bias.grad.clone()).abs().sum(), (net.layers[l-1].bias.grad.clone()).abs().sum())
                  # print(orig_net.dbiases[l][:].dtype,  net.layers[l-1].bias.grad.dtype)

                  # oldbias = orig_net.biases[l][:] - .00001*orig_net.dbiases[l][:]
                  # newbias = net.layers[l-1].bias.clone() - .00001*net.layers[l-1].bias.grad.clone().half()
                  # print("manual", (newbias-oldbias).abs().sum())

      # print("g1", (orig_net.gs[2] - net.gs1.detach().clone()).abs().sum())
      # print("orig s", orig_net.ss[2])

  # print(orig_net.As[3].a[-10:, :])
  # print(net.layers[2].A[-10:, :])

  fig = plt.figure(figsize=(5,5))

  plt.plot(orig_losses, label="Original Library Loss", linestyle="solid")
  plt.plot(losses, label="New Library Loss", linestyle="dashed")
  plt.legend()
  plt.title(f'Accuracy on Dummy Data, Original vs. Inf Library')
  plt.ylabel('Accuracy')
  plt.xlabel('Train Epochs')
  plt.show()


if __name__ == "__main__":
  main()