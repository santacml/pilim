'''
A file specifically for comparing results between pilimit_lib and pilimit_orig.

Do not use unless you are debugging code in the repo.
'''

import torch
from torch import nn
from inf.layers import *
from inf.optim import *
torch.set_default_dtype(torch.float16)
from torchvision import models
from examples.networks import InfMLP
from inf.utils import *
import time
import matplotlib.pyplot as plt
import seaborn as sns
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


import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)
from pilimit_orig.inf.pimlp import *


# 0/0


torch.manual_seed(33)
np.random.seed(33)
device="cuda"

data = torch.linspace(-np.pi, np.pi, device=device).reshape(-1, 1)
labels = torch.sin(data) #.reshape(-1)
data = torch.cat([data, torch.ones_like(data, device=device)], dim=1)
# print(data.shape)
# print(labels.shape)
# 0/0

d_in = 2
d_out = 1
r = 200
epoch = 100
L = 1
layernorm = False

first_layer_alpha = .5
# bias_alpha = 0.5
# last_layer_alpha = 100
# first_layer_alpha = 1
bias_alpha = .5
last_layer_alpha =  2
first_layer_lr_mult = 1
last_layer_lr_mult = 1
bias_lr_mult = 1
no_apply_lr_mult_to_wd = True


net = InfMLP(d_in, d_out, r, L, device=device, first_layer_alpha=first_layer_alpha, bias_alpha=bias_alpha, last_layer_alpha=last_layer_alpha, layernorm=layernorm)

# orig_net = InfPiMLP(d=d_in, dout=d_out, L=L+1, r=r, bias_alpha=1, initbuffersize=1000, first_layer_alpha=1, last_layer_alpha=10, _last_layer_grad_no_alpha=True)
orig_net = InfPiMLP(d=d_in, dout=d_out, L=L+1, r=r, first_layer_alpha=first_layer_alpha, bias_alpha=bias_alpha, initbuffersize=1000, last_layer_alpha=last_layer_alpha, layernorm=layernorm, device=device)

with torch.no_grad():
    for l in range(1, L+3):
        if l == 1:
            orig_net.As[l] = net.layers[0].A.clone()
            orig_net.biases[l] = net.layers[0].bias.clone()
        else:
            orig_net.As[l].a[:] = net.layers[l-1].A.detach().clone()
            orig_net.As[l].arr.requires_grad = False
            orig_net.Amult[l].a[:] = net.layers[l-1].Amult.detach().clone()
            orig_net.Amult[l].arr.requires_grad = False
            orig_net.Bs[l].a[:] = net.layers[l-1].B.detach().clone()
            orig_net.Bs[l].arr.requires_grad = False
            orig_net.biases[l][:] = net.layers[l-1].bias.clone()

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
lr = .1
gclip = 0.5
# gclip = 0
wd=.3
gclip_per_param = True


X = data
y = labels

orig_forwards = []
orig_backwards = []
orig_gclips = []
orig_steps = []
timer = Timer()
# optimizer = InfSGD(infnet, lr, wd=wd,
#   first_layer_lr_mult=args.first_layer_lr_mult,
#   last_layer_lr_mult=args.last_layer_lr_mult,
#   bias_lr_mult=args.bias_lr_mult,
#   apply_lr_mult_to_wd=not args.no_apply_lr_mult_to_wd)
for i in range(epoch):
    timer.reset()
    orig_net.zero_grad()
    timer.reset()
    yhat = orig_net(X).detach()
    orig_forwards.append(timer.track())
    loss = 0.5 * ((yhat - y)**2).sum()

    if i % 10 == 0: 
      print(i, loss.item())
      # print("orig", torch.cuda.memory_reserved() / 1e9, torch.cuda.max_memory_reserved()  / 1e9)

    dloss = yhat - y
    timer.reset()
    orig_net.backward(dloss)
    orig_backwards.append(timer.track())
    timer.reset()
    if gclip: 
      orig_net.gclip(gclip, per_param=gclip_per_param)
    orig_gclips.append(timer.track())
    
    # orig_losses.append(orig_net.dAs[2][0].detach().cpu().numpy())
    # orig_losses.append(orig_net.dbiases[2].detach().numpy())
    # print(orig_net.dbiases[2].detach().numpy().shape)

    timer.reset()
    orig_net.step(lr, wd=wd, apply_lr_mult_to_wd=False)
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
    'lr': bias_lr_mult * lr
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
    timer.reset()
    optimizer.zero_grad()
    net.zero_grad()
    
    timer.reset()
    prediction = net(data)
    forwards.append(timer.track())
    
    loss = torch.sum((prediction - labels)**2)* .5
   
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
    optimizer.step()
    steps.append(timer.track())

    # print(net.layers[1])
    losses.append(loss.item())
    # losses.append(net.layers[2](net.layers[1](net.layers[0](data))).detach().numpy())


orig_losses = np.array(orig_losses)
losses = np.array(losses)
# print(orig_losses - losses)
# print(np.mean(orig_losses - losses))
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


def print_stats(orig, new, msg=""):
  print(msg, (orig != new).sum(), (orig - new).abs().sum())

with torch.no_grad():
    for l in range(1, L+3):
        if l == 1:
            print_stats(orig_net.As[l], net.layers[0].A.clone())
            print_stats(orig_net.biases[l], net.layers[0].bias.clone())
        else:
            # print((orig_net.As[l].a[:] != net.layers[l-1].A.detach().clone()).sum())
            # print(orig_net.As[l].a[:])
            # print(net.layers[l-1].A.detach().clone())
            print_stats(orig_net.As[l].a[:], net.layers[l-1].A.detach().clone())
            print_stats(orig_net.Amult[l].a[:], net.layers[l-1].Amult.detach().clone())
            print_stats(orig_net.Bs[l].a[:], net.layers[l-1].B.detach().clone())
            print_stats(orig_net.biases[l][:], net.layers[l-1].bias.clone())

fig = plt.figure(figsize=(5,5))

plt.plot(orig_losses, label="Original Library Loss", linestyle="solid")
plt.plot(losses, label="New Library Loss", linestyle="dashed")
plt.legend()
plt.title(f'Accuracy on Dummy Data, Original vs. Inf Library')
plt.ylabel('Accuracy')
plt.xlabel('Train Epochs')
plt.show()