import unittest
import numpy as np
from tqdm import tqdm
from inf.pimlp import *
# from inf.optim import *
from itertools import product

def record_pgdlim(net, X, y, lr, T, cuda=False, seed=None, batchsize=None,
      momentum=0, dampening=0, wd=0, lr_drop_T=float('inf'), lr_drop_ratio=1,
      first_layer_lr_mult=1, bias_lr_mult=1, gclip=0, gclip_per_param=False):
  if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
  if cuda:
    net = net.cuda()
    X = X.cuda()
    y = y.cuda()
  losses = []
  Xall = X
  yall = y
  for t in range(T):
    if batchsize is None:
      X, y = Xall, yall
    else:
      batchidxs = np.random.choice(len(Xall), batchsize, replace=False)
#       print(batchidxs)
      X = Xall[batchidxs]
      y = yall[batchidxs]
    net.zero_grad()
    yhat = net(X)
    # loss = 0.5 * ((yhat - y)**2).mean()
    loss = F.mse_loss(yhat, y)
#     print(i, loss.item())
    losses.append(loss.item())
    # dloss = (yhat - y) / len(X)
    loss.backward()
    net.backward(yhat.grad)
    if gclip > 0:
      net.gclip(gclip, per_param=gclip_per_param)
    lr_ = lr if t < lr_drop_T else lr * lr_drop_ratio
    net.step(lr_, momentum=momentum, wd=wd, first_layer_lr_mult=first_layer_lr_mult, bias_lr_mult=bias_lr_mult,
    dampening=dampening)
  return losses

def record_pgdfin(infnet, X, y, lr, width, T, cuda=False, center=False, wd=0, momentum=0, dampening=0, seed=None, batchsize=None, lr_drop_T=float('inf'), lr_drop_ratio=1,
first_layer_lr_mult=1, bias_lr_mult=1, gclip=0, gclip_per_param=False,
fincls=FinPiMLP):
  if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
  finnet = infnet.sample(width, fincls=fincls)
  
  paramgroups = []
  # first layer weights
  paramgroups.append({
    'params': [finnet._linears[0].weight],
    'lr': first_layer_lr_mult * lr
  })
  # biases
  if finnet._linears[0].bias is not None:
    paramgroups.append({
      'params': [l.bias for l in finnet._linears],
      'lr': bias_lr_mult * lr
    })
  # all other weights
  paramgroups.append({
    'params': [l.weight for l in finnet._linears[1:]],
  })
  optimizer = optim.SGD(paramgroups, lr, weight_decay=wd, momentum=momentum, dampening=dampening)
  losses = []
  if cuda:
    finnet = finnet.cuda()
    X = X.cuda()
    y = y.cuda()
  if center:
    frznet = deepcopy(finnet)
    frznet.requires_grad_(False)
  Xall = X
  yall = y
  for t in range(T):
    if batchsize is None:
      X, y = Xall, yall
    else:
      batchidxs = np.random.choice(len(Xall), batchsize, replace=False)
#       print(batchidxs)
      X = Xall[batchidxs]
      y = yall[batchidxs]
    finnet.zero_grad()
    yhat = finnet(X) #.reshape(-1)
#     print(yhat.shape, y.shape)
    if center:
      yhat -= frznet(X) #.reshape(-1)
    # loss = 0.5 * ((yhat - y)**2).mean()
    loss = F.mse_loss(yhat, y)
#     dloss = yhat - y
#     yhat.backward(dloss)
    loss.backward()
    finnet.Gproj()
    if gclip > 0:
      if gclip_per_param:
        for param in finnet.parameters():
          torch.nn.utils.clip_grad_norm_(param, gclip)
      else:
        torch.nn.utils.clip_grad_norm_(finnet.parameters(), gclip)
    # lr_ = lr if t < lr_drop_T else lr * lr_drop_ratio
    if t == lr_drop_T:
      for group in optimizer.param_groups:
        group['lr'] *= lr_drop_ratio
    optimizer.step()
    losses.append(loss.item())
  return losses

class Test_PiNet(unittest.TestCase):
  def test_fin_vs_inf(self):
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.linspace(-np.pi, np.pi).reshape(-1, 1)
    y = torch.sin(X) #.reshape(-1)
    X = torch.cat([X, torch.ones_like(X)], dim=1)
    lr = 0.2
    T = 100
    batchsize = 64
    infnet = InfPiMLP(d=2, dout=1, L=2, r=5, initbuffersize=1000, bias_alpha=0)
    # print(infnet.biases)
    # finlosses = record_pgdfin(infnet, X, y, lr=lr, width=10000, T=T, seed=1, cuda=True, batchsize=batchsize)
    inflosses = record_pgdlim(infnet, X, y, lr=lr, T=T, seed=1, batchsize=batchsize, cuda=True)
    # print(infnet.biases)

    lrs = [0.2, 0.1, 0.05, 0.02, 0.01]
    batches = list(2**np.arange(7)) + [100]

    losses = {}

    # i = 1
    T = 100
    L = 2
    rank = 2
    width = 1000
    for batch, lr in tqdm(list(product(batches, lrs))):
      # for lr in lrs:
      torch.manual_seed(1)
      np.random.seed(1)
      infnet = InfPiMLP(d=2, dout=1, L=L, r=rank, initbuffersize=1000, bias_alpha=1, quiet=True)
      # need to do finite net first, since record_pgdlim changes infnet
      if (lr, batch, width) not in losses:
        fin = losses[(lr, batch, width)] = record_pgdfin(
            infnet, X, y, lr, width, T, cuda=True, center=True, batchsize=batch, seed=0)
      if (lr, batch, np.inf) not in losses:
        inf = losses[(lr, batch, np.inf)] = record_pgdlim(
            infnet, X, y, lr, T, cuda=True, batchsize=batch, seed=0)
      # i += 1
      fin = np.array(losses[(lr, batch, width)])
      inf = np.array(losses[(lr, batch, np.inf)])
      with self.subTest('', batch=batch, lr=lr):
        self.assertTrue(np.linalg.norm(fin - inf) / np.linalg.norm(inf) < 0.03)

if __name__ == '__main__':
    unittest.main()