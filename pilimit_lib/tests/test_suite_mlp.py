from examples.networks import *
from inf.optim import *
from inf.utils import *
from tqdm.notebook import tqdm
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from copy import deepcopy
sns.set()
def tight_layout(plt):
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

X = torch.linspace(-np.pi, np.pi).reshape(-1, 1)
y = torch.sin(X) #.reshape(-1)
X = torch.cat([X, torch.ones_like(X)], dim=1)

# plt.plot(X[:, 0], y)

lrs = [0.1]
momenta = [0, 0.6]
# momenta = [0]
batches = [32]
lns = [False, True]
# lns = [False]
wds = [0, 0.1]
# wds = [0]
bias_alphas = [0, 1]
# bias_alphas = [0]
gclips = [0, 0.1]
gclip_perp_params = [False, True]

kws = dict(
  lrs = [0.1],
  momenta = [0],
  batches = [32],
#   lns = [True],
  lns = [True],
  wds = [0, 0.1],
#   wds = [0.001],
  bias_alphas = [0, 1],
  gclips = [0, 0.1],
  gclip_perp_params = [False, True],
  damps = [0],
  last_layer_alpha = [1, 10],
  # bias_alphas = [0],
)

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
  
  
  paramgroups = []
  # first layer weights
  paramgroups.append({
    'params': [net.layers[0].A],
    'lr': first_layer_lr_mult * lr
  })
  # biases
  if net.layers[0].bias is not None:
    paramgroups.append({
      'params': [l.bias for l in net.layers],
      'lr': bias_lr_mult * lr
    })
  # all other weights
  paramgroups.append({
    'params': [l.Amult for l in net.layers[1:]],
  })
  paramgroups.append({
    'params': [l.A for l in net.layers[1:]],
  })
  paramgroups.append({
    'params': [l.B for l in net.layers[1:]],
  })
  optimizer = PiSGD(paramgroups, lr, weight_decay=wd, momentum=momentum, dampening=dampening)


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
    loss = torch.nn.functional.mse_loss(yhat, y)
#     print(i, loss.item())
    losses.append(loss.item())
    # dloss = (yhat - y) / len(X)
    loss.backward()
    # net.backward(yhat.grad)
    if gclip > 0:
    #   net.gclip(gclip, per_param=gclip_per_param)
        store_pi_grad_norm_(net.modules())
        if gclip_per_param:
            for param in net.parameters():
                clip_grad_norm_(param, gclip)
        else:
            clip_grad_norm_(net.parameters(), gclip)
        
    # lr_ = lr if t < lr_drop_T else lr * lr_drop_ratio
    # net.step(lr_, momentum=momentum, wd=wd, first_layer_lr_mult=first_layer_lr_mult, bias_lr_mult=bias_lr_mult,
    # dampening=dampening)
    if t == lr_drop_T:
      for group in optimizer.param_groups:
        group['lr'] *= lr_drop_ratio
    optimizer.step()
  return losses

def record_pgdfin(infnet, X, y, lr, width, T, cuda=False, center=False, wd=0, momentum=0, dampening=0, seed=None, batchsize=None, lr_drop_T=float('inf'), lr_drop_ratio=1,
first_layer_lr_mult=1, bias_lr_mult=1, gclip=0, gclip_per_param=False,
fincls=FinPiMLPSample):
  if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)

  finnet = FinPiMLPSample(infnet, width)
  
  paramgroups = []
  # first layer weights
  paramgroups.append({
    'params': [finnet.layers[0].weight],
    'lr': first_layer_lr_mult * lr
  })
  # biases
  if finnet.layers[0].bias is not None:
    paramgroups.append({
      'params': [l.bias for l in finnet.layers],
      'lr': bias_lr_mult * lr
    })
  # all other weights
  paramgroups.append({
    'params': [l.weight for l in finnet.layers[1:]],
  })
  optimizer = PiSGD(paramgroups, lr, weight_decay=wd, momentum=momentum, dampening=dampening)
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
    loss = torch.nn.functional.mse_loss(yhat, y)
#     dloss = yhat - y
#     yhat.backward(dloss)
    loss.backward()
    # finnet.Gproj()
    if gclip > 0:
      if gclip_per_param:
        for param in finnet.parameters():
        #   torch.nn.utils.clip_grad_norm_(param, gclip)
            clip_grad_norm_(param, gclip)
      else:
        # torch.nn.utils.clip_grad_norm_(finnet.parameters(), gclip)
        clip_grad_norm_(finnet.parameters(), gclip)
    # lr_ = lr if t < lr_drop_T else lr * lr_drop_ratio
    if t == lr_drop_T:
      for group in optimizer.param_groups:
        group['lr'] *= lr_drop_ratio
    optimizer.step()
    losses.append(loss.item())

    if np.inf in losses or np.nan in losses or not all(loss < 1e5 for loss in losses):
      print("Got nan loss, breaking")
      break

  return losses

def getlosses(L, kws, widths=None, rank=2, T=100, lr_drop_T=3, lr_drop_ratio=0.5, bias_lr_mult=4, first_layer_lr_mult=5, first_layer_alpha=0.5):
  losses = {}
  if widths is None:
    # widths = [10000, 20000]
    widths = [1000]
  # combos = product(lrs, momenta, batches, lns, wds, bias_alphas, gclips, gclip_perp_params, damps)
  combos = product(*kws.values())
  for i, comb in list(enumerate(combos)):
    (lr, mom, batch, ln, wd, bias_alpha, gclip, gclip_per_param, damp, last_layer_alpha) = comb
    if gclip == 0 and gclip_per_param:
      continue
    fig = plt.figure()
    plt.ylabel('loss')
    plt.xlabel('iter')
    torch.manual_seed(1)
    np.random.seed(1)
    # infnet = InfPiMLP(d=2, dout=1, L=L, r=rank, initbuffersize=1000,
    #                   bias_alpha=bias_alpha, quiet=True, layernorm=ln,
    #                  first_layer_alpha=first_layer_alpha,
    #                  last_layer_alpha=last_layer_alpha)
    infnet = InfMLP(2, 1, rank, L,
                    bias_alpha=bias_alpha,
                    first_layer_alpha=first_layer_alpha,
                    last_layer_alpha=last_layer_alpha,
                    layernorm=ln, device="cuda")

    # need to do finite net first, since record_pgdlim changes infnet
    for width in widths:
      losses[(comb, width)] = record_pgdfin(infnet, X, y, lr, width, T,
                                             cuda=True, center=True,
                                             batchsize=batch, seed=0, momentum=mom, dampening=damp,
                                             wd=wd, lr_drop_T=lr_drop_T, lr_drop_ratio=lr_drop_ratio,
                                             bias_lr_mult=bias_lr_mult, first_layer_lr_mult=first_layer_lr_mult,
                                             gclip=gclip, gclip_per_param=gclip_per_param)
      arr = np.copy(losses[(comb, width)])
      arr = np.nan_to_num(arr, nan=1e10, posinf=1e10, neginf=-1e10)
      arr[arr>10] = np.nan
      plt.plot(arr, label=str(width))
  #   if (lr, batch, np.inf) not in losses:
    losses[(comb, np.inf)] = record_pgdlim(infnet, X, y, lr, T,
                                           cuda=True, batchsize=batch, seed=0,
                                           momentum=mom, wd=wd, dampening=damp,
                                           lr_drop_T=lr_drop_T, lr_drop_ratio=lr_drop_ratio,
                                           bias_lr_mult=bias_lr_mult, first_layer_lr_mult=first_layer_lr_mult,
                                           gclip=gclip, gclip_per_param=gclip_per_param)
    arr = np.copy(losses[(comb, np.inf)])
    arr = np.nan_to_num(arr, nan=1e10, posinf=1e10, neginf=-1e10)
    arr[arr>10] = np.nan
    plt.plot(arr, '--', label='inf')
    plt.legend()
  #   plt.title(f'lr={lr}, mom={mom}, bsz={batch}, ln={ln}, wd={wd}, bias_alpha={bias_alpha}, '
  #             f'gclip={gclip}, per_param={gclip_per_param}')
    print(dict(zip(kws.keys(), comb)))
    plt.show()
  return losses

_ = getlosses(0, kws)