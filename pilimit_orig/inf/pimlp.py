import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from inf.dynamicarray import DynArr, CycArr
from inf.utils import safe_sqrt, safe_acos, F00ReLUsqrt, F11ReLUsqrt, F02ReLUsqrt, VReLUmatrix, ABnorm

# counter = 0
# counterlist = range(242,246)
# counterlist = range(236,237)
# counterlist = range(2,6)
# endcounter = 4

class MyLinear(nn.Linear):

  def __init__(self, *args, **kw):
    self.device = kw.pop('device', 'cpu')
    self.bias_alpha = kw.pop('bias_alpha', 1)
    super().__init__(*args, **kw)

  def reset_parameters(self) -> None:
    self.to(self.device)
    super().reset_parameters()

  def forward(self, input):
    return F.linear(input, self.weight,
      self.bias * self.bias_alpha if self.bias is not None else self.bias)

def divbystd(x):
  return x / (1e-5 + torch.norm(x, dim=1, keepdim=True))

class FinPiMLP(nn.Module):
  '''
  Note:
  This network is designed so that the (pre)activations have coordinates
  of order width^-1/2, so that no input/output multipliers are needed.
  This means for normalization layers, we need to divide by width^1/2.
  '''

  def __init__(self, datadim, width, ncls, L, bias_alpha=0, last_bias_alpha=None, nonlin=nn.ReLU, device='cpu', lincls=MyLinear,
      first_layer_alpha=1, last_layer_alpha=1, layernorm=False):
    super().__init__()
    self.datadim = datadim
    self.width = width
    self.ncls = ncls
    self.L = L
    self.device = device
    self.nonlin = nonlin()
    self.linears = {}
    self.first_layer_alpha = first_layer_alpha
    self.last_layer_alpha = last_layer_alpha
    self.layernorm = layernorm
    bias = bias_alpha != 0
    self.bias_alpha = bias_alpha
    if last_bias_alpha is None:
      last_bias_alpha = bias_alpha
    # elif last_bias_alpha != bias_alpha:
    #   raise NotImplementedError("this branch is a bit stale; check that it's implemented for finnet/maml/cifar10")
    self.last_bias_alpha = last_bias_alpha
    # _linears is used purely to register modules
    self._linears = nn.ModuleList()
    for l in range(1, L+2):
      if l == 1:
        self.linears[l] = lincls(datadim, width, bias=bias, bias_alpha=bias_alpha, device=self.device)
      elif l == L+1:
        self.linears[l] = lincls(width, ncls, bias=bias, bias_alpha=last_bias_alpha, device=self.device)
      else:
        self.linears[l] = lincls(width, width, bias=bias, bias_alpha=bias_alpha, device=self.device)
      self._linears.append(self.linears[l])

  def initialize(self, infnet, keepomegas=False, tieomegas=False):
    L = infnet.L
    self.r = r = infnet.r
    self.first_layer_alpha = infnet.first_layer_alpha
    self.last_layer_alpha = infnet.last_layer_alpha
    n = self.width
    As = {l: A.a for l, A in infnet.As.items() if l != 1}
    As[1] = infnet.As[1]
    Amult = {l: Amult.a for l, Amult in infnet.Amult.items() if l != 1}
    Bs = {l: B.a for l, B in infnet.Bs.items()}
    biases = infnet.biases
    dtype = infnet.As[2].a.dtype
    if not keepomegas:
      self.omegas = omegas = {}
      self.Gcovinvs = {}
      if tieomegas:
        orig_omega = torch.randn(n, r, device=self.device).float()
      for l in range(1, L+1):
        if tieomegas:
          omegas[l] = orig_omega.clone()
        else:
          omegas[l] = torch.randn(n, r, device=self.device).float()
        self.Gcovinvs[l] = torch.inverse(omegas[l].T @ omegas[l]).to(dtype)
        omegas[l] = omegas[l].to(dtype)
    else:
      omegas = self.omegas
    
    with torch.no_grad():
      for l in range(1, L+2):
        if l == 1:
          self.linears[l].weight[:] = n**-0.5 * omegas[l] @ As[1].T.to(dtype)
        elif l == L+1:
          self.linears[l].weight[:] = n**-0.5 * (Amult[l] * As[l].T).to(dtype) @ (self.nonlin(omegas[l-1] @ Bs[l].T)).T.to(dtype)
        else:
          self.linears[l].weight[:] = n**-1.0 * omegas[l] @ (Amult[l] * As[l].T).to(dtype) @ (self.nonlin(omegas[l-1] @ Bs[l].T)).T.to(dtype)
        if biases:
          if l == L+1:
            self.linears[l].bias[:] = biases[l].to(dtype)
          else:
            self.linears[l].bias[:] = n**-0.5 * omegas[l] @ biases[l].to(dtype)
    
  def Gproj(self):
    if self.r >= self.width:
      return
    # import pdb; pdb.set_trace()
    with torch.no_grad():
      for l in range(1, self.L+1):
        grad = self.linears[l].weight.grad
        om = self.omegas[l]
        self.linears[l].weight.grad[:] = om @ (self.Gcovinvs[l] @ (om.T @ grad))
        if self.linears[l].bias is not None:
          self.linears[l].bias.grad[:] = om @ (self.Gcovinvs[l] @ (om.T @ self.linears[l].bias.grad))

  def cuda(self):
    if hasattr(self, 'omegas'):
      for l in range(1, self.L+1):
        self.omegas[l] = self.omegas[l].cuda()
        self.Gcovinvs[l] = self.Gcovinvs[l].cuda()
    return super().cuda()

  def half(self):
    if hasattr(self, 'omegas'):
      for l in range(1, self.L+1):
        self.omegas[l] = self.omegas[l].half()
        self.Gcovinvs[l] = self.Gcovinvs[l].half()
    return super().half()
  
  def forward(self, x, save_kernel_output=False):
    L = self.L
    # import pdb; pdb.set_trace()
    for l in range(1, L+1):
      nonlin = self.nonlin
      if self.layernorm:
        nonlin = lambda x: self.nonlin(divbystd(x))
      if l == 1:
        x = nonlin(self.first_layer_alpha * self.linears[l](x))
      else:
        x = nonlin(self.linears[l](x))
    # if save_kernel_output:
      # self.kernel_output = x.clone()
    # x = self.linears[L+1](x) * self.last_layer_alpha
    # return x
    
    if save_kernel_output:
      kernel_output = x.clone()
      x = self.linears[L+1](x) * self.last_layer_alpha
      return x, kernel_output
    else:
      x = self.linears[L+1](x) * self.last_layer_alpha
      return x

  def load(self, filename, load_last=True): 
    '''LEGACY LOADING METHOD - DO NOT USE, TEMPORARY USAGE ONLY'''
    import pickle
    with open(filename, 'rb') as handle:
      params = pickle.load(handle)

      last_l = self.L+2 if load_last else self.L+1
      
      with torch.no_grad():
        for l in range(1, last_l):
          self.linears[l].weight[:] = params[(l, "W")]
          self.linears[l].bias[:] = params[(l, "b")] 
          if l < self.L+1:
            self.omegas[l] = params[(l, "w")]

        for l in range(1, self.L+1):
          self.Gcovinvs[l] = torch.inverse(self.omegas[l].T.float() @ self.omegas[l].float()).type_as(self.omegas[l])

class InfPiMLP():
  def __init__(self, d, dout, L, r, initsize=None, initbuffersize=None, maxsize=10000, quiet=False, device='cpu', arrbackend=DynArr, bias_alpha=0, last_bias_alpha=None, first_layer_alpha=1, last_layer_alpha=1,
  layernorm=False, readout_zero_init=False, _last_layer_grad_no_alpha=False, resizemult=2):
    '''
    Inputs:
      d: dim of input
      dout: dim of output
      L: number of hidden layers
      r: rank of probability space
      initbuffersize: initial M
    '''
    self.d = d
    self.dout = dout
    self.r = r
    self.L = L
    self.first_layer_alpha = first_layer_alpha
    self.last_layer_alpha = last_layer_alpha
    self._last_layer_grad_no_alpha = _last_layer_grad_no_alpha
    self.device = device
    self.layernorm = layernorm
    if initsize is None:
      initsize = r
    self.initsize = initsize
    bias = bias_alpha != 0
    self.bias_alpha = bias_alpha
    if last_bias_alpha is None:
      last_bias_alpha = bias_alpha
    self.last_bias_alpha = last_bias_alpha
    self.As = {}  # infnet is parameterized by A: (L x d x r), B: (L x d x r)
    self.Bs = {}
    self.Amult = {}
    self.dAs = {}
    self.dBs = {}
    self.As[1] = torch.randn(d, r, device=device).float() / d**0.5
    self.dAs[1] = torch.zeros(d, r, device=device)
    # lastlayerbuffersize = initbuffersize
    # lastlayerinitsize = initsize
    lastlayerbuffersize = 1 if readout_zero_init else initbuffersize
    lastlayerinitsize = 1 if readout_zero_init else initsize
    for l in range(2, L+2):
      self.Bs[l] = arrbackend(r, initsize=lastlayerinitsize if l == L+1 else initsize,
          initbuffersize=lastlayerbuffersize if l == L+1 else initbuffersize,
          maxsize=maxsize, device=device)
      self.dBs[l] = []
    for l in range(2, L+1):
      self.As[l] = arrbackend(r, initsize=initsize, initbuffersize=initbuffersize, maxsize=maxsize, resizemult=resizemult, device=device)
      self.Amult[l] = arrbackend(None, initsize=initsize, initbuffersize=initbuffersize, maxsize=maxsize, resizemult=resizemult, device=device).float()
      self.dAs[l] = []
    self.As[L+1] = arrbackend(dout, initsize=lastlayerinitsize, initbuffersize=lastlayerbuffersize, maxsize=maxsize, resizemult=resizemult, device=device)
    self.Amult[L+1] = arrbackend(None, initsize=lastlayerinitsize, initbuffersize=lastlayerbuffersize, maxsize=maxsize, resizemult=resizemult, device=device).float()
    self.dAs[L+1] = []
    self.biases = None
    self.dbiases = None
    if bias:  # biases are parameterized like (L x r) and (1 x dout) for the last layer
      self.biases = {}
      self.dbiases = {}
      for l in range(1, L+1):
        self.biases[l] = torch.zeros(r, device=device).float()
        self.dbiases[l] = torch.zeros(r, device=device)
      self.biases[L+1] = torch.zeros(dout, device=device).float()
      self.dbiases[L+1] = torch.zeros(dout, device=device)
    self.initialize(quiet, readout_zero_init=readout_zero_init)
    # internal statistics keeping track of how much pruning done
    self._prune_count = {}
    self.reset_prune_count()

  def initialize_from_data(self, X, sigma=2**0.5, dotest=True):
    # X has shape (B, d)
    # we need r == M == B
    # import pdb; pdb.set_trace()
    assert self.r == self.initsize == X.shape[0]
    d = X.shape[1]
    Sigmas = [X @ X.T / d]
    Ds = [torch.diag(Sigmas[0])**0.5]
    Us = [torch.cholesky(Sigmas[-1], upper=True)]
    for _ in range(self.L+1):
      Sigmas.append(VReLUmatrix(sigma**2 * Sigmas[-1]))
      Ds.append(torch.diag(Sigmas[-1])**0.5)
      Us.append(torch.cholesky(Sigmas[-1], upper=True))
      # print((Us[-1].T @ Us[-1] - Sigmas[-1]).norm())

    # might need regularization in the inverse
    self.As[1][:] = sigma * X.T @ torch.inverse(Sigmas[0]) @ Us[0].T / d
    
    for l in range(2, self.L+1):
      self.As[l].a[:] = sigma**2 * Ds[l-2][:, None] * torch.inverse(Sigmas[l-1]) @ Us[l-1].T
      self.Bs[l].a[:] = Ds[l-2][:, None]**-1 * Us[l-2].T
      self.Amult[l].a[:] = 1
      assert (torch.std(self.Bs[l].a.norm(dim=1) - 1) < 1e-4)

    if dotest:
      self.forward(X.cuda())
      for l in range(1, self.L+1):
        std = (Sigmas[l].cuda() - VReLUmatrix(self.gs[l] @ self.gs[l].T)).std()
        assert (std < 2e-4), f'error std is {std}'

  def initialize(self, quiet=True, readout_zero_init=False):
    if self.d == self.r == self.initsize:
      if not quiet:
        print('init with identity')
      self.As[1][:] = torch.eye(self.r, device=self.device)
      for l in range(2, self.L+2):
        B = self.Bs[l].a
        B[:] = torch.eye(self.r, device=self.device)
        self.Amult[l].a[:] = 1
        A = self.As[l].a
        if l == self.L+1:
          A.zero_()
        else:
          A[:] = torch.eye(self.r, device=self.device)
    else:
      self.As[1].normal_()
      self.As[1] /= self.As[1].norm(dim=0, keepdim=True)
      for l in range(2, self.L+2):
        B = self.Bs[l].a
        B.normal_()
        B[:] = torch.nn.functional.normalize(B, dim=1)
        self.Amult[l].a[:] = 1
        self.As[l].a.normal_()
        if l == self.L+1:
          if not readout_zero_init:
            self.As[l].a.mul_(self.As[l].size**-1)
          else:
            self.As[l].a.mul_(0)
            self.Amult[l].a.mul_(0)
        else:
          self.As[l].a.mul_(self.As[l].size**-0.5)
  
  def parameters(self):
    return [self.As[l].arr if l > 1 else self.As[l] for l in range(1, self.L+1)] + [self.Bs[l].arr for l in range(2, self.L+1)]
    
  def zero_grad(self):
    self.dAs[1].zero_()
    for l in range(2, self.L+2):
      self.dBs[l] = []
      self.dAs[l] = []
    if self.dbiases is not None:
      for _, v in self.dbiases.items():
        v.zero_()

  def zero_readout_grad(self):
    L = self.L
    for d in list(self.dAs[L+1]) + list(self.dBs[L+1]):
      d.zero_()
    if self.dbiases is not None:
      for d in list(self.dbiases[L+1]):
        d.zero_()

  def checkpoint(self):
    with torch.no_grad():
      self.A1_chkpt = self.As[1].clone()
      if self.biases is not None:
        self.biases_chkpt = {k: v.clone() for k, v in self.biases.items()}
        # deepcopy(self.biases)
    for l in range(2, self.L+2):
      self.As[l].checkpoint()
      self.Amult[l].checkpoint()
      self.Bs[l].checkpoint()

  def restore(self):
    self.As[1][:] = self.A1_chkpt
    for l in range(2, self.L+2):
      self.As[l].restore()
      self.Amult[l].restore()
      self.Bs[l].restore()
    if self.biases is not None:
      for l in range(1, self.L+2):
        self.biases[l][:] = self.biases_chkpt[l]

  def cuda(self):
    self.device = 'cuda:0'
    self.As[1] = self.As[1].cuda()
    self.dAs[1] = self.dAs[1].cuda()
    # self.Amult[1] = self.Amult[1].cuda()
    for l in range(2, self.L+2):
      self.As[l] = self.As[l].cuda()
      self.Bs[l] = self.Bs[l].cuda()
      self.Amult[l] = self.Amult[l].cuda()
    if self.biases is not None:
      for l in range(1, self.L+2):
        self.biases[l] = self.biases[l].cuda()
        self.dbiases[l] = self.dbiases[l].cuda()
    return self

  def cpu(self):
    self.device = 'cpu'
    self.As[1] = self.As[1].cpu()
    self.dAs[1] = self.dAs[1].cpu()
    for l in range(2, self.L+2):
      self.As[l] = self.As[l].cpu()
      self.Bs[l] = self.Bs[l].cpu()
      self.Amult[l] = self.Amult[l].cpu()
    if self.biases is not None:
      for l in range(1, self.L+2):
        self.biases[l] = self.biases[l].cpu()
        self.dbiases[l] = self.dbiases[l].cpu()
    return self

  def half(self):
    self.As[1] = self.As[1].half()
    self.dAs[1] = self.dAs[1].half()
    for l in range(2, self.L+2):
      self.As[l] = self.As[l].half()
      self.Bs[l] = self.Bs[l].half()
    # if self.biases is not None:
    #   for l in range(1, self.L+2):
    #     self.biases[l] = self.biases[l].half()
    #     self.dbiases[l] = self.dbiases[l].half()
    return self
  
  def float(self):
    self.As[1] = self.As[1].float()
    self.dAs[1] = self.dAs[1].float()
    for l in range(2, self.L+2):
      self.As[l] = self.As[l].float()
      self.Bs[l] = self.Bs[l].float()
      self.Amult[l] = self.Amult[l].float()
    return self

  def __call__(self, X, doreshape=True):
    return self.forward(X, doreshape=doreshape)

  def forward(self, X, doreshape=True):
    '''
    Input:
      X: (batch, inputdim)
    Output:
      output of network
    '''
    
    # global counter
    # global counterlist
    # global endcounter

    if doreshape:
      self.X = X = X.reshape(X.shape[0], -1)
    else:
      self.X = X
    self.gs = {}
    self.gbars = {}
    self.qs = {}
    self.gs[1] = X @ self.As[1].type_as(X)
    if self.biases is not None:
      self.gs[1] += self.bias_alpha * self.biases[1].type_as(X)
    self.gs[1] *= self.first_layer_alpha
    self.ss = {}
    L = self.L
    # if counter == 2:
      # import pdb; pdb.set_trace()
      # self.As[2].size -= 5
      # self.Amult[2].size -= 5
      # self.Bs[2].size -= 5
      # size = self.As[2].size
      # self.As[2].a[size:] = 0
      # self.Amult[2].a[size:] = 0
      # self.Bs[2].a[size:] = 0
    for l in range(2, L+2):
      # (B, 1)
      self.ss[l-1] = self.gs[l-1].norm(dim=1, keepdim=True)
      # (B, r)
      self.gbars[l-1] = self.gs[l-1] / self.ss[l-1]
      # (B, M)
      self.qs[l] = self.gbars[l-1] @ self.Bs[l].a.T
      # if counter == 2 and l == 3:
      #   # import pdb; p
      #   torch.save(self.qs[l], 'q3.th')
      #   torch.save(self.gbars[l-1], 'gbar2.th')
      #   torch.save(self.Bs[l], 'B3.th')
        # print('\tgbar2', self.gbars[2][0, 0].item(), self.gbars[2].norm(p=1).item())
        # print('\tq3', self.qs[l][0, 0].item(), self.qs[l].norm(p=1).item())
        # print('\tB3', self.Bs[l].a[0, 0].item(), self.Bs[l].a.norm(p=1).item())
      if self.layernorm:
        s = 1
      else:
        s = self.ss[l-1]
      # (B, r) or (B, dout)
      self.gs[l] = (
          F00ReLUsqrt(self.qs[l], 1, s)
          * self.Amult[l].a.type_as(self.qs[l])
          ) @ self.As[l].a
      if self.biases is not None:
        if l == L+1:
          # # if counter == 3:
          # #   import pdb; pdb.set_trace()
          # print('forwarding')
          # print('q3', self.qs[l][0, 0].item())
          # print('A3', self.As[3].a.norm(p=1).item())
          # print('B3', self.Bs[3].a.norm(p=1).item())
          self.gs[l] += self.last_bias_alpha * self.biases[l].type_as(X)
        else:
          self.gs[l] += self.bias_alpha * self.biases[l].type_as(X)
    self.out = self.gs[L+1] * self.last_layer_alpha
    # self.out.requires_grad_()
    # self.out.retain_grad()

    # counter += 1
    # 244 ok, 245 bad
    # if counter in counterlist:
    #   print('counter', counter)
    #   for k, v in self.gs.items():
    #     print(f'g{k}', v.norm().item(), v.norm(p=1).item(), v.mean().item())
    #   for k, v in self.gbars.items():
    #     print(f'gbar{k}', v.norm().item(), v.norm(p=1).item(), v.mean().item())
    #   # print('ss1', self.ss[1].norm().item())
    #   # print('ss2', self.ss[2].norm().item())
    #   # print('q2', self.qs[2].norm().item())
    #   # print('q3', self.qs[3].norm().item())
    #   # print('out', self.out.norm().item())
    # if counter == endcounter:
    #   import sys; sys.exit()
    return self.out

  def save_intermediate(self, out_grad):
    self.X_ = self.X
    self.gs_ = self.gs
    self.gbars_ = self.gbars
    self.qs_ = self.qs
    self.ss_ = self.ss
    self.out_ = self.out
    self.out_grad_ = out_grad

  def del_intermediate(self):
    del self.X_
    del self.gs_
    del self.gbars_
    del self.qs_
    del self.ss_
    del self.out_
    del self.out_grad_

  def readout_backward(self, delta, buffer=None):
    # accumulate gradients
    dAs = self.dAs
    dBs = self.dBs
    dbiases = self.dbiases
    if buffer is not None:
      dAs = buffer[0]
      dBs = buffer[1]
      if self.dbiases is not None:
        dbiases = buffer[2]
    
    L = self.L
    if self.layernorm:
      s = 1
    else:
      s = self.ss[L]
    dAs[L+1].append(delta * s * self.last_layer_alpha)
    dBs[L+1].append(self.gbars[L])
    if self.dbiases is not None:
      dbiases[L+1] += self.last_bias_alpha * self.last_layer_alpha * delta.sum(dim=0)
    # import pdb; pdb.set_trace()

  def backward_somaml(self, delta, buffer=None, readout_fixed_at_zero=False):
    '''
    Input:
      delta: (batch, dout) loss derivative
      buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
    '''
    # first order backward
    self.backward(delta, buffer=buffer)
    # backward through the step 1 final embeddings in the last layer gradients
    self._backward_somaml(delta, buffer=buffer)
    # # backward through loss derivatives of step 1
    if readout_fixed_at_zero:
      # if readout weights and biases are fixed at 0, then
      # no gradient through loss derivatives of step 1
      return
    L = self.L
    ckpt = self.As[L+1]._checkpoint
    
    # Below, B is test batch and B' was train batch for 2nd order maml
    # shape (B, B')
    q = F00ReLUsqrt(self.qs[L+1][:, ckpt:], self.ss_[L].T, self.ss[L])
    # multiply by the multipliers on loss derivatives from train batch
    q *= self.Amult[L+1].a[ckpt:].flatten() * self.last_layer_alpha
    # shape (B', dout)
    c = q.T @ delta
    # self.restore()
    self.As[L+1].restore()
    self.Amult[L+1].restore()
    self.Bs[L+1].restore()
    # shape (B', dout)
    delta2 = torch.autograd.grad(
                self.out_grad_,
                [self.out_],
                c)[0].detach()
    # import pdb; pdb.set_trace()
    # print('dbias', [b.norm().item() for b in buffer[2].values()])
    # print('dA1', buffer[0][1].norm().item())
    # print('dA2', [a.norm().item() for a in buffer[0][2]])
    # # print('dA3', [a.norm().item() for a in buffer[0][3]])
    # print('dB2', [a.norm().item() for a in buffer[1][2]])
    self._backward(delta2, buffer=buffer,
                  gbars=self.gbars_, ss=self.ss_, gs=self.gs_, qs=self.qs_,
                  X=self.X_)
    # import pdb; pdb.set_trace()
    # print('\tdbias', [b.norm().item() for b in buffer[2].values()])
    # print('\tdA1', buffer[0][1].norm().item())
    # print('\tdA2', [a.norm().item() for a in buffer[0][2]])
    # # print('dA3', [a.norm().item() for a in buffer[0][3]])
    # print('\tdB2', [a.norm().item() for a in buffer[1][2]])
  
    # global counter
    # global counterlist, endcounter
    # if counter in counterlist:
    #   print('backward')
    #   print('dA1', self.dAs[1].norm().item())
    #   print('dA2', [a.norm().item() for a in self.dAs[2]])
    #   print('dA3', [a.norm().item() for a in self.dAs[3]])
    #   print('dB2', [a.norm().item() for a in self.dBs[2]])
    #   print('dB3', [a.norm().item() for a in self.dBs[3]])
    #   print('db1', self.dbiases[1].norm().item())
    #   print('db2', self.dbiases[2].norm().item())
    #   print('db3', self.dbiases[3].norm().item())
    # if counter == endcounter:
    #   import sys; sys.exit()

  def _backward_somaml(self, delta, buffer=None):
    '''
    Backprop through the step 1 final embeddings in the last layer gradients
    '''
    self._backward(delta, buffer,
                   gbars=self.gbars_, ss=self.ss_, gs=self.gs_, qs=self.qs_,
                   X=self.X_, somaml=True)

  def backward(self, delta, buffer=None):
    '''
    Input:
      delta: (batch, dout) loss derivative
      buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
    '''
    # import pdb; pdb.set_trace()
    self._backward(delta, buffer)

  def _backward(self, delta, buffer=None,
                gbars=None, ss=None, gs=None, qs=None,
                As=None, Bs=None, X=None,
                somaml=False):
    '''
    Input:
      delta: (batch, dout) loss derivative
      buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
    '''
    L = self.L
    self.dgammas = {}
    self.dalpha02s = {}
    self.dalpha11s = {}
    self.dbeta02s = {}
    self.dbeta11s = {}
    self._dAs = {}
    if gbars is None:
      gbars = self.gbars
    if ss is None:
      ss = self.ss
    if gs is None:
      gs = self.gs
    if qs is None:
      qs = self.qs
    if As is None:
      As = self.As
    if Bs is None:
      Bs = self.Bs
    if X is None:
      X = self.X
    # Below, B is test batch and B' was train batch for 2nd order maml
    # but they are the same for normal SGD
    # (B, M)
    if not somaml:
      self.dgammas[L+1] = delta @ As[L+1].a.T \
        * self.Amult[L+1].a.type_as(delta) * self.last_layer_alpha
    else:
      ckpt = As[L+1]._checkpoint
      # (B', B)
      self.dgammas[L+1] = (
        # (B, dout)
        delta \
        # (dout, B')
        @ self.out_grad_.detach().T \
        # Amult contains lr used in train batch
        * self.Amult[L+1].a[ckpt:].type_as(delta) \
        # 1 copy from train backprop, 1 copy from test backprop
        * self.last_layer_alpha**2
      # using self.ss here for the ss on test batch
            # shape (B)
      ).T * self.ss[L].flatten()

    for l in range(L+1, 1, -1):
      if self.layernorm:
        s = 1
        g = gbars[l-1]
      else:
        s = ss[l-1]
        g = gs[l-1]
      if l == L+1 and somaml:
        # using self.gbars here for gbars on test batch
        # (B, r)
        B = self.gbars[L]
        # (B', B)
        # q = gbars[l-1] @ B.T
        q = self.qs[l][:, ckpt:].T
      else:
        # (M, r)
        B = Bs[l].a
        q = qs[l]
      # (B', M')
      self.dalpha02s[l] = F02ReLUsqrt(q, 1, s)
      # (B', M')
      self.dalpha11s[l] = F11ReLUsqrt(q, 1, s)
      # (B', r)
      self.dbeta11s[l-1] = (self.dgammas[l] * self.dalpha11s[l]) @ B
      # (B', M)
      self.dbeta02s[l-1] = torch.einsum('bm,bm,br->br', self.dalpha02s[l], self.dgammas[l], g)
      # (B', M)
      self._dAs[l-1] = (self.dbeta11s[l-1] + self.dbeta02s[l-1])
      if self.layernorm:
        drho = torch.einsum('br,br->b', self._dAs[l-1], g)
        self._dAs[l-1] -= drho[:, None] * g
        self._dAs[l-1] /= ss[l-1]
      if l > 2:
        self.dgammas[l-1] = self._dAs[l-1] @ As[l-1].a.T \
            * self.Amult[l-1].a.type_as(delta)
    # accumulate gradients
    dAs = self.dAs
    dBs = self.dBs
    dbiases = self.dbiases
    if buffer is not None:
      dAs = buffer[0]
      dBs = buffer[1]
      if self.dbiases is not None:
        dbiases = buffer[2]
    # (B', r)
    dAs[1] += X.T @ self._dAs[1] * self.first_layer_alpha
    for l in range(2, L+2):
      if l == L+1 and somaml:
        continue
      if self.layernorm:
        s = 1
      else:
        s = ss[l-1]
      if l == L+1:
        if self._last_layer_grad_no_alpha:
          mul = 1
        else:
          mul = self.last_layer_alpha
        dAs[l].append(delta * s * mul)
      else:
        dAs[l].append(self._dAs[l] * s)
      dBs[l].append(gbars[l-1])
    if self.dbiases is not None:
      for l in range(1, L+1):
        dbiases[l] += self.bias_alpha * self._dAs[l].sum(dim=0) * (
          self.first_layer_alpha if l == 1 else 1)
      if not somaml:
        dbiases[L+1] += self.last_bias_alpha * self.last_layer_alpha * delta.sum(dim=0)

  def newgradbuffer(self):
    dAs = {1: torch.zeros_like(self.dAs[1])}
    dBs = {}
    for l in range(2, self.L+2):
      dAs[l] = []
      dBs[l] = []
    if self.biases is not None:
      dbiases = {}
      for l in range(1, self.L+2):
        dbiases[l] = torch.zeros_like(self.dbiases[l])
      return dAs, dBs, dbiases
    return dAs, dBs

  def resetbuffer(self, buffer):
    dAs = buffer[0]
    dBs = buffer[1]
    dAs[1].zero_()
    for l in range(2, self.L+2):
      del dAs[l][:]
      del dBs[l][:]
    if self.biases is not None:
      dbiases = buffer[2]
      for l in range(1, self.L+2):
        dbiases[l].zero_()
      return dAs, dBs, dbiases
    return dAs, dBs

  def step(self, lr, wd=0, buffer=None, momentum=0, dampening=0,
          bias_lr_mult=1, first_layer_lr_mult=1,
          last_layer_lr_mult=1,
          apply_lr_mult_to_wd=True,
          prunetol=None):
    
      
    dAs = self.dAs
    dBs = self.dBs
    dbiases = self.dbiases
    if buffer is not None:
      dAs = buffer[0]
      dBs = buffer[1]
      if self.biases is not None:
        dbiases = buffer[2]

    # if buffer is not None:
    #   print('meta grad buffer')
    #   print('\tdbias', [b.norm(p=1).item() for b in dbiases.values()])
    #   print('\tdA1', dAs[1].norm(p=1).item())
    #   print('\tdA2', [a.norm(p=1).item() for a in dAs[2]])
    #   print('\tdA3', [a.norm(p=1).item() for a in dAs[3]])
    #   print('\tdB2', [a.norm(p=1).item() for a in dBs[2]])
    #   print('\tdB3', [a.norm(p=1).item() for a in dBs[3]])
    # else:
    #   print('self grad buffer')
    #   print('\tdbias', [b.norm().item() for b in dbiases.values()])
    #   # print('\tdA1', dAs[1].norm().item())
    #   # print('\tdA2', [a.norm().item() for a in dAs[2]])
    #   print('\tdA3', [a.norm(p=1).item() for a in dAs[3]])
    #   print('\tdB3', [a.norm(p=1).item() for a in dBs[3]])

    # if prunetol:
    #   self.prune_grad(prunetol)

    if momentum > 0:
      # momentum buffer
      if not hasattr(self, 'Vmult'):
        self.Vmult = {}
        for l in range(2, self.L+2):
          self.Vmult[l] = deepcopy(self.Amult[l])
          if wd == 0:
            self.Vmult[l].a[:] = 0
          else:
            self.Vmult[l].a[:] = wd
          # self.Vmult[l].a[:self.initsize] = 0
        self.A1V = torch.zeros_like(self.As[1])
      if self.biases is not None and not hasattr(self, 'biasesV'):
        self.biasesV = {}
        for l in range(1, self.L+2):
          self.biasesV[l] = torch.zeros_like(self.biases[l])
      # TODO: implement pruning for momentum
      # TODO: implement nesterov
      self.A1V.mul_(momentum).add_(dAs[1] + self.As[1] * wd,
                                    alpha=1-dampening)
      self.As[1] -= lr * self.A1V * first_layer_lr_mult
      for l in range(2, self.L+2):
        self.Vmult[l].a.mul_(momentum)
        if wd > 0:
          self.Vmult[l].a[:] += self.Amult[l].a * wd * (1 - dampening)
        mult = last_layer_lr_mult if l == self.L+1 else 1
        # if l == self.L+1:
        #   import pdb; pdb.set_trace()
        if mult == 0:
          continue
        # update existing Amult
        self.Amult[l].a[:] -= lr * mult * self.Vmult[l].a
        # update new grads
        self.As[l].cat(*dAs[l])
        for i, m in enumerate([self.Vmult[l], self.Amult[l]]):
          m.cat(
            # (-1)**i * lr * mult * 
            (1 if i == 0 else -lr * mult)
            * (1 - dampening) * 
            torch.ones(sum(a.shape[0] for a in dAs[l]),
              dtype=dAs[l][0].dtype, device=dAs[l][0].device)
            )
        self.Bs[l].cat(*dBs[l])
      if self.biases is not None:
        for l in range(1, self.L+2):
          self.biasesV[l].mul_(momentum).add_(
            dbiases[l] + self.biases[l] * wd, alpha=1-dampening)
          self.biases[l] -= lr * bias_lr_mult * self.biasesV[l]
        
    else:

      if wd > 0:
        # originally used the following factor for all params
        # TODO: option for turning this back on
        # factor = 1 - lr * wd
        factor = 1 - lr * wd * (first_layer_lr_mult if apply_lr_mult_to_wd else 1)
        self.As[1] *= factor
        for l in range(2, self.L+2):
          if l == self.L+1:
            factor = 1 - lr * wd * (last_layer_lr_mult if apply_lr_mult_to_wd else 1)
          else:
            factor = 1 - lr * wd
          self.Amult[l].a[:] *= factor
          # A = self.As[l].a
          # A[:] *= factor
        # NOTE: earlier results didn't do wd for biases
        # TODO: option for turning off wd for biases
        if self.biases is not None:
          for l in range(1, self.L+2):
            factor = 1 - lr * wd * (bias_lr_mult if apply_lr_mult_to_wd else 1)
            self.biases[l] *= factor
      
      # if prunetol:
      #   self.prune_update(lr, prunetol)
      if prunetol:
        self.prune_greg(lr, prunetol)

      self.As[1] -= lr * dAs[1] * first_layer_lr_mult
      for l in range(2, self.L+2):
        mult = last_layer_lr_mult if l == self.L+1 else 1
        if mult == 0:
          # print(self.Amult[l].a.norm().item())
          # print(len(self.Amult[l].arr), self.Amult[l].size)
          continue
        #   # import pdb; pdb.set_trace()
        # torch.manual_seed(0)
        self.As[l].cat(*dAs[l])
        self.Amult[l].cat(
          -lr * mult * torch.ones(sum(a.shape[0] for a in dAs[l]),
            dtype=torch.float32, device=self.As[l].a.device)
          )
        self.Bs[l].cat(*dBs[l])
      if self.biases is not None:
        for l in range(1, self.L+2):
          self.biases[l] -= lr * bias_lr_mult * dbiases[l]

    # print('\tA3', self.As[3].a.norm(p=1))
    # print('\tB3', self.Bs[3].a.norm(p=1))

  def reset_prune_count(self):
    for l in range(2, self.L+2):
      self._prune_count[l] = 0

  def prune_greg(self, lr, tol=1e-1):
    raise NotImplementedError('need to implement Amult')
    for l in range(2, self.L+2):
      A = self.As[l].a
      B = self.Bs[l].a
      _dAs = []
      _dBs = []
      if isinstance(tol, dict):
        _tol = tol[l]
      elif isinstance(tol, list):
        _tol = tol[l-2]
      elif isinstance(tol, float):
        _tol = tol
      else:
        raise ValueError()
      for dA, dB in zip(self.dAs[l], self.dBs[l]):
        # shape (M, B)
        cov = B @ dB.T
        thres = 1 - _tol / 2
        # collision 0/1, shape (M, B)
        col = (cov > thres).int()
        # collision counts
        # shape (B,)
        dB_counts = col.sum(dim=0)
        # # shape (M,)
        # B_counts = col.sum(dim=1)

        new = dB_counts == 0
        old = dB_counts > 0
        self._prune_count[l] += old.int().sum().item()
        # if (dB_counts >= 2).int().sum() > 0:
        #   print('\t', dB_counts[dB_counts >= 2])
        # TODO: fast scatter version
        # A -= lr * col @ dA
        if True:
          A -= lr * (col[:, old] / dB_counts[old]) @ dA[old]
        # else:
        #   idx_old = torch.nonzero(old)
        #   mydA = dA[idx_old]
        #   col_old = col[:, idx_old]
          

        _dAs.append(dA[new])
        _dBs.append(dB[new])
        
      self.dAs[l].clear()
      self.dBs[l].clear()
      self.dAs[l].extend(_dAs)
      self.dBs[l].extend(_dBs)


  def prune_grad(self, tol=1e-1):
    raise NotImplementedError('need to implement Amult')
    for l in range(2, self.L+2):
      do_again = True
      while(do_again):
        dA = self.dAs[l][0]
        A = self.As[l].a
        dB = self.dBs[l][0]
        B = self.Bs[l].a.cuda()
        m = B.shape[0]
        m_g = dB.shape[0]
        
        grad_dist_matrix = torch.norm(dB.unsqueeze(1) - dB.unsqueeze(0), dim=2) 
        grad_dist_matrix[torch.tril(torch.ones(m_g, m_g)) == 1] = float("inf")

        indices = (grad_dist_matrix < tol).nonzero()
        if indices.shape[0] < 10:
          do_again = False
          break
        zeroed = {}
        pairs = []

        indices = np.array(indices.cpu())
        indices = indices[np.unique(indices[:,[0]],return_index=True,axis=0)[1]]
        indices = indices[np.unique(indices[:,[1]],return_index=True,axis=0)[1]]
        indices_to_keep = torch.ones(indices.shape[0])
        for n, (i,j) in enumerate(indices):
            if i == j or int(i) in zeroed or int(j) in zeroed:
                indices_to_keep[n] = 0
                continue
            else:
                zeroed[int(i)] = 1
                zeroed[int(j)] = 1
                pairs.append((int(i),int(j)))
        indices = np.array(pairs)
        
        rows_to_take = torch.ones(m_g)
        rows_to_take[indices[:, 1]] = 0
        dA[indices[:, 0]] += dA[indices[:, 1]]
        
        dA = dA[rows_to_take.to(dtype=torch.bool)]
        dB = dB[rows_to_take.to(dtype=torch.bool)]
        self.dAs[l][0] = dA
        self.dBs[l][0] = dB

  def prune_update(self, lr, tol=1e-1):
    raise NotImplementedError('need to implement Amult')
    for l in range(2, self.L+2):
      do_again = True
      while(do_again):
        if self.dAs[l][0].shape[0] == 0:
          # completely removed grad previously
          break
        dA = self.dAs[l][0]
        A = self.As[l].a
        dB = self.dBs[l][0]
        B = self.Bs[l].a.cuda()
        m = B.shape[0]
        m_g = dB.shape[0]
        
        indices = None
        # indices = []
        batch_size = 10000
        for n in range(int(B.shape[0] / batch_size) + 1):
          idx1 = int(n*batch_size)
          idx2 = int((n+1)*batch_size)
          if (idx2 > B.shape[0]): idx2 = B.shape[0]
          if idx1 == idx2: break
          B_batch = B[idx1:idx2, :]
          m_batch = idx2-idx1
          dist_matrix = torch.norm(B_batch.unsqueeze(1) - dB.unsqueeze(0), dim=2) 
          batch_indices = (dist_matrix < tol).nonzero()
          batch_indices[:, 0] += int(n*batch_size)
          if indices == None:
            indices = batch_indices
          else:
            indices = torch.cat([indices, batch_indices], dim=0)
        
        if indices.shape[0] == 0:
          do_again = False
          break
        indices = np.array(indices.cpu())
        indices = indices[np.unique(indices[:,[0]],return_index=True,axis=0)[1]]
        indices = indices[np.unique(indices[:,[1]],return_index=True,axis=0)[1]]
        pairs = []
        zeroed = {}
        for (i,j) in indices:
            if i == j or int(i) in zeroed or int(j) in zeroed:
                continue
            else:
                zeroed[int(i)] = 1
                zeroed[int(j)] = 1
                pairs.append((int(i),int(j)))
        indices = np.array(pairs)

        rows_to_take = torch.ones(m_g)
        rows_to_take[indices[:, 1]] = 0
        A[indices[:, 0]] += -lr * dA[indices[:, 1]].to(A.device)

        dA = dA[rows_to_take.to(dtype=torch.bool)]
        dB = dB[rows_to_take.to(dtype=torch.bool)]
        self.dAs[l][0] = dA
        self.dBs[l][0] = dB

  def train(self):
    pass

  def eval(self):
    pass

  def to(self, device):
    if 'cpu' in device.type:
      self.cpu()
    else:
      self.cuda()
    return self
  
  def wnorms(self):
    raise NotImplementedError('need to implement Amult')
    As = self.As
    Bs = self.Bs
    biases = self.biases
    norms = {'weight.1': As[1].norm().item()}
    for l in range(2, self.L+2):
      A = As[l].a
      B = Bs[l].a
      norms[f'weight.{l}'] = ABnorm(A, B)
    if biases is not None:
      norms.update({f'bias.{l}': d.norm().item() for l, d in biases.items()})
    return norms

  def gnorms(self, buffer=None):
    dAs = self.dAs
    dBs = self.dBs
    dbiases = self.dbiases
    if buffer:
      dAs = buffer[0]
      dBs = buffer[1]
      if self.biases is not None:
        dbiases = buffer[2]
    norms = {'weight.1': dAs[1].norm().item()}
    for l in range(2, self.L+2):
      A = torch.cat(dAs[l])
      B = torch.cat(dBs[l])
      norms[f'weight.{l}'] = ABnorm(A, B)
    if dbiases is not None:
      norms.update({f'bias.{l}': d.norm().item() for l, d in dbiases.items()})
    return norms

  def gnorm(self, buffer=None, exclude_last_layer=False):
    gnorms = self.gnorms(buffer)
    if exclude_last_layer:
      del gnorms[f'weight.{self.L+1}']
    # print(gnorms)
    return sum(n**2 for n in gnorms.values())**0.5

  def gclip(self, norm, buffer=None, per_param=False, exclude_last_layer=False):
    if per_param:
      gnorms = self.gnorms(buffer)
      dAs = self.dAs
      dbiases = self.dbiases
      if buffer:
        dAs = buffer[0]
        if self.biases is not None:
          dbiases = buffer[2]
      dAs[1] *= min(1, norm / (1e-10 + gnorms['weight.1']))
      for l in range(2, self.L+2):
        for d in dAs[l]:
          d *= min(1, norm / (1e-10 + gnorms[f'weight.{l}']))
      if self.biases is not None:
        for l in range(1, self.L+2):
          dbiases[l] *= min(1, norm / (1e-10 + gnorms[f'bias.{l}']))
    else:
      gnorm = self.gnorm(buffer, exclude_last_layer=exclude_last_layer)
      ratio = 1
      if gnorm > norm:
        ratio = norm / gnorm
      dAs = self.dAs
      dbiases = self.dbiases
      if buffer:
        dAs = buffer[0]
        if self.biases is not None:
          dbiases = buffer[2]
      dAs[1] *= ratio
      for l in range(2, self.L+2):
        for d in dAs[l]:
          d *= ratio
      if self.biases is not None:
        for l in range(1, self.L+2):
          dbiases[l] *= ratio

  def sample(self, width, fincls=FinPiMLP, tieomegas=False):
    finnet = fincls(self.d, width, self.dout, self.L, bias_alpha=self.bias_alpha, last_bias_alpha=self.last_bias_alpha,
    first_layer_alpha=self.first_layer_alpha, last_layer_alpha=self.last_layer_alpha,
    device=self.device, layernorm=self.layernorm)
    finnet = finnet.to(self.As[2].a.dtype)
    finnet.initialize(self, tieomegas=tieomegas)
    return finnet

  __saved_attrs__ = ['As', 'Amult', 'Bs', 'biases', 'bias_alpha', 'last_bias_alpha', 'first_layer_alpha', 'last_layer_alpha', 'layernorm']

  def state_dict(self):
    d = dict()
    for a in self.__saved_attrs__:
      d[a] = getattr(self, a)
    # d = dict(As=self.As, Amult=self.Amult, Bs=self.Bs, biases=self.biases, bias_alpha=self.bias_alpha, last_bias_alpha=self.last_bias_alpha,
    # first_layer_alpha=self.first_layer_alpha, last_layer_alpha=self.last_layer_alpha, layernorm=self.layernorm)
    return d

  def load_state_dict(self, d):
    for a in self.__saved_attrs__:
      setattr(self, a, d[a])
    # self.As = d['As']
    # self.Amult = d['Amult']
    # self.Bs = d['Bs']
    # self.biases = d['biases']

  def load_state_dict(self, d):
    for a in self.__saved_attrs__:
      setattr(self, a, d[a])
    # self.As = d['As']
    # self.Amult = d['Amult']
    # self.Bs = d['Bs']
    # self.biases = d['biases']


  def load(self, filename):
    '''LEGACY LOADING METHOD - DO NOT USE, TEMPORARY USAGE ONLY'''
    import pickle
    with open(filename, 'rb') as handle:
      import sys
      #sys.path.append("inf/dynamicarray.py")
      import inf.dynamicarray as dynamicarray 
      sys.modules['dynamicarray'] = dynamicarray
      state_dict = pickle.load(handle)

      self.load_state_dict(state_dict)
      

if __name__ == '__main__':
  X = torch.linspace(-np.pi, np.pi).reshape(-1, 1)
  y = torch.sin(X) #.reshape(-1)
  X = torch.cat([X, torch.ones_like(X)], dim=1)
  X = X#.double()
  y = y#.double()
  net = InfPiMLP(d=2, dout=1, L=1, r=2, initbuffersize=1000)

  # print(net.As[1].norm(dim=))
  lr = 0.01
  for i in range(10):
    net.zero_grad()
    yhat = net(X)
    loss = 0.5 * ((yhat - y)**2).sum()
    print(i, loss.item())
    dloss = yhat - y
    print(dloss.shape)
    net.backward(dloss)
    net.step(lr)

  
  # mynet = net.sample(2048)
  # out = mynet(X)
  # print(out) 
  # with open("./test", 'wb') as f:
  #   torch.save(mynet, f)
    
  # newnet = torch.load("./test")

  # print("diff", newnet(X) - out)

