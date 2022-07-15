import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from .dynamicarray import DynArr, CycArr
from .utils import J1, safe_sqrt, safe_acos, F00ReLUsqrt, F11ReLUsqrt, F02ReLUsqrt, VReLUmatrix, ABnorm, F11norm, VStepmatrix, J0

class KernelModel():
  def __init__(self, input_dim, output_dim, kerfn, initbuffersize=1, device='cpu', arrbackend=DynArr):
    '''
    Inputs:
      input_dim: 
      kerfn: Takes two tensors of shapes (B1, input_dim) and (B2, input_dim), and returns a tensor (B1, B2) of the kernel applied to each combination of B1 * B2 row vectors.
    '''
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.kerfn = kerfn
    def arr(dim):
      return arrbackend(dim, initsize=1, initbuffersize=initbuffersize, device=device)
    self.B = arr(input_dim)
    self.A = arr(output_dim)
    self.zero_grad()
    self.initialize()

  def initialize(self):
    self.A.a[:] = 0
    self.B.a[:] = 0

  def zero_grad(self):
    self.dA = []
    self.dB = []

  def checkpoint(self):
    for arr in [self.A, self.B]:
      arr.checkpoint()

  def restore(self):
    for arr in [self.A, self.B]:
      arr.restore()

  def newgradbuffer(self):
    return [], []

  def resetbuffer(self, buffer):
    for d in buffer:
      d[:] = []
    return buffer

  def __call__(self, X, doreshape=True):
    return self.forward(X, doreshape=doreshape)

  def forward(self, X, doreshape=True):
    if doreshape:
      self.X = X = X.reshape(X.shape[0], -1)
    else:
      self.X = X
    
    # (batchsize, output_dim)
    out = self.out = self.kerfn(X, self.B.a) @ self.A.a
    out.requires_grad_()
    out.retain_grad()
    return out

  def backward(self, delta, buffer=None, X=None):
    '''
    Input:
      delta: shape B x dout
    '''
    dA = self.dA
    dB = self.dB
    if buffer is not None:
      dA, dB = buffer
    X = self.X if X is None else X
    dB.append(X)
    dA.append(delta)

  def readout_backward(self, *args, **kwargs):
    return self.backward(*args, **kwargs)

  # def backward(self, delta, buffer=None):
  #   '''
  #   Input:
  #     delta: shape B x dout
  #   '''
  #   dA = self.dA
  #   dB = self.dB
  #   if buffer is not None:
  #     dA, dB = buffer
  #   dB.append(self.X)
  #   dA.append(delta)

  def save_intermediate(self, out_grad):
    self.X_ = self.X
    self.out_ = self.out
    self.out_grad_ = out_grad

  def del_intermediate(self):
    del self.X_
    del self.out_
    del self.out_grad_

  def backward_somaml(self, delta, buffer=None, readout_fixed_at_zero=False):
    # first order backward
    self.backward(delta, buffer=buffer)
    if readout_fixed_at_zero:
      raise NotImplementedError()
    
    ckpt = self.A._checkpoint

    # Below, B is test batch and B' was train batch for 2nd order maml
    # self.B.a[ckpt:] contains the train batch
    # self.X contains the test batch
    # shape (B, B')
    q = self.kerfn(self.X, self.B.a[ckpt:])
    # multiply by the multipliers on loss derivatives from train batch
    q *= -self._lr
    # shape (B', dout)
    c = q.T @ delta

    self.A.restore()
    self.B.restore()

    delta2 = torch.autograd.grad(self.out_grad_, [self.out_], c)[0].detach()
    self.backward(delta2, buffer=buffer, X=self.X_)


  def step(self, lr, wd=0, momentum=None, buffer=None, **kw):
    # TODO: momentum not implemented
    dA, dB = self.dA, self.dB
    if buffer is not None:
      dA, dB = buffer
    if wd > 0:
      factor = 1 - lr * wd
      self.A.a[:] *= factor
    self.B.cat(*dB)
    self.A.cat(*[-lr * d for d in dA])
    # for second order maml
    self._lr = lr

  def gnorm(self, buffer=None):
    dA, dB = self.dA, self.dB
    if buffer is not None:
      dA, dB = buffer
    A = torch.cat(dA)
    B = torch.cat(dB)
    # import pdb; pdb.set_trace()
    cov = self.kerfn(B, B)
    return torch.einsum('id,jd,ij->', A, A, cov)**0.5

  def gclip(self, norm, buffer=None, per_param=False, exclude_last_layer=False):
    if exclude_last_layer:
      return
    gnorm = self.gnorm(buffer)
    ratio = 1
    if gnorm > norm:
      ratio = norm / gnorm
    dA = self.dA
    if buffer is not None:
      dA, dB = buffer
    for d in dA:
      d *= ratio
  
  def state_dict(self):
    d = {'A': self.A, 'B': self.B, 'kerfn': self.kerfn,
        'input_dim': self.input_dim, 'output_dim': self.output_dim}
    return d

  def load_state_dict(self, d):
    self.A = d['A']
    self.B = d['B']
    self.kerfn = d['kerfn']
    self.input_dim = d['input_dim']
    self.output_dim = d['output_dim']

  ### format conversion

  def cuda(self):
    self.A = self.A.cuda()
    self.B = self.B.cuda()
    self.device = 'cuda'
    return self

  def cpu(self):
    self.A = self.A.cpu()
    self.B = self.B.cpu()
    self.device = 'cpu'
    return self

  def half(self):
    self.A = self.A.half()
    self.B = self.B.half()
    return self

  def float(self):
    self.A = self.A.float()
    self.B = self.B.float()
    return self
    
  def to(self, device):
    if 'cpu' in device.type:
      self.cpu()
    else:
      self.cuda()
    return self

  def train(self):
    pass

  def eval(self):
    pass

def relu_gp_fn(varws, varbs):
  '''
  Inputs:
    varws: length is number of (linear, relu) in the network. Layer l has init variance `varws[l]/fanin`.
    varbs: length is number of (linear, relu) in the network
  '''
  def kerfn(X, Y):
    '''
      Takes two tensors of shapes (B1, input_dim) and (B2, input_dim), and returns a tensor (B1, B2) of the kernel applied to each combination of B1 * B2 row vectors.
    '''
    d = X.shape[1]
    # shape (B1,)
    varX = X.norm(dim=1)**2 / d
    # shape (B2,)
    varY = Y.norm(dim=1)**2 / d
    # shape (B1, B2)
    cov = X @ Y.T / d
    for varw, varb in zip(varws, varbs):
      # import pdb; pdb.set_trace()
      # preactivation var, cov, cor
      varX = varX * varw + varb
      varY = varY * varw + varb
      cov = cov * varw + varb
      cor = cov * varX[:, None]**-0.5 * varY[None, :]**-0.5
      # postactivation cor, var, cov
      cor = J1(cor)
      varX /= 2
      varY /= 2
      cov = cor * varX[:, None]**0.5 * varY[None, :]**0.5
    return cov
  return kerfn


def relu_ntk_fn(varws, varbs, lrws, lrbs):
  '''
  Inputs:
    varws: length is number of weight matrices in the network, i.e. #hidden_layers+1. Layer l has init variance `varws[l]/fanin`.
    varbs: length is number of biases in the network, i.e. #hidden_layers+1.
    lrws: lr multipliers for weight matrices. Length is #hidden_layers + 1
    lrbs: lr multipliers for biases. Length is #hidden_layers+1
  '''
  def kerfn(X, Y):
    '''
      Takes two tensors of shapes (B1, input_dim) and (B2, input_dim), and returns a tensor (B1, B2) of the kernel applied to each combination of B1 * B2 row vectors.
    '''
    j0s = []
    j1s = []
    d = X.shape[1]
    # shape (B1,)
    varX = X.norm(dim=1)**2 / d
    # shape (B2,)
    varY = Y.norm(dim=1)**2 / d
    # shape (B1, B2)
    cov = cov_in = X @ Y.T / d
    j1s.append(cov_in)
    for varw, varb in zip(varws[:-1], varbs[:-1]):
      # import pdb; pdb.set_trace()
      # preactivation var, cov, cor
      varX = varX * varw + varb
      varY = varY * varw + varb
      cov = cov * varw + varb
      cor = cov * varX[:, None]**-0.5 * varY[None, :]**-0.5
      # postactivation cor, var, cov
      cor = J1(cor)
      cor0 = J0(cor)
      varX /= 2
      varY /= 2
      cov = cor * varX[:, None]**0.5 * varY[None, :]**0.5
      j0s.append(cor0)
      j1s.append(cov)
    # j0s has length #hidden_layers
    # j1s has length #hidden_layers + 1
    # gcovs contains gradient cov at preactivations, starting with output
    gcovs = [torch.ones_like(j0s[-1])]
    for j0, varw in list(zip(j0s, varws[1:]))[::-1]:
      gcovs.append(gcovs[-1] * j0 * varw)
    # gcovs has length #hidden_layers + 1
    gcovs = gcovs[::-1]
    ntk = torch.zeros_like(cov_in)
    # summing up contributions from each layer's weights and biases
    # weighted by the lr mults
    for lrw, lrb, j1, gcov in zip(lrws, lrbs + [0], j1s, gcovs):
      ntk += (lrw * j1 + lrb) * gcov
    return ntk
  return kerfn
      
class InfReLUGPModel(KernelModel):
  def __init__(self, varws, varbs, input_dim, output_dim, *args, **kwargs):
    kerfn = relu_gp_fn(varws, varbs)
    super().__init__(input_dim, output_dim, kerfn, *args, **kwargs)
    self.varws = varws
    self.varbs = varbs

  def sample(self, width, fincls=None):
    if fincls is None:
      fincls = FinReLUGPModel
    finnet = fincls(self, width)
    return finnet


class FinReLUGPModel(nn.Module):
  def __init__(self, infmodel, width, lincls=nn.Linear):
    super().__init__()
    self.width = width
    self.varws = infmodel.varws
    self.varbs = infmodel.varbs
    self.input_dim = infmodel.input_dim
    self.output_dim = infmodel.output_dim
    self.linears = nn.ModuleList()
    for i, (varw, varb) in enumerate(zip(self.varws, self.varbs)):
      if i == 0:
        d_in = self.input_dim
      else:
        d_in = width
      lin = lincls(d_in, width)
      with torch.no_grad():
        lin.weight.normal_()
        lin.bias.normal_()
        lin.weight *= (varw/d_in)**0.5
        lin.bias *= varb**0.5
        lin.weight.requires_grad_(False)
        lin.bias.requires_grad_(False)
      self.linears.append(lin)
    self.readout = lincls(width, self.output_dim, bias=False)
    with torch.no_grad():
      self.readout.weight *= 0

  def forward(self, X, doreshape=True):
    if doreshape:
      X = X.reshape(X.shape[0], -1)
    for lin in self.linears:
      X = F.relu(lin(X))
    return self.readout(X) / self.width**0.5


class InfReLUNTKModel(KernelModel):
  def __init__(self, varws, varbs, lrws, lrbs, input_dim, output_dim, *args, **kwargs):
    kerfn = relu_ntk_fn(varws, varbs, lrws, lrbs)
    super().__init__(input_dim, output_dim, kerfn, *args, **kwargs)
    self.varws = varws
    self.varbs = varbs

if __name__ == '__main__':
  nmats = 3
  # ntkfn = relu_ntk_fn([1]*nmats, [0]*nmats, [1]*nmats, [1]*nmats)
  ntkfn = relu_ntk_fn([1]*nmats, [0]*nmats, [0]*(nmats-1) + [1], [0]*nmats)
  gpfn = relu_gp_fn([1]*(nmats-1), [0]*(nmats-1))
  X = torch.randn(3, 5)
  Y = torch.randn(4, 5)
  print(ntkfn(X, Y))
  print(gpfn(X, Y))