import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from inf.dynamicarray import DynArr, CycArr
from inf.utils import safe_sqrt, safe_acos, F00ReLUsqrt, F11ReLUsqrt, F02ReLUsqrt, VReLUmatrix, ABnorm

class MyLinear(nn.Linear):
  def __init__(self, *args, **kw):
    '''
    Custom linear class that uses bias alpha.
    '''
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
  '''
  Divide x by its standard deviation.
  '''
  return x / (1e-5 + torch.norm(x, dim=1, keepdim=True))

class FinPiMLP(nn.Module):
  def __init__(
      self, 
      datadim, 
      width, 
      ncls, 
      L, 
      bias_alpha=0, 
      last_bias_alpha=None, 
      nonlin=nn.ReLU, 
      device='cpu', 
      lincls=MyLinear,
      first_layer_alpha=1, 
      last_layer_alpha=1, 
      layernorm=False):
    '''
    This class creates a L+1 (for output) layer Finite-width Pi-MLP as specified in our paper: https://openreview.net/forum?id=tUMr0Iox8XW

    This class is mostly compatible with pytorch functions (i.e. forward, backward, step),
    however, Gproj is a necessary function to project the gradient into r-space
    like the infinite-width network does.

    It is highly recommended to not use this class directly, 
    but instead to create an InfPiMLP and sample it,
    as this class isn't really anything special on its own
    and will in fact have issues without infnet initialization.

    See the sample() function in InfPiMLp

    Inputs:
      datadim: which dimension data is on.
      width: width of network
      ncls: dim of output
      L: number of hidden layers
      bias_alpha: scalar to multiply to bias  
      last_bias_alpha: different scalar to multiply to bias of last layer
      nonlin: nonlinearity to use (we only ever do ReLU)
      device: torch device to use
      lincls: custom linear layer (for bias alpha)
      first_layer_alpha: scalar to multiply to layer outputs
      last_layer_alpha: different scalar to multiply to last layer outptus
      layernorm: use layernorm in between layers (not used in paper)
    Note:
      This network is designed so that the (pre)activations have coordinates
      of order width^-1/2, so that no input/output multipliers are needed.
      This means for normalization layers, we need to divide by width^1/2.
    '''
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
    self.last_bias_alpha = last_bias_alpha
    self._linears = nn.ModuleList()
    
    for l in range(1, L+2):
      if l == 1:
        self.linears[l] = lincls(datadim, width, bias=bias, bias_alpha=bias_alpha, device=self.device)
      elif l == L+1:
        self.linears[l] = lincls(width, ncls, bias=bias, bias_alpha=last_bias_alpha, device=self.device)
      else:
        self.linears[l] = lincls(width, width, bias=bias, bias_alpha=bias_alpha, device=self.device)
      self._linears.append(self.linears[l])

  def initialize(
      self, 
      infnet, 
      keepomegas=False, 
      tieomegas=False):
    '''
    Initialize from an infnet.

    Inputs:
      infnet: the infnet to initialize from
      keepomegas: whether to maintain current omegas or not
      tieomegas: keep omegas the same across layers (for debugging/testing - don't use this)
    '''
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

    # initialize omegas and projection operators
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
    
    # initialize weight matrices using omegas, infnet A/B
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
    '''
    Project the gradient into r-space as specified in the paper.
    '''
    if self.r >= self.width:
      return
    with torch.no_grad():
      for l in range(1, self.L+1):
        grad = self.linears[l].weight.grad
        om = self.omegas[l]
        self.linears[l].weight.grad[:] = om @ (self.Gcovinvs[l] @ (om.T @ grad))
        if self.linears[l].bias is not None:
          self.linears[l].bias.grad[:] = om @ (self.Gcovinvs[l] @ (om.T @ self.linears[l].bias.grad))

  def cuda(self):
    '''
    Put network on cuda (only works in 1-gpu environments).
    '''
    if hasattr(self, 'omegas'):
      for l in range(1, self.L+1):
        self.omegas[l] = self.omegas[l].cuda()
        self.Gcovinvs[l] = self.Gcovinvs[l].cuda()
    return super().cuda()

  def half(self):
    '''
    Convert network to fp16.
    '''
    if hasattr(self, 'omegas'):
      for l in range(1, self.L+1):
        self.omegas[l] = self.omegas[l].half()
        self.Gcovinvs[l] = self.Gcovinvs[l].half()
    return super().half()
  
  def forward(
      self, 
      x, 
      save_kernel_output=False):
    '''
    Give an input to the network

    Inputs:
      x: input
      save_kernel_output: whether to save the penultimate outputs for feature kernel creation
    '''
    L = self.L
    for l in range(1, L+1):
      nonlin = self.nonlin
      if self.layernorm:
        nonlin = lambda x: self.nonlin(divbystd(x))
      if l == 1:
        x = nonlin(self.first_layer_alpha * self.linears[l](x))
      else:
        x = nonlin(self.linears[l](x))
    
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
  def __init__(
      self, 
      d, 
      dout, 
      L, 
      r, 
      initsize=None, 
      initbuffersize=None, 
      maxsize=10000, 
      quiet=False, 
      device='cpu', 
      arrbackend=DynArr, 
      bias_alpha=0, 
      last_bias_alpha=None, 
      first_layer_alpha=1, 
      last_layer_alpha=1,
      layernorm=False, 
      readout_zero_init=False, 
      _last_layer_grad_no_alpha=False, 
      resizemult=2):
    '''
    This class creates a L+1 (for output) layer Infinite-width Pi-MLP as specified in our paper: https://openreview.net/forum?id=tUMr0Iox8XW

    The Pi-MLP cannot be trained using normal pytorch functions and does not contain pytorch layers.
    Instead, we implement custom forward, backward, gclip, and step functions. 
    Please see cifar10mlp.py for example usage.

    Inputs:
      d: dim of input
      dout: dim of output
      L: number of hidden layers
      r: rank of probability space
      initsize: initial size of matrix to use
      initbuffersize: initial M
      maxsize: maximum size for cyclic array (deprecated)
      quiet: don't print comments
      device: torch device to use
      arrbackend: using dynamic array or cyclic array (deprecated)
      bias_alpha: scalar to multiply to bias  
      last_bias_alpha: different scalar to multiply to bias of last layer
      first_layer_alpha: scalar to multiply to layer outputs
      last_layer_alpha: different scalar to multiply to last layer outptus
      layernorm: use layernorm in between layers (not used in paper)
      readout_zero_init: initialize last layer with 0s
      _last_layer_grad_no_alpha: don't use alpha on last layer's gradient (for testing)
      resizemult: how much to increase the isze of the dynamic arrays when necessary
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

    # As, Bs, dAs, and dBs are all stored in these dictionaries for the network
    # Amult is an important array that stores learning rate and momentum as fp32 instead of fp16
    # without Amult, floating point errors would accumulate 
    self.As = {}  # infnet is parameterized by A: (L x d x r), B: (L x d x r)
    self.Bs = {}
    self.Amult = {}
    self.dAs = {}
    self.dBs = {}
    self.As[1] = torch.randn(d, r, device=device).float() / d**0.5
    self.dAs[1] = torch.zeros(d, r, device=device)
    
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

  def initialize_from_data(self, X, sigma=2**0.5, dotest=True):
    '''
    This was an alternate initialization method we attempted - did not end up getting used.
    '''
    assert self.r == self.initsize == X.shape[0]
    d = X.shape[1]
    Sigmas = [X @ X.T / d]
    Ds = [torch.diag(Sigmas[0])**0.5]
    Us = [torch.cholesky(Sigmas[-1], upper=True)]
    for _ in range(self.L+1):
      Sigmas.append(VReLUmatrix(sigma**2 * Sigmas[-1]))
      Ds.append(torch.diag(Sigmas[-1])**0.5)
      Us.append(torch.cholesky(Sigmas[-1], upper=True))

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
    '''
    Initialize the matrices in the network as specified in the paper.
    '''
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
    '''
    Obtain the raw array parameters from the model.
    '''
    return [self.As[l].arr if l > 1 else self.As[l] for l in range(1, self.L+1)] + [self.Bs[l].arr for l in range(2, self.L+1)]
    
  def zero_grad(self):
    '''
    Delete the stored gradient for the network.
    '''
    self.dAs[1].zero_()
    for l in range(2, self.L+2):
      self.dBs[l] = []
      self.dAs[l] = []
    if self.dbiases is not None:
      for _, v in self.dbiases.items():
        v.zero_()

  def zero_readout_grad(self):
    '''
    Delete only the readout gradient for the network.
    '''
    L = self.L
    for d in list(self.dAs[L+1]) + list(self.dBs[L+1]):
      d.zero_()
    if self.dbiases is not None:
      for d in list(self.dbiases[L+1]):
        d.zero_()

  def checkpoint(self):
    '''
    Checkpoint all of the parameters in the network.
    '''
    with torch.no_grad():
      self.A1_chkpt = self.As[1].clone()
      if self.biases is not None:
        self.biases_chkpt = {k: v.clone() for k, v in self.biases.items()}
    for l in range(2, self.L+2):
      self.As[l].checkpoint()
      self.Amult[l].checkpoint()
      self.Bs[l].checkpoint()

  def restore(self):
    '''
    Restore all of the parameters in the network from the latest checkpoint.
    '''
    self.As[1][:] = self.A1_chkpt
    for l in range(2, self.L+2):
      self.As[l].restore()
      self.Amult[l].restore()
      self.Bs[l].restore()
    if self.biases is not None:
      for l in range(1, self.L+2):
        self.biases[l][:] = self.biases_chkpt[l]

  def cuda(self):
    '''
    Convert the network to cuda (only works in 1-gpu environments).
    '''
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
    '''
    Convert the network to cpu.
    '''
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
    '''
    Convert the network to fp16.
    Only converts As and Bs, Amult and bias need to be fp32.
    '''
    self.As[1] = self.As[1].half()
    self.dAs[1] = self.dAs[1].half()
    for l in range(2, self.L+2):
      self.As[l] = self.As[l].half()
      self.Bs[l] = self.Bs[l].half()

    # bias needs to be fp32
    # if self.biases is not None:
    #   for l in range(1, self.L+2):
    #     self.biases[l] = self.biases[l].half()
    #     self.dbiases[l] = self.dbiases[l].half()
    return self
  
  def float(self):
    '''
    Converts the network to fp32.
    '''
    self.As[1] = self.As[1].float()
    self.dAs[1] = self.dAs[1].float()
    for l in range(2, self.L+2):
      self.As[l] = self.As[l].float()
      self.Bs[l] = self.Bs[l].float()
      self.Amult[l] = self.Amult[l].float()
    return self

  def __call__(self, X, doreshape=True):
    '''
    Give an input to the network.
    '''
    return self.forward(X, doreshape=doreshape)

  def forward(self, X, doreshape=True):
    '''
    Give an input to the network and calculate the forward pass.
    Note this will save various intermediate outputs for backpropogation purposes (gs, ss, qs).

    There will be minimal comments in this function. For a more in-depth explanation, see pilimit_lib.

    Input:
      X: (batch, inputdim)
      doreshape: flatten the last dimension
    Output:
      output of network
    '''
    
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
    for l in range(2, L+2):
      # (B, 1)
      self.ss[l-1] = self.gs[l-1].norm(dim=1, keepdim=True)
      # (B, r)
      self.gbars[l-1] = self.gs[l-1] / self.ss[l-1]
      # (B, M)
      self.qs[l] = self.gbars[l-1] @ self.Bs[l].a.T
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
          self.gs[l] += self.last_bias_alpha * self.biases[l].type_as(X)
        else:
          self.gs[l] += self.bias_alpha * self.biases[l].type_as(X)
    self.out = self.gs[L+1] * self.last_layer_alpha
    
    return self.out

  def save_intermediate(self, out_grad):
    '''
    Save intermediate outputs from the network for MAML purposes.
    '''
    self.X_ = self.X
    self.gs_ = self.gs
    self.gbars_ = self.gbars
    self.qs_ = self.qs
    self.ss_ = self.ss
    self.out_ = self.out
    self.out_grad_ = out_grad

  def del_intermediate(self):
    '''
    Delete saved intermediate outputs for the network.
    '''
    del self.X_
    del self.gs_
    del self.gbars_
    del self.qs_
    del self.ss_
    del self.out_
    del self.out_grad_

  def readout_backward(self, delta, buffer=None):
    '''
    Perform gradient backward for only the readout (last) layer.
    '''
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

  def backward_somaml(self, delta, buffer=None, readout_fixed_at_zero=False):
    '''
    Input:
      delta: (batch, dout) loss derivative
      buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
      readout_fixed_at_zero: no gradient after readout if they are 0
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
    self._backward(delta2, buffer=buffer,
                  gbars=self.gbars_, ss=self.ss_, gs=self.gs_, qs=self.qs_,
                  X=self.X_)

  def _backward_somaml(self, delta, buffer=None):
    '''
    Backprop through the step 1 final embeddings in the last layer gradients
    '''
    self._backward(delta, buffer,
                   gbars=self.gbars_, ss=self.ss_, gs=self.gs_, qs=self.qs_,
                   X=self.X_, somaml=True)

  def backward(self, delta, buffer=None):
    '''
    Call backwards.

    Input:
      delta: (batch, dout) loss derivative
      buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
    '''
    self._backward(delta, buffer)

  def _backward(self, delta, buffer=None,
                gbars=None, ss=None, gs=None, qs=None,
                As=None, Bs=None, X=None,
                somaml=False):
    '''
    Perform backpropogation on the infinite-width network. 
    Note that this requires the saved items from the forward pass (gs, ss, qs).

    There will be minimal comments in this function. For a more in-depth explanation, see pilimit_lib.

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
    '''
    Create new gradient buffers for MAML.
    '''
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
    '''
    Reset the gradient buffer for MAML.
    '''
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

  def step(
    self,
    lr,
    wd=0, 
    buffer=None, 
    momentum=0, 
    dampening=0,
    bias_lr_mult=1, 
    first_layer_lr_mult=1,
    last_layer_lr_mult=1,
    apply_lr_mult_to_wd=True):
    '''
    Perform a gradient descent step on the Pi-Net.

    Note this first requires doing a forwards and backwards pass to store the dAs and dBs, 
    which are gradient updates for A and B matrices.
    
    Amult only stores the learning rate and momentum in fp32 format.

    There will be minimal comments in this function. For a more in-depth explanation, see pilimit_lib.

    Inputs:
      lr: learning rate
      wd: weight decay
      buffer: buffered gradients for MAML
      momentum: momentum
      dampening: dampening (not used in paper)
      bias_lr_mult: extra learning rate multiplier for bias
      first_layer_lr_mult: extra learning rate multiplier for the first layer
      last_layer_lr_mult: extra learning rate multiplier for the last layer
      apply_lr_mult_to_wd: whether to apply the learning rate multipliers to weight decay
    '''
    
      
    dAs = self.dAs
    dBs = self.dBs
    dbiases = self.dbiases
    if buffer is not None:
      dAs = buffer[0]
      dBs = buffer[1]
      if self.biases is not None:
        dbiases = buffer[2]

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
        self.A1V = torch.zeros_like(self.As[1])
      if self.biases is not None and not hasattr(self, 'biasesV'):
        self.biasesV = {}
        for l in range(1, self.L+2):
          self.biasesV[l] = torch.zeros_like(self.biases[l])
      
      # TODO: implement nesterov
      self.A1V.mul_(momentum).add_(dAs[1] + self.As[1] * wd,
                                    alpha=1-dampening)
      self.As[1] -= lr * self.A1V * first_layer_lr_mult
      for l in range(2, self.L+2):
        self.Vmult[l].a.mul_(momentum)
        if wd > 0:
          self.Vmult[l].a[:] += self.Amult[l].a * wd * (1 - dampening)
        mult = last_layer_lr_mult if l == self.L+1 else 1
        if mult == 0:
          continue

        # update existing Amult
        self.Amult[l].a[:] -= lr * mult * self.Vmult[l].a
        # update new grads
        self.As[l].cat(*dAs[l])
        for i, m in enumerate([self.Vmult[l], self.Amult[l]]):
          m.cat(
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
        # NOTE: earlier results didn't do wd for biases
        # TODO: option for turning off wd for biases
        if self.biases is not None:
          for l in range(1, self.L+2):
            factor = 1 - lr * wd * (bias_lr_mult if apply_lr_mult_to_wd else 1)
            self.biases[l] *= factor
      
      self.As[1] -= lr * dAs[1] * first_layer_lr_mult
      for l in range(2, self.L+2):
        mult = last_layer_lr_mult if l == self.L+1 else 1
        if mult == 0:
          continue
          
        self.As[l].cat(*dAs[l])
        self.Amult[l].cat(
          -lr * mult * torch.ones(sum(a.shape[0] for a in dAs[l]),
            dtype=torch.float32, device=self.As[l].a.device)
          )
        self.Bs[l].cat(*dBs[l])
      if self.biases is not None:
        for l in range(1, self.L+2):
          self.biases[l] -= lr * bias_lr_mult * dbiases[l]

  def train(self):
    '''
    Necessary blank function for pytorch.
    '''
    pass

  def eval(self):
    '''
    Necessary blank function for pytorch.
    '''
    pass

  def to(self, device):
    '''
    Cast to device
    '''
    if 'cpu' in device.type:
      self.cpu()
    else:
      self.cuda()
    return self
  
  def wnorms(self):
    '''
    Weight norms (deprecated).
    '''
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
    '''
    Gradient norms (for gradient clipping).
    '''
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
    '''
    Gradient norm accumulation (for gradient clipping).
    '''
    gnorms = self.gnorms(buffer)
    if exclude_last_layer:
      del gnorms[f'weight.{self.L+1}']
    return sum(n**2 for n in gnorms.values())**0.5

  def gclip(self, norm, buffer=None, per_param=False, exclude_last_layer=False):
    '''
    Gradient clipping.

    Input:
      norm: the norm to clIp to
      buffer: used for metalearning.
      per_param: whether to clip each param individually or clip all params at once.
      exclude_last_layer: whether to exclude the last layer from gradient clipping.
    '''
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
    '''
    Sample a finite network from this infinite-width network.

    This can be used any time, but generally we use thisat initialization,
    so we can see the finite networks approach infinite width throughout training

    Input:
      width: the size of the finite-width network
      fincls: the finite-width network class to use
      tieomegas: whether to use different omegas per layer (for debugging/testing, don't use this)
    '''

    finnet = fincls(self.d, width, self.dout, self.L, bias_alpha=self.bias_alpha, last_bias_alpha=self.last_bias_alpha,
    first_layer_alpha=self.first_layer_alpha, last_layer_alpha=self.last_layer_alpha,
    device=self.device, layernorm=self.layernorm)
    finnet = finnet.to(self.As[2].a.dtype)
    finnet.initialize(self, tieomegas=tieomegas)
    return finnet

  # attrs necessary for saving/loading
  __saved_attrs__ = ['As', 'Amult', 'Bs', 'biases', 'bias_alpha', 'last_bias_alpha', 'first_layer_alpha', 'last_layer_alpha', 'layernorm']

  def state_dict(self):
    '''
    Get the state dict of all necessary attrs from this network.
    '''
    d = dict()
    for a in self.__saved_attrs__:
      d[a] = getattr(self, a)
    return d

  def load_state_dict(self, d):
    '''
    Load a state dict of all necessary attrs from this network.
    '''
    for a in self.__saved_attrs__:
      setattr(self, a, d[a])

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
  # if this file is run on its own, it will create a very tiny InfPiMLP.
  # this is highly useful for testing small changes.


  X = torch.linspace(-np.pi, np.pi).reshape(-1, 1)
  y = torch.sin(X) #.reshape(-1)
  X = torch.cat([X, torch.ones_like(X)], dim=1)
  X = X#.double()
  y = y#.double()
  net = InfPiMLP(d=2, dout=1, L=1, r=2, initbuffersize=1000)

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

