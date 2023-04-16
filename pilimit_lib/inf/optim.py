import torch
import math
import torch.optim._functional as optim_F
from torch.optim.optimizer import Optimizer
from .tensors import FinPiParameter, InfPiParameter
from .utils import *

class PiSGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0):
        '''
        A custom optimizer to perform SGD with pi-nets.

        This optimizer is almost entirely copied from the vanilla torch optimizer.
        The only differenes are in the step function.
        '''
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        super(PiSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PiSGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    if isinstance(p, FinPiParameter):
                        #TODO: accomplish these modifications instead with a decorator or callbacks or something, to not obfuscate sgd code
                        if p.omega is None or p.gcovinv is None:
                            raise ValueError(f"{type(p)} found without an omega or without a gcovinv", p)
                        else:
                            # p.grad[:] = p.omega @ (p.gcovinv @ (p.omega.T @ p.grad))

                            # project the finite pi-net gradient into r-space
                            grad = p.grad.view(p.grad.shape[0], -1) # for conv layers
                            p.grad[:] = (p.pi_proj @ grad).view(p.grad.shape)
                    
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])


                elif isinstance(p, InfPiParameter) and p.optim_mode == "project" and p.pi.grad is not None:
                    # apply weight decay if needed (for Amult)
                    if p.apply_lr:
                        p *= 1 - lr * weight_decay

                    # concatenate the gradient with lr applied and do not add to params_with_grad
                    grad = p.pi.grad
                    p.cat_grad(grad, alpha=-lr)
                



            try:
              optim_F.sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    dampening=dampening,
                    nesterov=False)
            except:  # torch version compatibility - this is a bit gross
              optim_F.sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    dampening=dampening,
                    nesterov=False,
                    maximize=False)
               

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

class MetaInfSGD():
  def __init__(self, model, lr, wd=0, momentum=0,
      bias_lr_mult=1, first_layer_lr_mult=1,
      last_layer_lr_mult=1,
      apply_lr_mult_to_wd=True,
      gclip=0,
      gclip_per_param=False):
    self.lr = lr
    self.bias_lr_mult = bias_lr_mult
    self.first_layer_lr_mult = first_layer_lr_mult
    self.last_layer_lr_mult = last_layer_lr_mult
    self.wd = wd
    self.apply_lr_mult_to_wd = apply_lr_mult_to_wd
    self.momentum = momentum
    self.gclip = gclip
    self.gclip_per_param = gclip_per_param
    self.model = model
  def step(self, metaops, buffer=None):
    if self.gclip > 0:
    #   self.model.gclip(self.gclip, buffer=buffer, per_param=self.gclip_per_param, exclude_last_layer=self.last_layer_lr_mult==0)
    
        store_pi_grad_norm_(self.model.modules(), exclude_last_layer=self.last_layer_lr_mult==0, buffer=buffer)

        # TODO misantac this doesn't exclude last layer BIAS gradient from calculation, but neither did old repo

        if self.gclip_per_param:
            for param in self.model.parameters():
                clip_grad_norm_(param, self.gclip, buffer=buffer)
                # torch.nn.utils.clip_grad_norm_(param, gclip_sch.gclip)  # normal torch usage
        else:
            clip_grad_norm_(self.model.parameters(), self.gclip, buffer=buffer)
    
    # self.model.step(self.lr, wd=self.wd, buffer=buffer,
    #   momentum=self.momentum, bias_lr_mult=self.bias_lr_mult,
    #   first_layer_lr_mult=self.first_layer_lr_mult,
    #   last_layer_lr_mult=self.last_layer_lr_mult,
    #   apply_lr_mult_to_wd=self.apply_lr_mult_to_wd)

    metaops.step(self.lr,  buffer=buffer, bias_lr_mult=self.bias_lr_mult,
      first_layer_lr_mult=self.first_layer_lr_mult,
      last_layer_lr_mult=self.last_layer_lr_mult)

  def zero_grad(self):
    self.model.zero_grad()

class GClipWrapper():
  def __init__(self, optimizer, model, gclip, gclip_per_param, gclip_exclude_last_layer=False):
    self.optimizer = optimizer
    self.model = model
    self.gclip = gclip
    self.gclip_per_param = gclip_per_param
    self.gclip_exclude_last_layer = gclip_exclude_last_layer

  def step(self):
    if self.gclip_per_param:
      for param in self.model.parameters():
        torch.nn.utils.clip_grad_norm_(param, self.gclip)
    else:
      params = dict(self.model.named_parameters())
      if self.gclip_exclude_last_layer:
        # last_layer_lr_mult only apply to weights
        # so only excluding weights here
        del params[f'_linears.{self.model.L}.weight']
      # import pdb; pdb.set_trace()
      torch.nn.utils.clip_grad_norm_(params.values(), self.gclip)
    self.optimizer.step()

  def zero_grad(self):
    self.optimizer.zero_grad()

class MultiStepGClip():
  def __init__(self, gclip, milestones, gamma):
    '''
    A class to keep track of gradient clipping throughout a number of epochs,
    with milestones to multiply gclip by gamma.
    '''
    self.gclip = gclip
    self.milestones = milestones
    self.gamma = gamma
    self.epoch = 0

  def step(self):
    if self.epoch in self.milestones:
      self.gclip *= self.gamma
    self.epoch += 1
    
class InfLinearLR():
  def __init__(self, optimizer, totalsteps, lr0):
    self.optimizer = optimizer
    self.lr0 = lr0
    self.totalsteps = totalsteps
    self.optimizer.lr = lr0
    self.nstep = 0

  def step(self):
    self.optimizer.lr = self.getlr()
    self.nstep += 1
  
  def getlr(self):
    r = self.nstep/self.totalsteps
    return self.lr0 * (1 - r)

class InfCosineAnnealingLR():
  def __init__(self, optimizer, T_max, lr_max, lr_min=0):
    self.optimizer = optimizer
    self.lr_max = lr_max
    self.lr_min = lr_min
    optimizer.lr = lr_max
    self.T_max = T_max
    self.T = 0

  def step(self):
    self.optimizer.lr = self.getlr()
    self.T += 1

  def getlr(self):
    return self.lr_min + 0.5 * (self.lr_max - self.lr_min
            ) * (1 + math.cos(self.T / self.T_max * math.pi))

class InfExpAnnealingLR():
  def __init__(self, optimizer, lnbase, lr_max):
    self.optimizer = optimizer
    self.lr_max = lr_max
    optimizer.lr = lr_max
    self.lnbase = lnbase
    self.T = 0

  def step(self):
    self.optimizer.lr = self.getlr()
    self.T += 1

  def getlr(self):
    return math.exp(self.T * self.lnbase) * self.lr_max
            
class InfMultiStepLR():
  def __init__(self, optimizer, milestones, gamma):
    self.optimizer = optimizer
    self.milestones = milestones
    self.gamma = gamma
    self.epoch = 0

  def step(self):
    if self.epoch in self.milestones:
      self.optimizer.lr *= self.gamma
    self.epoch += 1
    
class MultiStepGClip():
  def __init__(self, gclip, milestones, gamma):
    self.gclip = gclip
    self.milestones = milestones
    self.gamma = gamma
    self.epoch = 0

  def step(self):
    if self.epoch in self.milestones:
      self.gclip *= self.gamma
    self.epoch += 1
    