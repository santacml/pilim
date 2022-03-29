import math
import torch

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

class InfSGD():
  def __init__(self, model, lr, wd=0, momentum=0,
      bias_lr_mult=1, first_layer_lr_mult=1,
      last_layer_lr_mult=1,
      apply_lr_mult_to_wd=True,
      gclip=0,
      gclip_per_param=False,
      prunetol=None):
    self.lr = lr
    self.bias_lr_mult = bias_lr_mult
    self.first_layer_lr_mult = first_layer_lr_mult
    self.last_layer_lr_mult = last_layer_lr_mult
    self.wd = wd
    self.apply_lr_mult_to_wd = apply_lr_mult_to_wd
    self.momentum = momentum
    self.gclip = gclip
    self.gclip_per_param = gclip_per_param
    self.prunetol = prunetol
    self.model = model
  def step(self, buffer=None):
    if self.gclip > 0:
      self.model.gclip(self.gclip, buffer=buffer, per_param=self.gclip_per_param, exclude_last_layer=self.last_layer_lr_mult==0)
    self.model.step(self.lr, wd=self.wd, buffer=buffer,
      momentum=self.momentum, bias_lr_mult=self.bias_lr_mult,
      first_layer_lr_mult=self.first_layer_lr_mult,
      last_layer_lr_mult=self.last_layer_lr_mult,
      apply_lr_mult_to_wd=self.apply_lr_mult_to_wd,
      prunetol=self.prunetol)
  def zero_grad(self):
    self.model.zero_grad()

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
    