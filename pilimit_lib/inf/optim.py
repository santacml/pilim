from inf.tensors import FinPiParameter, InfPiParameter
import torch
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer

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
    

class PiSGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0):
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


                            grad = p.grad.view(p.grad.shape[0], -1) # for conv layers
                            p.grad[:] = (p.pi_proj @ grad).view(p.grad.shape)
                    
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    # if weight_decay > 0:
                    #     p *= 1 - lr * weight_decay

                    # p.add_(-lr*p.grad.to(torch.get_default_dtype()) )


                elif isinstance(p, InfPiParameter) and p.optim_mode == "project" and p.pi.grad is not None:
                    # no momentum

                    # if p.apply_lr:
                    #     p *= 1 - lr * weight_decay
                    #     p.pi.grad *= -lr

                    # p.add_(p.pi.grad, alpha=1)
                    
                    if p.apply_lr:
                        p *= 1 - lr * weight_decay

                    grad = p.pi.grad

                    p.cat_grad(grad, alpha=-lr)
                

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])


            F.sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=False)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss