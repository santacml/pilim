import warnings
from .layers import InfPiInputLinearReLU, InfPiLinearReLU
from .tensors import InfPiParameter
import torch
from .math import ABnorm
from .layers import *


@torch.no_grad()
def stage_grad(module):
    '''
    Stage the gradient of all pi parameters in a module
    so that multiple steps can be run before applying lr.
    '''

    for param in module.parameters():
        if not isinstance(param, InfPiParameter): continue

        param.stage_grad()
        
@torch.no_grad()
def unstage_grad(module):
    '''
    Unstage the staged gradient on all parameters.
    '''
    for param in module.parameters():
        if not isinstance(param, InfPiParameter): continue

        param.unstage_grad()

@torch.no_grad()
def pi_init(module):
    '''
    Apply initialization to all pi-modules.
    Does not have a way to pass omegas to finite layers currently.
    '''
    #TODO: make a way to provide a dictionary of previous omegas for finnets
    if isinstance(module, InfPiInputLinearReLU):
        module.initialize()

    elif isinstance(module, InfPiLinearReLU):
        module.initialize()

    elif isinstance(module, FinPiInputLinearReLU):
        module.initialize(None)

    elif isinstance(module, FinPiLinearReLU):
        module.initialize(None)


@torch.no_grad()
def store_pi_grad_norm_(modules, exclude_last_layer=False, buffer=None):
    '''
    Store the ABnorm of all A matrices in a pi modules
    for gradient clipping.

    TODO: refactor this
    '''
    to_store = []
    for module in modules:
        if isinstance(module, InfPiLinearReLU):
            if exclude_last_layer and module.output_layer: continue
            to_store.append(module)


    if buffer is None:
        for module in to_store:
            module.A.pi_grad_norm = ABnorm(module.A.pi.grad.detach(), module.B.pi.grad.detach())
    else:
        cnt = 1
        for module in to_store:
            dA = torch.cat(buffer[0][cnt])
            dB = torch.cat(buffer[1][cnt])
            module.A.pi_grad_norm = ABnorm(dA, dB)
        cnt += 1


def clip_grad_norm_(
        parameters, max_norm, norm_type = 2.0,
        error_if_nonfinite = False,
        buffer=None):
    '''
    A copy of torch's clip_grad_norm_, 
    except that it adds pi_parameters which clip their gradient using
    pi_grad_norm stored in store_pi_grad_norm_.

    TODO: refactor this
    '''

    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    pi_parameters = []
    reg_parameters = []
    for p in parameters:
        if isinstance(p, InfPiParameter) and p.pi_grad_norm is not None:
            pi_parameters.append(p)
        elif p.grad is not None:
            reg_parameters.append(p)

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(reg_parameters) == 0 and len(pi_parameters) == 0:
        return torch.tensor(0.)
    # device = parameters[0].grad.device
    if norm_type == "inf":
        raise NotImplementedError()
        # norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        # total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        norms = []
        for p in reg_parameters:
            norms.append(p.grad.detach().norm())
        for p in pi_parameters:
            norms.append(p.pi_grad_norm)
            p.pi_grad_norm = None # remove afterwards
        
        total_norm = torch.norm(torch.stack(norms), norm_type)
        
    if total_norm.isnan() or total_norm.isinf():
        if error_if_nonfinite:
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        else:
            warnings.warn("Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. "
                          "Note that the default behavior will change in a future release to error out "
                          "if a non-finite total norm is encountered. At that point, setting "
                          "error_if_nonfinite=false will be required to retain the old behavior.",
                          FutureWarning, stacklevel=2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in reg_parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
        
        if buffer is None:
            for p in pi_parameters:
                p.pi.grad.detach().mul_(clip_coef.to(p.pi.grad.device))
        else:
            cnt = 1
            dAs = buffer[0]
            for p in pi_parameters:
                dAs[cnt] = [d * clip_coef.to(d.device) for d in dAs[cnt]]
                # p.pi.grad.detach().mul_(clip_coef.to(p.pi.grad.device))
                cnt += 1
    return total_norm
