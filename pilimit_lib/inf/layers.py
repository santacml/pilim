from numpy.core.fromnumeric import resize
import torch
from torch import nn
import torch.functional as F
from inf.functional import *
from inf.tensors import *
import collections

# make sure to clarify different than a regular linear layer
# outputs are from preactiation to preactivation and includes activation inside layer

def divbystd(x):
  return x / (1e-5 + torch.norm(x, dim=1, keepdim=True))

class FinPiInputLinearReLU(nn.Module):
    def __init__(self, r, n_in, n_out=None, bias_alpha=1, layernorm=False, inf_layer=None, device="cpu"):
        super(FinPiInputLinearReLU, self).__init__()
        self.r = r
        self.n_in = n_in
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        # self.bias_alpha = bias_alpha
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm
        self.device = device
        self.dtype = torch.get_default_dtype()
        # if inf_layer: self.dtype = inf_layer.A.dtype

        self.weight = FinPiParameter(torch.zeros((n_out, n_in), device=self.device, dtype=self.dtype))
        if bias_alpha:
            self.bias = FinPiParameter(torch.zeros(n_out, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        omega = torch.randn(self.n_out, self.r, device=self.device).float()
        gcovinv = torch.inverse(omega.T @ omega).type_as(self.weight)
        omega = omega.type_as(self.weight)
        self.register_buffer("omega", omega)
        self.register_buffer("gcovinv", gcovinv)
        self.register_buffer("pi_proj", self.omega @ (self.gcovinv @ self.omega.T))
        
        self.initialize(inf_layer)

    @torch.no_grad()
    def initialize(self, inf_layer):
        if inf_layer is not None:
            A = inf_layer.A
        else:
            A = torch.randn(self.n_in, self.r, device=self.device)
            A /= A.norm(dim=0, keepdim=True)

        A = A.type_as(self.weight)
        
        self.weight[:] = self.n_out**-0.5 * (self.omega @ A.T)
        self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

        if self.bias is not None:
            inf_bias = torch.randn(self.r, device=self.device) if inf_layer is None else inf_layer.bias
            inf_bias = inf_bias.type_as(self.weight)
            
            self.bias[:] = self.n_out**-0.5 * self.omega @ inf_bias
            self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def forward(self, input):
        bias = self.bias * self.bias_alpha if self.bias is not None else self.bias
        out = torch.nn.functional.linear(input, self.weight, bias)

        # if self.layernorm:
        #     out = divbystd(out)
        return out

    def half(self):
        super(FinPiInputLinearReLU, self).half()
        self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)
        if self.bias is not None:
            self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def extra_repr(self) -> str:
        return 'n_in={}, n_out={}, bias={}'.format(
            self.n_in, self.n_out, self.bias is not None
        )

class InfPiInputLinearReLU(nn.Module):
    def __init__(self, r, r_out=None, bias_alpha=1, layernorm=False, device="cpu"):
        super(InfPiInputLinearReLU, self).__init__()
        self.r = r
        if r_out is None:
            r_out = r
        self.r_out = r_out
        # self.bias_alpha = bias_alpha
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm # does nothing for inf input as layernorm affects downstream layer
        self.device = device

        self.A = nn.Parameter(torch.randn(r, r_out, device=device, dtype=torch.float32))
        
        if bias_alpha:
            self.bias = nn.Parameter(torch.zeros(r_out, device=device, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.initialize()

    @torch.no_grad()
    def initialize(self):
        self.A.normal_()
        self.A /= self.A.norm(dim=0, keepdim=True)

    def forward(self, g_in):
        g_out = g_in @ self.A.type_as(g_in)

        if self.bias is not None:
            g_out += (self.bias.unsqueeze(0) * self.bias_alpha).type_as(g_out)

        return g_out

    def sample(self, n_in, n_out):
        return FinPiInputLinearReLU(self.r_out, n_in, n_out=n_out, bias_alpha=self.bias_alpha, layernorm=self.layernorm, inf_layer=self, device=self.device)

    def extra_repr(self):
        return 'Rank={}, Output Rank={}, Bias={}'.format(
            self.r, self.r_out, self.bias is not None
        )

class FinPiLinearReLU(nn.Module):
    def __init__(self, r, n_in, n_out=None, bias_alpha=1, output_layer=False, layernorm=False, inf_layer=None, prev_omega=None, nonlin=nn.ReLU, device="cpu"):
        super(FinPiLinearReLU, self).__init__()
        self.r = r
        self.n_in = n_in
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        # self.bias_alpha = bias_alpha
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.output_layer = output_layer
        self.layernorm = layernorm
        self.device = device
        self.nonlin = nonlin()
        self.dtype = torch.get_default_dtype()
        # if inf_layer: self.dtype = inf_layer.A.dtype

        param_type = nn.Parameter if self.output_layer else FinPiParameter
        
        self.weight = param_type(torch.zeros((n_out, n_in), device=self.device, dtype=self.dtype))
        if bias_alpha:
            self.bias = param_type(torch.zeros(n_out, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        if not self.output_layer:
            omega = torch.randn(self.n_out, self.r, device=self.device).float()
            gcovinv = torch.inverse(omega.T @ omega).type_as(self.weight)
            omega = omega.type_as(self.weight)

            self.register_buffer("omega", omega)
            self.register_buffer("gcovinv", gcovinv)
            self.register_buffer("pi_proj", self.omega @ (self.gcovinv @ self.omega.T))
        
        self.initialize(inf_layer, prev_omega)

    @torch.no_grad()
    def initialize(self, inf_layer, prev_omega=None):
        if inf_layer is not None:
            A = inf_layer.A
            Amult = inf_layer.Amult
            B = inf_layer.B
        else:
            A = torch.randn(self.r, self.n_out if self.output_layer else self.r, device=self.device)
            Amult = torch.ones(self.r, device=self.device, dtype=torch.float32)
            B = torch.randn(self.r, self.r, device=self.device)

            A.mul_(0 if self.output_layer else A.shape[1]**-0.5)
            B[:] = torch.nn.functional.normalize(B, dim=1)

        dtype = self.weight.dtype

        A = (Amult * A.T).T.type_as(self.weight)
        B = B.type_as(self.weight)

        if prev_omega is None:
            prev_omega = torch.randn(self.n_in, self.r, device=self.device)
        prev_omega = prev_omega.type_as(self.weight)

        if self.output_layer:
            self.weight[:] = self.n_out**-0.5 * (A.T) @ (self.nonlin(prev_omega @ B.T)).T
        else:
            self.weight[:] = self.n_out**-1.0 * self.omega @ (A.T) @ (self.nonlin(prev_omega @ B.T)).T
            self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

            if self.bias is not None:
                inf_bias = torch.randn(self.r, device=self.device) if inf_layer is None else inf_layer.bias
                inf_bias = inf_bias.type_as(self.weight)

                self.bias[:] = self.n_out**-0.5 * self.omega @ inf_bias
                self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def forward(self, input):
        bias = self.bias * self.bias_alpha if self.bias is not None else self.bias
        out = torch.nn.functional.linear(input, self.weight, bias)
        
        # TODO: michael
        '''
        layernorm is wonky
        it needs to happen after layer alpha, and before relu
        this lib was built to extract layer alpha outside of layer
        however, for infnet, layernorm needs to happen inside the layer after... for grad purposes
        each layer needs layernorm to happen at a different time

        layernorm could be made it's own layer or something, idk. this is very confusing. need a better way of handling this.
        but for now, it works to do the layernorm inside the net itself but set the flag in the layer. not pretty.
        '''
        # if self.layernorm and not self.output_layer:
        #     out = divbystd(out)
        return out

    def half(self):
        super(FinPiLinearReLU, self).half()
        if not self.output_layer:
            self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)
            if self.bias is not None:
                self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def extra_repr(self) -> str:
        return 'n_in={}, n_out={}, bias={}'.format(
            self.n_in, self.n_out, self.bias is not None
        )

class InfPiLinearReLU(nn.Module):
    def __init__(self, r, r_out=None, bias_alpha=1, output_layer=False, layer_alpha=1, layernorm=False, optim_mode="project", device="cpu", cuda_batch_size=None):
        super(InfPiLinearReLU, self).__init__()
        self.r = r
        if optim_mode not in ["project"]:
            raise ValueError("optim_mode must be 'project'")
        if r_out is None:
            r_out = r
        self.r_out = r_out
        # self.bias_alpha = torch.tensor([bias_alpha])
        # self.bias_alpha = bias_alpha
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.output_layer = output_layer
        # self.layer_alpha = layer_alpha
        self.register_buffer("layer_alpha", torch.tensor(layer_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm
        self.device = device
        self.InfPiLinearReLUFunction = InfPiLinearReLUFunctionBuilder(layernorm=layernorm, cuda_batch_size=cuda_batch_size)

        self.dynA = DynamicTensor(r_out, initsize=r, device=device, resizemult=1)
        self.register_parameter(name='A', param=InfPiParameter(self.dynA, apply_lr=False, requires_grad=False, optim_mode=optim_mode))
        self.dynAmult = DynamicTensor(None, initsize=r, device=device, dtype=torch.float32, resizemult=1)
        self.register_parameter(name='Amult', param=InfPiParameter(self.dynAmult, apply_lr=True, lr_mult=layer_alpha, requires_grad=False, optim_mode=optim_mode))
        self.dynB = DynamicTensor(r, initsize=r, device=device, resizemult=1)
        self.register_parameter(name='B', param=InfPiParameter(self.dynB, apply_lr=False, requires_grad=False, optim_mode=optim_mode))
        
        self.project()

        if bias_alpha:
            self.bias = nn.Parameter(torch.zeros(r_out, device=self.device, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.initialize()

        self.layer_buckets = None # for pruning

    @torch.no_grad()
    def initialize(self):
        self.A.normal_()
        self.B.normal_()
        self.B[:] = torch.nn.functional.normalize(self.B, dim=1)

        if self.output_layer:
            self.A.mul_(0)
        else:
            self.A.mul_(self.A.shape[1]**-0.5)

        self.Amult[:] = 1

    def project(self):
        self.optim_mode = "project"
        self.A.project()
        self.B.project()

    def forward(self, g_in, gbar_in=None, s_in=None):
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("A and B have different sizes for M. Check that the gradient is applied to both.")
        
        self.A.set_pi_size(g_in.shape[0])
        self.Amult.set_pi_size(g_in.shape[0])
        self.B.set_pi_size(g_in.shape[0])


        # bias = self.bias * self.bias_alpha if self.bias is not None else self.bias
        bias = (self.bias * self.bias_alpha) if self.bias_alpha else self.bias
        g_out = self.InfPiLinearReLUFunction.apply(g_in, self.A, self.Amult, self.B, self.A.pi, self.Amult.pi, self.B.pi, gbar_in, s_in, bias)
        
        
        return g_out

    def sample(self, n_in, n_out, prev_omega=None):
        return FinPiLinearReLU(self.r, n_in, n_out=n_out, bias_alpha=self.bias_alpha, output_layer=self.output_layer, layernorm=self.layernorm, inf_layer=self, prev_omega=prev_omega, device=self.device)

    def extra_repr(self):
        return 'Rank={}, Output Rank={}, Bias={}'.format(
            self.r, self.r_out, self.bias is not None
        )
