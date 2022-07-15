from numpy.core.fromnumeric import resize
import torch
from torch import nn
import torch.functional as F
from .functional import *
from .tensors import *
import collections

# make sure to clarify different than a regular linear layer
# outputs are from preactiation to preactivation and includes activation inside layer

def divbystd(x):
    ''' 
    Divide an input by it's standard deviation (for layernorm).
    '''
    return x / (1e-5 + torch.norm(x, dim=1, keepdim=True))

class FinPiInputLinearReLU(nn.Module):
    def __init__(
            self, 
            r, 
            n_in, 
            n_out=None, 
            bias_alpha=1, 
            layernorm=False, 
            inf_layer=None, 
            device="cpu"):
        super(FinPiInputLinearReLU, self).__init__()
        '''
        Finite-width input layer to a pi-net, with ReLU activation.

        TODO: activation actually happens outside this clas.

        Inputs:
            r: rank of probability space
            n_in: dim of input
            r_out: dim of output
            bias_alpha: scalar to multiply to bias  
            layernorm: use layernorm in between layers
            inf_layer: inf layer to sample from
            device: torch device to use
        '''
        self.r = r
        self.n_in = n_in
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm
        self.device = device
        self.dtype = torch.get_default_dtype()

        self.weight = FinPiParameter(torch.zeros((n_out, n_in), device=self.device, dtype=self.dtype))
        if bias_alpha:
            self.bias = FinPiParameter(torch.zeros(n_out, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        # store omega, gcovinv, and pi-proj operators for later use
        omega = torch.randn(self.n_out, self.r, device=self.device).float()
        gcovinv = torch.inverse(omega.T @ omega).type_as(self.weight)
        omega = omega.type_as(self.weight)
        self.register_buffer("omega", omega)
        self.register_buffer("gcovinv", gcovinv)
        self.register_buffer("pi_proj", self.omega @ (self.gcovinv @ self.omega.T))
        
        self.initialize(inf_layer)

    @torch.no_grad()
    def initialize(self, inf_layer):
        '''
        Initialize finite-width pi-net.
        If an infinite-width layer is given, will sample from it.
        Otherwise, sample from a randomly initialized infinite-width layer.
        '''
        if inf_layer is not None:
            A = inf_layer.A
        else:
            A = torch.randn(self.n_in, self.r, device=self.device)
            A /= A.norm(dim=0, keepdim=True)

        A = A.type_as(self.weight)
        
        self.weight[:] = self.n_out**-0.5 * (self.omega @ A.T)
        # store_gproj_vars stores these for projection in the optimizer
        self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

        if self.bias is not None:
            inf_bias = torch.randn(self.r, device=self.device) if inf_layer is None else inf_layer.bias
            inf_bias = inf_bias.type_as(self.weight)
            
            self.bias[:] = self.n_out**-0.5 * self.omega @ inf_bias
            self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def forward(self, input):
        '''
        Forward pass given an input.
        The forward pass is just a regular ffn. Only differences are in gradient updating.
        '''

        bias = self.bias * self.bias_alpha if self.bias is not None else self.bias
        out = torch.nn.functional.linear(input, self.weight, bias)

        # if self.layernorm:
        #     out = divbystd(out)
        return out

    def half(self):
        '''
        Convert layer to float16.
        '''
        super(FinPiInputLinearReLU, self).half()
        self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)
        if self.bias is not None:
            self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def extra_repr(self) -> str:
        return 'n_in={}, n_out={}, bias={}'.format(
            self.n_in, self.n_out, self.bias is not None
        )

class InfPiInputLinearReLU(nn.Module):
    def __init__(
            self, 
            r, 
            r_out=None, 
            bias_alpha=1, 
            layernorm=False, 
            device="cpu",
            return_hidden=False):
        '''
        Infinite-width input layer to a pi-net, with ReLU activation.

        Mathematically, not much happens in this class and it is really equivalent to a regular FFN.
        However, this class is important to signify this is for an inf-width pi-net
        for things like initialization, sampling, etc.

        A *very* important concept to understand is that outputs are 
        the next layer pre-activations, and layer activation occurs inside the class itself.
        This is why the activation function is included in the class, 
        so that it can be applied to the previous layer output.

        Inputs:
            r: dim of input
            r_out: rank of output
            bias_alpha: scalar to multiply to bias  
            layernorm: use layernorm in between layers
            device: torch device to use
        '''
        super(InfPiInputLinearReLU, self).__init__()
        self.r = r
        if r_out is None:
            r_out = r
        self.r_out = r_out
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm # does nothing for inf input as layernorm affects downstream layer
        self.device = device
        self.return_hidden = return_hidden

        self.A = nn.Parameter(torch.randn(r, r_out, device=device, dtype=torch.float32))
        
        if bias_alpha:
            self.bias = nn.Parameter(torch.zeros(r_out, device=device, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.initialize()

    @torch.no_grad()
    def initialize(self):
        '''
        Initialize layer.
        '''
        self.A.normal_()
        self.A /= self.A.norm(dim=0, keepdim=True)

    def forward(self, g_in):
        '''
        Forward propagate an input (g_in).
        This is equivalent to a regular FFN and is updated using autograd.
        '''
        g_out = g_in @ self.A.type_as(g_in)

        if self.bias is not None:
            g_out += (self.bias.unsqueeze(0) * self.bias_alpha).type_as(g_out)

        if self.return_hidden:
            return g_out, None, None, None
        else:
            return g_out

    def sample(self, n_in, n_out):
        '''
        Sample a finite-width pi-net input layer using a given n_in and n_out.
        '''
        return FinPiInputLinearReLU(self.r_out, n_in, n_out=n_out, bias_alpha=self.bias_alpha, layernorm=self.layernorm, inf_layer=self, device=self.device)

    def extra_repr(self):
        return 'Rank={}, Output Rank={}, Bias={}'.format(
            self.r, self.r_out, self.bias is not None
        )

class FinPiLinearReLU(nn.Module):
    def __init__(
            self, 
            r, 
            n_in, 
            n_out=None, 
            bias_alpha=1, 
            output_layer=False, 
            layernorm=False, 
            inf_layer=None, 
            prev_omega=None, 
            nonlin=nn.ReLU, 
            device="cpu"):
        super(FinPiLinearReLU, self).__init__()
        '''
        Finite-width layer for a pi-Net, with ReLU activation.

        TODO: activation actually happens outside this clas.

        Inputs:
            r: rank of probability space
            n_in: dim of input
            r_out: dim of output
            bias_alpha: scalar to multiply to bias  
            output_layer: whether this is an output layer or not (determines projection)
            layernorm: use layernorm in between layers
            inf_layer: inf layer to sample from
            prev_omega: the previous layer's omega to use for initialization
            nonlin: the nonlinearity to use
            device: torch device to use
        '''
        self.r = r
        self.n_in = n_in
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.output_layer = output_layer
        self.layernorm = layernorm
        self.device = device
        self.nonlin = nonlin()
        self.dtype = torch.get_default_dtype()

        param_type = nn.Parameter if self.output_layer else FinPiParameter
        
        self.weight = param_type(torch.zeros((n_out, n_in), device=self.device, dtype=self.dtype))
        if bias_alpha:
            self.bias = param_type(torch.zeros(n_out, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        # we do not project output layer
        if not self.output_layer:
            omega = torch.randn(self.n_out, self.r, device=self.device).float()
            gcovinv = torch.inverse(omega.T @ omega).type_as(self.weight)
            omega = omega.type_as(self.weight)

            # store omega, gcovinv, and pi-proj operators for later use
            self.register_buffer("omega", omega)
            self.register_buffer("gcovinv", gcovinv)
            self.register_buffer("pi_proj", self.omega @ (self.gcovinv @ self.omega.T))
        
        self.initialize(inf_layer, prev_omega)

    @torch.no_grad()
    def initialize(self, inf_layer, prev_omega=None):
        '''
        Initialize finite-width pi-net.
        If an infinite-width layer is given, will sample from it.
        Otherwise, sample from a randomly initialized infinite-width layer.

        prev_omega is necessary for proper initialization,
        though in practice we find it isn't very important.
        '''
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

        A = (Amult * A.T).T.type_as(self.weight)
        B = B.type_as(self.weight)

        if prev_omega is None:
            prev_omega = torch.randn(self.n_in, self.r, device=self.device)
        prev_omega = prev_omega.type_as(self.weight)

        if self.output_layer:
            self.weight[:] = self.n_out**-0.5 * (A.T) @ (self.nonlin(prev_omega @ B.T)).T
        else:
            self.weight[:] = self.n_out**-1.0 * self.omega @ (A.T) @ (self.nonlin(prev_omega @ B.T)).T
            # store_gproj_vars stores these for projection in the optimizer
            self.weight.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

            if self.bias is not None:
                inf_bias = torch.randn(self.r, device=self.device) if inf_layer is None else inf_layer.bias
                inf_bias = inf_bias.type_as(self.weight)

                self.bias[:] = self.n_out**-0.5 * self.omega @ inf_bias
                self.bias.store_gproj_vars(self.omega, self.gcovinv, self.pi_proj)

    def forward(self, input):
        '''
        Forward pass given an input.
        The forward pass is just a regular ffn. Only differences are in gradient updating.
        '''
        bias = self.bias * self.bias_alpha if self.bias is not None else self.bias
        out = torch.nn.functional.linear(input, self.weight, bias)
        
        '''
        TODO: michael
        layernorm is wonky
        it needs to happen after layer alpha, and before relu
        this lib was built to extract layer alpha outside of layer
        however, for infnet, layernorm needs to happen inside the layer after... for grad purposes
        each layer needs layernorm to happen at a different time

        layernorm could be made it's own layer or something, idk. need a better way of handling this.
        but for now, it works to do the layernorm inside the net itself but set the flag in the layer. not pretty.
        '''
        # if self.layernorm and not self.output_layer:
        #     out = divbystd(out)
        return out

    def half(self):
        '''
        Convert layer to float16.
        '''
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
    def __init__(
            self, 
            r, 
            r_out=None, 
            bias_alpha=1, 
            output_layer=False, 
            layer_alpha=1, 
            layernorm=False, 
            optim_mode="project", 
            device="cpu", 
            cuda_batch_size=None,
            return_hidden=False):
        super(InfPiLinearReLU, self).__init__()
        '''
        Infinite-width layer for a pi-net, with ReLU activation.

        A *very* important concept to understand is that outputs are 
        the next layer pre-activations, and layer activation occurs inside the class itself.
        This is why the activation function is included in the class, 
        so that it can be applied to the previous layer output.

        Inputs:
            r: rank of input
            r_out: rank of output
            bias_alpha: scalar to multiply to bias  
            output_layer: whether this is an output layer (for initialization)
            layer_alpha: scalar to multiply to layer outputs
            layernorm: use layernorm in between layers
            optim_mode: which mode to optimize in (only projection available)
            device: torch device to use
            cuda_batch_size: 
                if your gpu doesn't have a lot of memory, this will batch
                operations from cpu to gpu in chunks of fixed size.
                This means the net can run for much longer (until RAM runs out),
                but is very, very, very slow.
            return_hidden: return hidden states for meta-learning
        '''

        self.r = r
        if optim_mode not in ["project"]:
            raise ValueError("optim_mode must be 'project'")
        if r_out is None:
            r_out = r
        self.r_out = r_out
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        self.output_layer = output_layer
        self.register_buffer("layer_alpha", torch.tensor(layer_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm
        self.device = device
        self.InfPiLinearReLUFunction = InfPiLinearReLUFunctionBuilder(layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=return_hidden)
        self.return_hidden = return_hidden

        A = torch.zeros([r, r_out], device=device, requires_grad=True)
        # Amult stores lr and momentum in float32 format to ensure accuracy.
        # not using Amult in float16 will eventually incur large errors.
        Amult = torch.zeros([r], device=device, requires_grad=True, dtype=torch.float32)
        B = torch.zeros([r, r], device=device, requires_grad=True)
        
        self.register_parameter(name='A', param=InfPiParameter(A, apply_lr=False, requires_grad=False, optim_mode=optim_mode))
        self.register_parameter(name='Amult', param=InfPiParameter(Amult, apply_lr=True, lr_mult=layer_alpha, requires_grad=False, optim_mode=optim_mode))
        self.register_parameter(name='B', param=InfPiParameter(B, apply_lr=False, requires_grad=False, optim_mode=optim_mode))
        
        self.project()

        if bias_alpha is not None:
            self.bias = nn.Parameter(torch.zeros(r_out, device=self.device, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.initialize()

    @torch.no_grad()
    def initialize(self):
        '''
        Initialize inf-width layer.
        '''
        self.A.normal_()
        self.B.normal_()
        self.B[:] = torch.nn.functional.normalize(self.B.float(), dim=1).to(self.B.dtype)

        if self.output_layer:
            self.A.mul_(0)
        else:
            self.A.mul_(self.A.shape[1]**-0.5)

        self.Amult[:] = 1

    def project(self):
        '''
        Determine optimizaton mode (not really used - stay tuned for other modes, maybe).
        '''
        self.optim_mode = "project"
        self.A.project()
        self.B.project()

    def forward(self, g_in, gbar_in=None, s_in=None):
        '''
        Forward pass given an input.
        
        This does 2 things:
            - set the pi sizes for A, B, and Amult
            - calls InfPiLinearReLUFunction, the real meat of the layer
        '''
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("A and B have different sizes for M. Check that the gradient is applied to both.")
        
        # set_pi_size allows for the dynamic sizing of the backwards projected gradient
        # as this tensor is dependent on batch size
        self.A.set_pi_size(g_in.shape[0])
        self.Amult.set_pi_size(g_in.shape[0])
        self.B.set_pi_size(g_in.shape[0])

        bias = (self.bias * self.bias_alpha) if self.bias_alpha else self.bias

        if self.return_hidden:
            g_out, gbar_in, q_in, s_in = self.InfPiLinearReLUFunction.apply(g_in, self.A, self.Amult, self.B, self.A.pi, self.Amult.pi, self.B.pi, gbar_in, s_in, bias)
            return g_out, gbar_in, q_in, s_in    
        else:
            g_out = self.InfPiLinearReLUFunction.apply(g_in, self.A, self.Amult, self.B, self.A.pi, self.Amult.pi, self.B.pi, gbar_in, s_in, bias)
            return g_out    

    def sample(self, n_in, n_out, prev_omega=None):
        '''
        Sample a finite-width version of this layer given n_in and n_out.

        prev_omega (omegas from the previous layer) strongly preferred as well.
        '''
        return FinPiLinearReLU(self.r, n_in, n_out=n_out, bias_alpha=self.bias_alpha, output_layer=self.output_layer, layernorm=self.layernorm, inf_layer=self, prev_omega=prev_omega, device=self.device)

    def extra_repr(self):
        return 'Rank={}, Output Rank={}, Bias={}'.format(
            self.r, self.r_out, self.bias is not None
        )
