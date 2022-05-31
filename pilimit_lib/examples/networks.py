from numpy import diff
import torch
from torch import nn
from inf.layers import *
from inf.optim import *
from torchvision import models

class PiNet(torch.nn.Module):
    '''
    This is a base class only to help with easy loading of pi-networks. 
    Resizing parameters is necessary before loading.
    '''

    def load_state_dict(self, state_dict):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state_dict.keys() and state_dict[name].shape != param.shape:
                    print("resizing", name)
                    param.resize_(state_dict[name].size())
        
        super().load_state_dict(state_dict)

class InfMLP(PiNet):
    def __init__(
            self, 
            d_in, 
            d_out, 
            r, 
            L, 
            first_layer_alpha=1, 
            last_layer_alpha=1, 
            bias_alpha=1, 
            last_bias_alpha=None, 
            layernorm=False, 
            cuda_batch_size=None, 
            device="cpu"):
        super(InfMLP, self).__init__()
        '''
        This class creates a L+2 (for input and output) layer Infinite-width Pi-MLP as specified in our paper: https://openreview.net/forum?id=tUMr0Iox8XW

        We create this class as example usage of the inf-width layers. 

        Inputs:
            d_in: dim of input
            d_out: dim of output
            r: rank of probability space
            L: number of hidden layers
            first_layer_alpha: scalar to multiply to layer outputs
            last_layer_alpha: different scalar to multiply to last layer outptus
            bias_alpha: scalar to multiply to bias  
            last_bias_alpha: different scalar to multiply to bias of last layer
            layernorm: use layernorm in between layers (not used in paper)
            cuda_batch_size: batch data to cuda in chunks (not used in paper)
            device: torch device to use
        '''

        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.L = L

        # save as buffers for saving
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device))
        for n in range(1, L+1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size))
        
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size))

        
    def forward(self, x):
        '''
        Forward propogate through the infnet with a given x.
        
        Note that:
            - we don't need to define a backward, as torch will do that for us
            - layer alphas are accounted for automatically with autograd
        '''
        for n in range(0, self.L+2):
            x = self.layers[n](x)
            if n == 0: 
                x *= self.first_layer_alpha
            if n == self.L+1: 
                x *= self.last_layer_alpha
        return x
        
        
class MetaInfMLP(PiNet):
    def __init__(
            self, 
            d_in, 
            d_out, 
            r, 
            L, 
            first_layer_alpha=1, 
            last_layer_alpha=1, 
            bias_alpha=1, 
            last_bias_alpha=None, 
            layernorm=False, 
            cuda_batch_size=None, 
            device="cpu"):
        super(InfMLP, self).__init__()
        '''
        This class creates a L+2 (for input and output) layer META LEARNING Infinite-width Pi-MLP as specified in our paper: https://openreview.net/forum?id=tUMr0Iox8XW

        We create this class as example usage of the inf-width layers. 

        Inputs:
            d_in: dim of input
            d_out: dim of output
            r: rank of probability space
            L: number of hidden layers
            first_layer_alpha: scalar to multiply to layer outputs
            last_layer_alpha: different scalar to multiply to last layer outptus
            bias_alpha: scalar to multiply to bias  
            last_bias_alpha: different scalar to multiply to bias of last layer
            layernorm: use layernorm in between layers (not used in paper)
            cuda_batch_size: batch data to cuda in chunks (not used in paper)
            device: torch device to use
        '''

        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.L = L

        # save as buffers for saving
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device, return_hidden=True))
        for n in range(1, L+1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=True))
        
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=True))

        
    def forward(self, x):
        '''
        Forward propogate through the infnet with a given x.
        
        Note that:
            - we don't need to define a backward, as torch will do that for us
            - layer alphas are accounted for automatically with autograd
        '''
        gs = [] 
        ss = []
        qs = []
        for n in range(0, self.L+2):
            g, s, q = self.layers[n](x)
            if n == 0: 
                g *= self.first_layer_alpha
            if n == self.L+1: 
                g *= self.last_layer_alpha
            gs.append(g.clone())
            ss.append(s.clone())
            qs.append(q.clone())

        return gs, ss, qs
        

class FinPiMLPSample(torch.nn.Module):
    def __init__(self, infnet, n):
        '''
        Sample an infinite-width network using a finite n.

        Inputs:
            infnet: infnet to use
            n: size of hidden dimension to sample to

        Note that all other parameters (alphas, r, L, etc) are obtained from the infinite-width network.
        '''
        super(FinPiMLPSample, self).__init__()

        self.d_in = infnet.d_in
        self.d_out = infnet.d_out
        self.r = infnet.r
        self.L = infnet.L
        self.first_layer_alpha = infnet.first_layer_alpha
        self.last_layer_alpha = infnet.last_layer_alpha
        self.bias_alpha = infnet.bias_alpha
        self.last_bias_alpha = infnet.last_bias_alpha
        self.layernorm = infnet.layernorm
        self.n = n

        self.layers = nn.ModuleList()

        self.layers.append(infnet.layers[0].sample(self.d_in, self.n))

        for n in range(1, self.L+1):
            self.layers.append(infnet.layers[n].sample(self.n, self.n, self.layers[n-1].omega))

        self.layers.append(infnet.layers[self.L+1].sample(self.n, self.d_out, self.layers[self.L].omega))

    def forward(self, x):
        '''
        Forward propogate through the finite network with a given x.
        '''
        for n in range(0, self.L+1):
            x = self.layers[n](x)
            if n == 0: 
                x = x * self.first_layer_alpha
            if self.layernorm:
                x = divbystd(x)
            x = nn.ReLU()(x)
        x = self.layers[self.L+1](x)
        x *= self.last_layer_alpha

        return x

class MetaInfMLPOps(object):
    def __init__(self, infnet):
        '''
        A class for infnet operations specific to meta-learning like gradient loading/deleting, checkpointing, etc.

        '''
        self.infnet = infnet


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

    def newgradbuffer(self):
        '''
        Create new gradient buffers for MAML.
        '''
        dAs = {1: torch.zeros_like(self.infnet.layers[0].A)}
        dBs = {}
        for l in range(1, self.L+2):
            dAs[l] = []
            dBs[l] = []
        if self.infnet.layers[0].bias is not None:
            dbiases = {}
            for l in range(0, self.L+1):
                dbiases[l] = torch.zeros_like(self.infnet.layers[l].bias)
            return dAs, dBs, dbiases
        return dAs, dBs
        

    def resetbuffer(self, buffer):
        '''
        Reset the gradient buffer for MAML.
        '''
        dAs = buffer[0]
        dBs = buffer[1]
        dAs[0].zero_()
        for l in range(1, self.L+2):
            del dAs[l][:]
            del dBs[l][:]
        if self.infnet.layers[0].bias is not None:
            dbiases = buffer[2]
            for l in range(0, self.L+1):
                dbiases[l].zero_()
            return dAs, dBs, dbiases
        return dAs, dBs