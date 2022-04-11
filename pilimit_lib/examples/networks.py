from numpy import diff
import torch
from torch import nn
from inf.layers import *
from inf.optim import *
from torchvision import models

class PiNet(torch.nn.Module):
    def load_state_dict(self, state_dict):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state_dict.keys() and state_dict[name].shape != param.shape:
                    print("resizing", name)
                    param.resize_(state_dict[name].size())
                    # param.resize_(new_params[name].size()).copy_(new_params[name])
                    # param.data = torch.zeros_like(new_params[name])
        
        super().load_state_dict(state_dict)

class InfMLP(PiNet):
    def __init__(self, d_in, d_out, r, L, first_layer_alpha=1, last_layer_alpha=1, bias_alpha=1, last_bias_alpha=None, layernorm=False, cuda_batch_size=None, device="cpu"):
        super(InfMLP, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.L = L
        # self.first_layer_alpha = first_layer_alpha
        # self.last_layer_alpha = last_layer_alpha
        # self.bias_alpha = bias_alpha
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        # self.last_bias_alpha = last_bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device))
        for n in range(1, L+1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size))
        
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size))

        
    def forward(self, x):
        for n in range(0, self.L+2):
            x = self.layers[n](x)
            if n == 0: 
                x *= self.first_layer_alpha
            if n == self.L+1: 
                x *= self.last_layer_alpha
        return x
        

class FinPiMLPSample(torch.nn.Module):
    def __init__(self, infnet, n):
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

        self.layers = []

        self.layers.append(infnet.layers[0].sample(self.d_in, self.n))

        for n in range(1, self.L+1):
            self.layers.append(infnet.layers[n].sample(self.n, self.n, self.layers[n-1].omega))

        self.layers.append(infnet.layers[self.L+1].sample(self.n, self.d_out, self.layers[self.L].omega))

        self._layers = nn.ModuleList()

        for layer in self.layers:
            self._layers.append(layer)
        
    def forward(self, x):
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

