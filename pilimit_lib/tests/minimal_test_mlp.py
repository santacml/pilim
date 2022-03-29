import torch
from torch import nn
from inf.layers import *
from inf.optim import *
from torchvision import models
from examples.networks import InfMLP, FinPiMLPSample
from inf.utils import *
import time


torch.set_default_dtype(torch.float16)
torch.manual_seed(3133)
np.random.seed(3331)
device="cuda"

data = torch.linspace(-np.pi, np.pi, device=device).reshape(-1, 1)
labels = torch.sin(data) #.reshape(-1)
data = torch.cat([data, torch.ones_like(data, device=device)], dim=1)


# data = torch.cat([data, data], dim=0)
# labels = torch.cat([labels, labels], dim=0)


d_in = 2
d_out = 3
r = 20
L = 1
first_layer_alpha = 2
last_layer_alpha = 2
bias_alpha = .5
batch_size = 50
net = InfMLP(d_in, d_out, r, L, device=device, first_layer_alpha=first_layer_alpha, last_layer_alpha=last_layer_alpha, bias_alpha=bias_alpha )
# import copy
# net = copy.deepcopy(net)

# net.apply(pi_init)

# stores even nested params!
# print(net.layers[1].bias.omega == model_copy.layers[1].bias.omega)

# 0/0


# print([type(n) for n in net.parameters()])
# 0/0





net.train()
epoch = 20
# gclip = 0
accum_steps = 1
gclip = .1
# epoch = 3
optimizer = PiSGD(net.parameters(), lr = .02)
tic = time.time()
for epoch in range(epoch):
    if epoch % accum_steps == 0:
        optimizer.zero_grad()
        net.zero_grad()
    
    prediction = net(data)
    
    loss = torch.sum((prediction - labels)**2)**.5

    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    
    loss.backward()
    stage_grad(net)

    if epoch % accum_steps == 0:
        unstage_grad(net)

        if gclip:
            store_pi_grad_norm_(net.modules())
            clip_grad_norm_(net.parameters(), gclip)

        optimizer.step()

    # print(net.layers[1])

    # print("MEM", torch.cuda.memory_reserved() / 1e9, torch.cuda.max_memory_reserved()  / 1e9)
print("time", time.time() - tic)

init_pred = net(data) 
torch.save(net.state_dict(), 'trained.th')
# print(net.state_dict().keys())
# 0/0

new_params = torch.load('trained.th')
# my_params = model_copy.state_dict()

# model_copy = type(net)(net.d_in, net.d_out, net.r, net.L, device=device, first_layer_alpha=first_layer_alpha, last_layer_alpha=last_layer_alpha, bias_alpha=bias_alpha )
model_copy = type(net)(net.d_in, net.d_out, net.r, net.L, device=device )
# model_copy.load_state_dict(net.state_dict()) # works if register buffer is used where appropriate
model_copy.load_state_dict(new_params)
    
print("post loading", (model_copy(data)  - net(data)).abs().sum())


# finnet = FinPiMLPSample(net, 400)
# torch.save(finnet.state_dict(), 'epoch0.th')

# new_finnet = FinPiMLPSample(net, 400)
# print("pre loading", (new_finnet(data)  - finnet(data)).sum())
# new_finnet.load_state_dict(torch.load('epoch0.th'))
# print("post loading", (new_finnet(data)  - finnet(data)).sum())

# model_copy.load_state_dict(torch.load('epoch0.th'))

# print((model_copy(data)  - init_pred).sum())
# 0/0