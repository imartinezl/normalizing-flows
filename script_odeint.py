# %%

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# %%

from torchdiffeq import odeint # backpropagation goes through solver internals 

def func(t, y):
    return 2*y

y0 = torch.ones(1)
t = torch.linspace(0,1,10)
odeint(func, y0, t)

# %% 1D flow

from torchdiffeq import odeint_adjoint as odeint # use the adjoint method

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()


    def forward(self, t, y):
        return 0.25*y

y0 = torch.linspace(0, 1, 50)
t = torch.linspace(0,1,2)

func = ODEFunc()
flow = odeint(func, y0, t)

@torch.no_grad()
def plot():
    plt.plot(flow[0], flow[1])
    plt.axis("equal")
plot()

# %% 2D flow

from torchdiffeq import odeint_adjoint as odeint # use the adjoint method

class ODEFunc(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=-0.5)

    def forward(self, t, y):
        return self.net(y)

grid = torch.linspace(0,1,20)
y0 = torch.cartesian_prod(grid, grid)
t = torch.linspace(0,1,2)

func = ODEFunc(2, 5)
flow = odeint(func, y0, t)


@torch.no_grad()
def plot():
    plt.figure()
    x = flow[0][:,0]
    y = flow[0][:,1]
    u = flow[1][:,0]
    v = flow[1][:,1]
    c = np.arctan2(u, v)
    plt.quiver(x, y, u, v, c)
plot()

