# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cpab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import fetch_dataloaders
from base import NormalizingFlow, train_and_eval, plot_loss

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal

# %% FLOWS ND

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

class RealNVP(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 8):
        super().__init__()
        self.dim = dim
        lower_dim = dim // 2
        upper_dim = dim - lower_dim
        self.t1 = FCNN(lower_dim, hidden_dim, upper_dim)
        self.s1 = FCNN(lower_dim, hidden_dim, upper_dim)
        self.t2 = FCNN(upper_dim, hidden_dim, lower_dim)
        self.s2 = FCNN(upper_dim, hidden_dim, lower_dim)

    def forward(self, x, y=None):
        lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)

        log_det = torch.sum(s1_transformed, dim=1) + torch.sum(s2_transformed, dim=1)
        log_det = torch.cat([s1_transformed, s2_transformed], dim=1)
        return z, log_det

    def inverse(self, z, y=None):
        lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + torch.sum(-s2_transformed, dim=1)
        return x, log_det


import math
class MAF(nn.Module):
    """
    Masked auto-regressive flow.
    [Papamakarios et al. 2018]
    """
    def __init__(self, dim, hidden_dim = 8):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        for i in range(1, dim):
            self.layers += [FCNN(i, hidden_dim, 2)]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, y=None):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def inverse(self, z, y=None):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            log_det += alpha
        return x, log_det


class CPABTransform(nn.Module):

    def __init__(self, left, device, hidden_dim=10, hidden_layers=3, tess_size=10, zero_boundary=True):
        super(CPABTransform, self).__init__()
        cpab_device = "cpu" if device.type == "cpu" else "gpu"
        self.T = cpab.Cpab(tess_size, backend="pytorch", device=cpab_device, zero_boundary=zero_boundary, basis="svd")
        # self.theta = self.T.identity(n_sample=1, epsilon=0)
        # self.theta = nn.Parameter(self.theta, requires_grad=True)
        self.d = self.T.params.d

        self.mlp = MLP(1, hidden_dim, hidden_layers, self.d)
        self.mask = torch.IntTensor([1,0]) if left else torch.IntTensor([0,1])
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        # self.mask = self.mask.to(device)


    def forward(self, x, y=None):
        eps = 1e-7
        x = torch.clip(x, eps, 1-eps)

        x1, x2 = torch.index_select(x, 1, self.mask).chunk(chunks=2, dim=1)
        theta = self.mlp(x1)

        z1 = x1
        z2 = self.T.transform_grid(x2, theta)
        z = torch.column_stack([z1, z2])

        dz_dx_1 = torch.ones_like(x1)
        dz_dx_2 = self.T.gradient_space(x2, theta)
        dz_dx = torch.column_stack([dz_dx_1, dz_dx_2])
        log_dz_dx = dz_dx.log()

        z = torch.index_select(z, 1, self.mask)
        log_dz_dx = torch.index_select(log_dz_dx, 1, self.mask)

        return z, log_dz_dx

    def backward(self, z, y=None):
        z1, z2 = torch.index_select(z, 1, self.mask).chunk(chunks=2, dim=1)
        theta = self.mlp(z1)

        x1 = z1
        x2 = self.T.transform_grid(z2, -theta)
        x = torch.column_stack([x1, x2])

        dx_dz_1 = torch.ones_like(z1)
        dx_dz_2 = self.T.gradient_space(z2, -theta)
        dx_dz = torch.column_stack([dx_dz_1, dx_dz_2])
        log_dx_dz = dx_dz.log()

        x = torch.index_select(x, 1, self.mask)
        log_dx_dz = torch.index_select(log_dx_dz, 1, self.mask)

        return x, log_dx_dz


# %% DATA GENERATION

np.random.seed(1)
torch.random.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ["SCURVE", "SWISSROLL", "POWER", "MNIST", "MINIBOONE", "HEPMASS", "GAS", "BSDS300"]

dataset_name = "POWER"
toy_train_size = 2500
toy_test_size = 500
batch_size = 1000

train_loader, test_loader = fetch_dataloaders(dataset_name, batch_size, device, toy_train_size, toy_test_size)
d = train_loader.dataset.input_dims

# %% MODEL

cpab_device = "gpu" if device.type == "cuda" else "cpu"

transforms = [
    RealNVP(d),
    CPABTransform(left, device)(d),
    MAF(d),
]
model = NormalizingFlow(transforms)
model.to(device)

# target_distribution = MultivariateNormal(torch.zeros(d), torch.eye(d))
low = torch.FloatTensor(torch.zeros(d)).to(device)
high = torch.FloatTensor(torch.ones(d)).to(device)
target_distribution = Normal(low, high)

# low = torch.FloatTensor([5, 5]).to(device)
# high = torch.FloatTensor([5, 5]).to(device)
# target_distribution = Beta(low, high) # for RealNVP

# alpha = torch.FloatTensor(torch.ones(d)*5).to(device)
# target_distribution = Dirichlet(alpha)

# %% TRAINING

epochs = 1000
lr = 5e-3
model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution)
plot_loss(train_losses, test_losses)

# %%
test_log_likelihood = -test_losses[-1]

scaler = np.mean(-np.log(train_loader.dataset.train_max - train_loader.dataset.train_min))
test_log_likelihood_original = test_log_likelihood + scaler
test_log_likelihood_original
# %% PLOTS

d = 2
t1 = MultivariateNormal(torch.zeros(d), torch.eye(d))
low = torch.FloatTensor(torch.zeros(d)).to(device)
high = torch.FloatTensor(torch.ones(d)).to(device)
t2 = Normal(low, high)

x = torch.zeros(d)
t1.log_prob(x).exp(), t2.log_prob(x).exp()