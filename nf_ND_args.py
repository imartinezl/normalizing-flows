# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cpab
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta

from data import fetch_dataloaders
from base import NormalizingFlow, train_and_eval, plot_loss


# %% ARGUMENTS

import argparse
parser = argparse.ArgumentParser(description='Normalizing flows ND')

parser.add_argument('--folder', type=str, help='results folder')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--train-size', type=int, help='train size')
parser.add_argument('--test-size', type=int, help='test size')
parser.add_argument('--batch-size', type=int, help='batch size')
parser.add_argument('--hidden-dim', type=int, help='hidden dim')
parser.add_argument('--hidden-layers', type=int, help='hidden layers')
parser.add_argument('--tess-size', type=int, help='tessellation size')
parser.add_argument('--flow-steps', type=int, help='flow steps')
parser.add_argument('--epochs', type=int, help='epochs')
parser.add_argument('--lr', type=float, help='learning rate')

# %% PATH
import time
def now():
    return round(time.time() * 1000)

args = parser.parse_args()
import sys
print("python " + " ".join(sys.argv))

import os
path = os.path.join(args.folder, args.dataset, str(now()))
from pathlib import Path
Path(path).mkdir(parents=True, exist_ok=True)

# %% CONFIGURATION
import json
with open(os.path.join(path, 'config.json'), 'w') as fp:
    json.dump(vars(args), fp)

# %% FLOWS ND

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(hidden_layers):
            layers.append( nn.Linear(hidden_dim, hidden_dim) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_dim, output_dim) )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.0)


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
    def __init__(self, dim, hidden_dim = 8, hidden_layers=3):
        super().__init__()
        self.dim = dim
        lower_dim = dim // 2
        upper_dim = dim - lower_dim
        # self.t1 = FCNN(lower_dim, hidden_dim, upper_dim)
        # self.s1 = FCNN(lower_dim, hidden_dim, upper_dim)
        # self.t2 = FCNN(upper_dim, hidden_dim, lower_dim)
        # self.s2 = FCNN(upper_dim, hidden_dim, lower_dim)
        self.t1 = MLP(lower_dim, hidden_dim, hidden_layers, upper_dim)
        self.s1 = MLP(lower_dim, hidden_dim, hidden_layers, upper_dim)
        self.t2 = MLP(upper_dim, hidden_dim, hidden_layers, lower_dim)
        self.s2 = MLP(upper_dim, hidden_dim, hidden_layers, lower_dim)

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

    def backward(self, z, y=None):
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

    def backward(self, z, y=None):
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

class Constraint(nn.Module):
    def __init__(self, lower=0, upper=1, eps=1e-7):
        super(Constraint, self).__init__()
        self.lower = lower
        self.upper = upper
        self.eps = eps

    def forward(self, x, y=None):
        lower = self.lower + self.eps
        upper = self.upper - self.eps
        z = torch.clip(x, lower, upper)
        log_dz_dx = torch.zeros_like(x)
        return z, log_dz_dx

    def backward(self, z, y=None):
        x = z
        log_dx_dz = torch.zeros_like(z)
        return x, log_dx_dz

# %% DATA GENERATION

np.random.seed(1)
torch.random.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ["SCURVE", "SWISSROLL", "POWER", "MNIST", "MINIBOONE", "HEPMASS", "GAS", "BSDS300"]


dataset_name = args.dataset
toy_train_size = args.train_size
toy_test_size = args.test_size
batch_size = args.batch_size

train_loader, test_loader = fetch_dataloaders(dataset_name, batch_size, device, toy_train_size, toy_test_size)
d = train_loader.dataset.input_dims

# %% MODEL

cpab_device = "gpu" if device.type == "cuda" else "cpu"

transforms = []
for i in range(args.flow_steps):
    #kwargs = {'hidden_dim':args.hidden_dim, 'hidden_layers':args.hidden_layers, 'tess_size':args.tess_size, 'zero_boundary':True, 'device':cpab_device}
    kwargs = {'hidden_dim':args.hidden_dim}
    m = RealNVP(d, **kwargs)
    transforms.append(m)
transforms.append(Constraint())    

model = NormalizingFlow(transforms)
model.to(device)

# low = torch.FloatTensor(torch.zeros(d)).to(device)
# high = torch.FloatTensor(torch.ones(d)).to(device)
# target_distribution = Normal(low, high)

alpha = 12
low = torch.FloatTensor(torch.ones(d)*alpha).to(device)
high = torch.FloatTensor(torch.ones(d)*alpha).to(device)
target_distribution = Beta(low, high)

# %% TRAINING

epochs = args.epochs
lr = args.lr
model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution, path)

fig, ax = plt.subplots(1, figsize=(6,4))
plot_loss(train_losses, test_losses, ax)
fig.savefig(os.path.join(path, "loss.pdf"), bbox_inches="tight")

# %% LOAD BEST

checkpoint = torch.load(os.path.join(path, "epoch_best.pth"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# %% PLOTS

if d > 3:
    exit()

@torch.no_grad()
def plot_3D(model, train_loader, target_distribution, path, n_samples=2000):
    # TRANSFORMATION EVOLUTION
    n = 2

    # FORWARD X => Z
    fig, axs = plt.subplots(3, 2, figsize=(4*n, 4*3), sharey=False)
    
    x = train_loader.dataset.tensors[0]
    
    s = np.random.choice(range(x.shape[0]), min(x.shape[0], n_samples), replace=False)
    x = x[s]
    z, log_dz_dx = model.forward(x)
    X = [x.cpu(), z.cpu()]

    for k in range(n):
        x = X[k]
        q = np.array([0,1,2])
        for p in range(3):
            q[2-p], q[2] = q[2], q[2-p]
            i, j, e = q

            # di vs dj
            axs[p][k].scatter(x=x[:,i], y=x[:,j], s=1, c="black")
            axs[p][k].set_xlabel(f"$d_{i}$", labelpad=0)
            axs[p][k].set_ylabel(f"$d_{j}$", labelpad=0)
            axs[p][k].set_frame_on(False)
            axs[p][k].set_xticks([])
            axs[p][k].set_yticks([])

    axs[0][0].set_title(f"$p(x)$")
    axs[0][1].set_title(f"$p(z)$")

    # SAVE
    fig.savefig(os.path.join(path, "plot_normalizing_flow.pdf"), bbox_inches="tight")
    
    # BACKWARD Z => X
    fig, axs = plt.subplots(3, n, figsize=(4*n, 4*3), sharey=False)
    
    Z, PZ = [], []
    z = target_distribution.sample([n_samples])
    pz = target_distribution.log_prob(z).exp()

    x, log_dx_dz = model.backward(z)
    zcopy, log_dz_dx = model.forward(x)
    px = (pz.log() + log_dz_dx).exp()
    Z = [z.cpu(), x.cpu()]
    PZ = [pz.cpu(), px.cpu()]
    
    for k in range(n):
        z, pz = Z[k], PZ[k]
        c = pz.mean(1)
        
        palette = "magma"
        cmap = palette if c.max() == c.min() else palette + "_r"

        q = np.array([0,1,2])
        for p in range(3):
            q[2-p], q[2] = q[2], q[2-p]
            i, j, e = q

            # di vs dj
            axs[p][k].scatter(x=z[:,i], y=z[:,j], c=c, s=1, cmap=cmap)
            axs[p][k].set_xlabel(f"$d_{i}$", labelpad=0)
            axs[p][k].set_ylabel(f"$d_{j}$", labelpad=0)
            axs[p][k].set_frame_on(False)
            axs[p][k].set_xticks([])
            axs[p][k].set_yticks([])

        axs[0][k].set_title(f"$p(z_{k})$")

    axs[0][0].set_title(f"$p(z)$")
    axs[0][1].set_title(f"$p(x)$")

    # SAVE
    fig.savefig(os.path.join(path, "plot_generative_flow.pdf"), bbox_inches="tight")

from matplotlib.collections import LineCollection
def plot_grid(g, ax, **kwargs):
    segs1 = g
    segs2 = g.transpose(1,0)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

@torch.no_grad()
def plot_evolution_3D(model, train_loader, target_distribution, path, n_samples=2000):
    # TRANSFORMATION EVOLUTION
    n = len(model.transforms)

    # FORWARD X => Z
    fig, axs = plt.subplots(3, n, figsize=(4*n, 4*3), sharey=False)
    plt.subplots_adjust(wspace=0.05)
    
    x = train_loader.dataset.tensors[0].to(train_loader.dataset.device)
    s = np.random.choice(range(x.shape[0]), min(x.shape[0], n_samples), replace=False)
    x = x[s]
    xmin, xmax = x.min(), x.max()
    X = [x.cpu()]


    for transform in model.transforms:
        z, _ = transform.forward(x)
        
        # new input x is z
        x = z
        X.append(x.cpu())

    # FORWARD GRID
    m = 20
    a = torch.linspace(xmin, xmax, m)
    g = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), axis=3)
    xgrid = g.reshape(-1,3).to(x.device)
    XG = [xgrid.cpu()]

    # grid position
    b = torch.linspace(1, m, m)
    ref = torch.stack(torch.meshgrid(b, b, b, indexing="ij"), axis=3).reshape(-1,3)

    for transform in model.transforms:
        zgrid, _ = transform.forward(xgrid)
        xgrid = zgrid
        XG.append(xgrid.cpu())

    for k in range(n):
        x = X[k]

        q = np.array([0,1,2])
        for p in range(3):
            q[2-p], q[2] = q[2], q[2-p]
            i, j, e = q

            # di vs dj
            axs[p][k].scatter(x=x[:,i], y=x[:,j], s=1, c="black")
            axs[p][k].set_xlabel(f"$d_{i}$", labelpad=0)
            axs[p][k].set_ylabel(f"$d_{j}$", labelpad=0)
            axs[p][k].set_frame_on(False)
            axs[p][k].set_xticks([])
            axs[p][k].set_yticks([])

            idx = torch.nonzero(ref[:,e] == (m//2)).flatten()
            xgrid = XG[k][idx][:,[i,j]].reshape(m, m, 2)
            plot_grid(xgrid, axs[p][k], color="black", alpha=0.1)

        axs[0][k].set_title(f"$p(x_{k})$")
        
    # SAVE
    fig.savefig(os.path.join(path, "plot_normalizing_flow_evolution.pdf"), bbox_inches="tight")

    # BACKWARD Z => X
    fig, axs = plt.subplots(3, n, figsize=(4*n, 4*3), sharey=False)
    plt.subplots_adjust(wspace=0)

    
    Z, PZ = [], []
    z = target_distribution.sample([n_samples])
    zmin, zmax = z.min(), z.max()
    pz = target_distribution.log_prob(z).exp()
    Z.append(z.cpu())
    PZ.append(pz.cpu())

    for transform in model.transforms[::-1]:
        x, log_dx_dz = transform.backward(z)
        zcopy, log_dz_dx = transform.forward(x)
        px = (pz.log() + log_dz_dx).exp()
        
        # new input z is x
        z = x
        pz = px
        Z.append(z.cpu())
        PZ.append(pz.cpu())


    # BACKWARD GRID
    m = 20
    a = torch.linspace(zmin, zmax, m)
    g = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), axis=3)
    zgrid = g.reshape(-1,3).to(z.device)
    ZG = [zgrid.cpu()]

    for transform in model.transforms[::-1]:
        xgrid, _ = transform.backward(zgrid)
        zgrid = xgrid
        ZG.append(zgrid.cpu())
    
    for k in range(n):
        z, pz = Z[k+1], PZ[k+1]
        c = pz.mean(1)
        
        palette = "magma"
        cmap = palette if c.max() == c.min() else palette + "_r"

        q = np.array([0,1,2])
        for p in range(3):
            q[2-p], q[2] = q[2], q[2-p]
            i, j, e = q

            # di vs dj
            axs[p][k].scatter(x=z[:,i], y=z[:,j], s=1, c="black")
            axs[p][k].set_xlabel(f"$d_{i}$", labelpad=0)
            axs[p][k].set_ylabel(f"$d_{j}$", labelpad=0)
            axs[p][k].set_frame_on(False)
            axs[p][k].set_xticks([])
            axs[p][k].set_yticks([])

            idx = torch.nonzero(ref[:,e] == (m//2)).flatten()
            xgrid = ZG[k+1][idx][:,[i,j]].reshape(m, m, 2)
            plot_grid(xgrid, axs[p][k], color="black", alpha=0.1)

        axs[0][k].set_title(f"$p(z_{k})$")

    # SAVE
    fig.savefig(os.path.join(path, "plot_generative_flow_evolution.pdf"), bbox_inches="tight")

plot_3D(model, train_loader, target_distribution, path, n_samples=3000)

plot_evolution_3D(model, train_loader, target_distribution, path, n_samples=3000)

# %%
