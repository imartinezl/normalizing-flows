# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cpab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import normalize_01
from base import NormalizingFlow, train_and_eval, plot_loss

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta

# %% FLOWS 1D

class Flow1d(nn.Module):
    def __init__(self, n_components=5):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x, y=None):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_dx

class LogitTransform(nn.Module):
    def __init__(self, alpha):
        super(LogitTransform, self).__init__()
        self.alpha = alpha 

    def forward(self, x, y=None):
        x_new = self.alpha/2 + (1-self.alpha)*x 
        z = torch.log(x_new) - torch.log(1-x_new)
        log_dz_dx = torch.log(torch.FloatTensor([1-self.alpha])) - torch.log(x_new) - torch.log(1-x_new)
        return z, log_dz_dx

class FlowCPABLinear(nn.Module):
    def __init__(self, tess_size=5, zero_boundary=False, basis="svd", device="cpu"):
        super(FlowCPABLinear, self).__init__()
        
        self.T = cpab.Cpab(tess_size, backend="pytorch", device=device, zero_boundary=zero_boundary, basis=basis)
        # self.T.params.use_slow = True
        self.theta = self.T.identity(n_sample=1, epsilon=0)
        self.theta = nn.Parameter(self.theta, requires_grad=True)
        
        self.a = torch.tensor([1.0])
        self.b = torch.tensor([0.0])
        self.a = nn.Parameter(self.a, requires_grad=True)
        self.b = nn.Parameter(self.b, requires_grad=True)

        self.c = torch.tensor([1.0])
        self.d = torch.tensor([0.0])
        self.c = nn.Parameter(self.c, requires_grad=True)
        self.d = nn.Parameter(self.d, requires_grad=True)

   
    def forward(self, x, y=None):
        eps = 1e-7
        x = torch.clip(x, eps, 1-eps)
      
        y = self.a*x + self.b
        w = self.T.transform_grid(y, self.theta)[0]
        z = self.c*w + self.d

        dy_dx = self.a
        dw_dy = self.T.gradient_space(y, self.theta)[0]
        dz_dw = self.c
        
        dz_dx = dz_dw * dw_dy * dy_dx
        log_dz_dx = dz_dx.log()
        
        return z, log_dz_dx

    def backward(self, z, y=None):
        w = (z - self.d) / self.c
        y = self.T.transform_grid(w, -self.theta)[0]
        x = (y - self.b) / self.a
        
        dx_dy = 1.0 / self.a
        dy_dw = self.T.gradient_space(w, -self.theta)[0]
        dw_dz = 1.0 / self.c
        dx_dz = dx_dy * dy_dw * dw_dz
        log_dx_dz = dx_dz.log()

        return x, log_dx_dz

class FlowCPAB(nn.Module):
    def __init__(self, tess_size=5, zero_boundary=False, basis="svd", device="cpu"):
        super(FlowCPAB, self).__init__()

        self.T = cpab.Cpab(tess_size, backend="pytorch", device=device, zero_boundary=zero_boundary, basis=basis)
        self.theta = self.T.identity(n_sample=1, epsilon=0)
        self.theta = nn.Parameter(self.theta, requires_grad=True)
        
    def forward(self, x, y=None):
        eps = 1e-7
        x = torch.clip(x, eps, 1-eps)

        z = self.T.transform_grid(x.T, self.theta).T

        dz_dx = self.T.gradient_space(x.T, self.theta).T
        log_dz_dx = dz_dx.log()
        return z, log_dz_dx

    def backward(self, z, y=None):
        x = self.T.transform_grid(z.T, -self.theta).T
        
        dx_dz = self.T.gradient_space(z.T, -self.theta).T
        log_dx_dz = dx_dz.log()

        return x, log_dx_dz

class FlowLinear(nn.Module):
    def __init__(self):
        super(FlowLinear, self).__init__()
        self.a = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, x, y=None):
        z = self.a*x + self.b
        log_dz_dx = (torch.ones_like(x)*self.a).log()
        return z, log_dz_dx

    def backward(self, z, y=None):
        x = (z - self.b) / self.a
        log_dx_dz = (torch.ones_like(x)/self.a).log()
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

def make_uniform(n):
    return np.random.uniform(0, 1, n)

def make_gaussian_mixture(n):
    m = n // 2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(m,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n-m,))
    return np.concatenate([gaussian1, gaussian2])

def make_gaussian(n):
    return np.random.normal(4.0, 1.0, n)

def make_power(n):
    return np.random.power(5, n)

import sklearn
def make_blobs(n):
    x, y = sklearn.datasets.make_blobs(n_samples=n, n_features=1, shuffle=True)
    return x

def fetch_dataloaders(generator, batch_size, device, toy_train_size, toy_test_size):

    train_x = generator(toy_train_size)
    test_x = generator(toy_test_size)

    # normalize
    train_min = train_x.min(0)
    train_max = train_x.max(0)

    train_x = normalize_01(train_x, train_min, train_max)
    test_x = normalize_01(test_x, train_min, train_max)

    # construct datasets
    train_dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)))
    test_dataset  = TensorDataset(torch.from_numpy(test_x.astype(np.float32)))
    train_dataset.device = device
    test_dataset.device = device

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader, test_loader


# %%
np.random.seed(1)
torch.random.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
batch_size = 128
toy_train_size, toy_test_size = 2000, 1000

generator = make_gaussian_mixture

train_loader, test_loader = fetch_dataloaders(generator, batch_size, device, toy_train_size, toy_test_size)
# %% MODEL

cpab_device = "gpu" if device.type == "cuda" else "cpu"

transforms = [FlowCPAB(100, False), FlowLinear()]
transforms = [FlowCPAB(10, False), Constraint()]
model = NormalizingFlow(transforms)
model.to(device)

eps = 1e-5
target_distribution = Uniform(0.0-eps, 1.0+eps)

mu = torch.tensor([0.0]).to(device)
sd = torch.tensor([1.0]).to(device)
target_distribution = Normal(mu, sd)

alpha = torch.tensor([5.0]).to(device)
beta = torch.tensor([5.0]).to(device)
target_distribution = Beta(alpha, beta)

# %% TRAINING

epochs = 150
lr = 5e-3
model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution)
plot_loss(train_losses, test_losses)


# %% PLOTS
from matplotlib.collections import LineCollection
def plot_grid(g, ax, **kwargs):
    segs1 = g
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.autoscale()

@torch.no_grad()
def plot_evolution_1D(model, train_loader, target_distribution, n_samples=2000):
    # TRANSFORMATION EVOLUTION
    n = len(model.transforms)+1

    # FORWARD X => Z
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
    plt.subplots_adjust(wspace=0)
    
    x = train_loader.dataset.tensors[0]
    s = np.random.choice(range(x.shape[0]), n_samples, replace=False)
    x = x[s]
    xmin, xmax = x.min(), x.max()
    X = [x.flatten().cpu()]

    for transform in model.transforms:
        z, _ = transform.forward(x)
        
        # new input x is z
        x = z
        X.append(x.flatten().cpu())

    # FORWARD GRID
    m = 30
    xgrid = torch.linspace(xmin, xmax, m).to(x.device)
    XG = [xgrid.flatten().cpu()]

    for transform in model.transforms:
        zgrid, _ = transform.forward(xgrid)
        xgrid = zgrid
        XG.append(xgrid.flatten().cpu())

    for k in range(n):
        x = X[k]
        s = sns.kdeplot(x=x, ax=axs[k], label="kernel")
        # sns.rugplot(x=x[:,0], y=x[:,1], ax=axs[k], alpha=0.05, c="black")
        axs[k].set_title(f"$p(x_{k})$")
        axs[k].axis("off")

        xgrid = XG[k]
        ymin, ymax = axs[k].get_ylim()
        xv = torch.linspace(ymin, ymax, m)
        g = torch.stack(torch.meshgrid(xgrid, xv), axis=2)
        plot_grid(g, axs[k], color="black", alpha=0.1)
    plt.legend()


    # BACKWARD Z => X
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
    plt.subplots_adjust(wspace=0)
    
    Z, PZ = [], []
    z = target_distribution.sample([n_samples])
    zmin, zmax = z.min(), z.max()
    pz = target_distribution.log_prob(z).exp()
    Z.append(z.flatten().cpu())
    PZ.append(pz.flatten().cpu())

    for transform in model.transforms[::-1]:
        x, log_dx_dz = transform.backward(z)
        zcopy, log_dz_dx = transform.forward(x)
        px = (pz.log() + log_dz_dx).exp()
        
        # new input z is x
        z = x
        pz = px
        Z.append(z.flatten().cpu())
        PZ.append(pz.flatten().cpu())

    # BACKWARD GRID
    m = 30
    zgrid = torch.linspace(zmin,zmax,m).to(x.device)
    ZG = [zgrid.flatten().cpu()]

    for transform in model.transforms[::-1]:
        xgrid, _ = transform.backward(zgrid)
        zgrid = xgrid
        ZG.append(zgrid.flatten().cpu())
   
    for k in range(n):
        z, pz = Z[k], PZ[k]
        zs, zi = torch.sort(z)
        c = pz
        
        palette = "magma"
        cmap = palette if c.max() == c.min() else palette + "_r"
        sns.kdeplot(x=z, ax=axs[k], label="kernel")
        axs[k].plot(zs, pz[zi], label="flow")
        # sns.rugplot(x=z[:,0], y=z[:,1], ax=axs[k], alpha=0.05, c="black")    
        axs[k].set_title(f"$p(z_{k})$")
        axs[k].axis("off")

        zgrid = ZG[k]
        ymin, ymax = axs[k].get_ylim()
        zv = torch.linspace(ymin, ymax, m)
        g = torch.stack(torch.meshgrid(zgrid, zv), axis=2)
        plot_grid(g, axs[k], color="black", alpha=0.1)

    plt.legend()

plot_evolution_1D(model, train_loader, target_distribution, n_samples=1000)

# %%
@torch.no_grad()
def plot_1D(model, train_loader, target_distribution, n_samples=2000):
    # sampling: sample z, then generative model gives x    
    z = target_distribution.sample([n_samples])
    x, log_dx_dz = model.backward(z)

    # density estimation: known x, obtain z and log_dz_dx, estimate p(x)
    z_copy, log_dz_dx = model.forward(x)
    pz = target_distribution.log_prob(z).exp()
    px = (pz.log() + log_dz_dx).exp()

    x = x.flatten().cpu()
    z = z.flatten().cpu()
    pz = pz.flatten().cpu()
    px = px.flatten().cpu()

    zs, zi = torch.sort(z)
    xs, xi = torch.sort(x)

    fig, axs = plt.subplots(1, 3, figsize=(9,3), constrained_layout=True)
    sns.kdeplot(z, ax=axs[0], label="empirical")
    axs[0].plot(zs, pz[zi], label="estimation")
    axs[0].axvline(z.mean(), ls='--', c='k')
    # axs[0].axvline(z[pz.argmax()], ls='--', c='k')
    axs[0].set_ylim(0, None)
    axs[0].set_xlabel("z")
    axs[0].set_ylabel("p(z)")
    axs[0].legend()

    sns.kdeplot(x, ax=axs[1], label="empirical")
    axs[1].plot(xs, px[xi], label="estimation")
    axs[1].axvline(x.mean(), ls='--', c='k')
    # axs[1].axvline(x[px.argmax()], ls='--', c='k')
    axs[0].set_ylim(0, None)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("p(x)")
    
    axs[2].plot(xs, z[xi])
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("z")

plot_1D(model, train_loader, target_distribution)
# %%