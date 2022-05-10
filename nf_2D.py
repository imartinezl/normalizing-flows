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

# %% FLOWS 2D

class Flow1d(nn.Module):
    def __init__(self, n_components):
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

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, hidden_layers=3, output_dim=None):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(hidden_layers - 1):
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

class ConditionalFlow1D(nn.Module):
    def __init__(self, n_components):
        super(ConditionalFlow1D, self).__init__()
        self.cdf = MLP(output_dim=n_components*3)

    def forward(self, x, condition):
        x = x.view(-1,1)
        mus, log_sigmas, weight_logits = torch.chunk(self.cdf(condition), 3, dim=1)
        weights = weight_logits.softmax(dim=1)
        distribution = Normal(mus, log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_dx

class Flow2d(nn.Module):
    def __init__(self, n_components):
        super(Flow2d, self).__init__()
        self.flow_dim1 = Flow1d(n_components)
        self.flow_dim2 = ConditionalFlow1D(n_components)

    def forward(self, x, y=None):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z1, log_dz1_by_dx1 = self.flow_dim1(x1)
        z2, log_dz2_by_dx2 = self.flow_dim2(x2, condition=x1)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        log_dz_dx = torch.cat([log_dz1_by_dx1.unsqueeze(1), log_dz2_by_dx2.unsqueeze(1)], dim=1)
        return z, log_dz_dx

class AffineTransform2D(nn.Module):
    def __init__(self, left, device, hidden_dim=64, hidden_layers=2):
        super(AffineTransform2D, self).__init__()
        self.mlp = MLP(2, hidden_dim, hidden_layers, 2)
        self.mask = torch.FloatTensor([1,0]) if left else torch.FloatTensor([0,1])
        self.mask = self.mask.view(1,-1)
        # self.mask = self.mask.to(device)
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, y=None):
        # x.size() is (B,2)
        x_masked = x * self.mask
        # log_scale and shift have size (B,1)
        log_scale, shift = self.mlp(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale
        # log_scale and shift have size (B,2)
        shift = shift  * (1-self.mask)
        log_scale = log_scale * (1-self.mask)
        z = x * torch.exp(log_scale) + shift
        return z, log_scale

    def backward(self, z, y=None):
        # z.size() is (B,2)
        z_masked = z * self.mask
        # log_scale and shift have size (B,1)
        log_scale, shift = self.mlp(z_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale
        # log_scale and shift have size (B,2)
        shift = shift  * (1-self.mask)
        log_scale = log_scale * (1-self.mask)       
        x = (z - shift) * torch.exp(-log_scale)
        return x, -log_scale

class CPABTransform2D(nn.Module):
    def __init__(self, left, hidden_dim=10, hidden_layers=3, tess_size=10, zero_boundary=True, device="cpu"):
        super(CPABTransform2D, self).__init__()
        self.T = cpab.Cpab(tess_size, backend="pytorch", device=device, zero_boundary=zero_boundary, basis="svd")
        # self.T.params.use_slow = True
        # self.theta = self.T.identity(n_sample=1, epsilon=0)
        # self.theta = nn.Parameter(self.theta, requires_grad=True)
        self.d = self.T.params.d

        self.mlp = MLP(1, hidden_dim, hidden_layers, self.d)
        # self.mlp.apply(self.mlp.init_weights) # initialize weights
        # self.mask = 0 if left else 1
        self.mask = torch.IntTensor([1,0]) if left else torch.IntTensor([0,1])
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        # self.mask = self.mask.to(device)
        # self.mask = self.mask.view(1,-1)

        # self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.b = nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, x, y=None):
        # eps = 1e-7
        # x = torch.clip(x, eps, 1-eps)

        x1, x2 = torch.index_select(x, 1, self.mask).chunk(chunks=2, dim=1)
        theta = self.mlp(x1)
        # theta = theta * self.a + self.b

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
        # theta = theta * self.a + self.b

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

# JACOBIAN IS NOT TRIANGULAR (BAD)
class CPABTransform2D_BIS(nn.Module):
    def __init__(self, left, hidden_dim=10, hidden_layers=3, tess_size=10, zero_boundary=True, device="cpu"):
        super(CPABTransform2D_BIS, self).__init__()
        
        self.T = cpab.Cpab(tess_size, backend="pytorch", device=device, zero_boundary=zero_boundary, basis="svd")
        self.d = self.T.params.d

        # self.theta = self.T.identity(n_sample=128, epsilon=0)
        # self.theta = nn.Parameter(self.theta, requires_grad=True)

        self.mlp1 = MLP(1, hidden_dim, hidden_layers, self.d)
        self.mlp2 = MLP(1, hidden_dim, hidden_layers, self.d)
        self.mask = torch.IntTensor([1,0]) if left else torch.IntTensor([0,1])
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        # self.mask = self.mask.to(device)


    def forward(self, x, y=None):
        eps = 1e-7
        x = torch.clip(x, eps, 1-eps)

        x1, x2 = torch.index_select(x, 1, self.mask).chunk(chunks=2, dim=1)

        theta2 = self.mlp2(x1)
        z2 = self.T.transform_grid(x2, theta2)

        theta1 = self.mlp1(z2)
        z1 = self.T.transform_grid(x1, theta1)
        z = torch.column_stack([z1, z2])

        dz_dx_1 = self.T.gradient_space(x1, theta1)
        dz_dx_2 = self.T.gradient_space(x2, theta2)
        dz_dx = torch.column_stack([dz_dx_1, dz_dx_2])
        log_dz_dx = dz_dx.log()

        z = torch.index_select(z, 1, self.mask)
        log_dz_dx = torch.index_select(log_dz_dx, 1, self.mask)

        return z, log_dz_dx

    def backward(self, z, y=None):
        z1, z2 = torch.index_select(z, 1, self.mask).chunk(chunks=2, dim=1)

        theta1 = self.mlp1(z2)
        x1 = self.T.transform_grid(z1, -theta1)

        theta2 = self.mlp2(x1)
        x2 = self.T.transform_grid(z2, -theta2)
        x = torch.column_stack([x1, x2])

        dx_dz_1 = self.T.gradient_space(z1, -theta1)
        dx_dz_2 = self.T.gradient_space(z2, -theta2)
        dx_dz = torch.column_stack([dx_dz_1, dx_dz_2])
        log_dx_dz = dx_dz.log()

        x = torch.index_select(x, 1, self.mask)
        log_dx_dz = torch.index_select(log_dx_dz, 1, self.mask)

        return x, log_dx_dz

# Reals R => [0,1] // opposite direction if reverse option
class LogitTransform2D(nn.Module):
    def __init__(self, d, reverse=False):
        super(LogitTransform2D, self).__init__()
        
        self.d = d

        self.k = torch.ones(self.d)
        self.x0 = torch.zeros(self.d)

        # self.k = nn.Parameter(torch.ones(d), requires_grad=True)
        # self.x0 = nn.Parameter(torch.zeros(d), requires_grad=True)

        if reverse:
            self.forward, self.backward = self.backward, self.forward

    def f(self, x):
        return 1 / (1 + torch.exp(-self.k * (x - self.x0)))

    def g(self, z):
        return self.x0 + torch.log( z/(1-z) ) / self.k

    def forward(self, x, y=None):
        z = self.f(x)
        dz_dx = self.k * self.f(x) * (1 - self.f(x))
        log_dz_dx = dz_dx.log()

        return z, log_dz_dx

    def backward(self, z, y=None):
        eps = 1e-3
        z = torch.clip(torch.zeros_like(z) + eps, torch.ones_like(z) - eps)
        x = self.g(z)
        dx_dz = 1 / (self.k * z * (1-z))
        log_dx_dz = dx_dz.log()
        log_dx_dz = -torch.log(self.k * z * (1-z))

        return x, log_dx_dz

# %% DATA GENERATION

np.random.seed(1)
torch.random.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# torch.backends.cudnn.benchmark = True

datasets = ["MOONS", "CIRCLES", "GAUSSIAN", "CRESCENT", 
    "CRESCENTCUBED", "SINEWAVE", "ABS", "SIGN", "TWOSPIRALS", 
    "CHECKERBOARD", "FOURCIRCLES", "DIAMOND", "FACE"]

dataset_name = "CHECKERBOARD"
toy_train_size = 25000
toy_test_size = 500
batch_size = 1024
batch_size = 2560

train_loader, test_loader = fetch_dataloaders(dataset_name, batch_size, device, toy_train_size, toy_test_size)

# %% MODEL

cpab_device = "gpu" if device.type == "cuda" else "cpu"

transforms = [
    # CPABTransform2D_BIS(True, device), CPABTransform2D_BIS(False, device), 
    # CPABTransform2D_BIS(True, device), CPABTransform2D_BIS(False, device), 
    CPABTransform2D(True, device=cpab_device, tess_size=20), CPABTransform2D(False, device=cpab_device, tess_size=20), 
    # CPABTransform2D(True, device=cpab_device), CPABTransform2D(False, device=cpab_device), 
    # AffineTransform2D(True, device), AffineTransform2D(False, device),
    # AffineTransform2D(True, device), AffineTransform2D(False, device),
    # AffineTransform2D(True, device), AffineTransform2D(False, device),
    # AffineTransform2D(True, device), AffineTransform2D(False, device),
    # AffineTransform2D(True, device), AffineTransform2D(False, device),
    # LogitTransform2D(2, True),
    Constraint()
    ]
model = NormalizingFlow(transforms)
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)

model.to(device)

h = 1e-4
low = torch.FloatTensor([0-h, 0-h]).to(device)
high = torch.FloatTensor([1+h, 1+h]).to(device)
target_distribution = Uniform(low, high)

low = torch.FloatTensor([0, 0]).to(device)
high = torch.FloatTensor([1, 1]).to(device)
target_distribution = Normal(low, high)

low = torch.FloatTensor([5, 5]).to(device)
high = torch.FloatTensor([5, 5]).to(device)
target_distribution = Beta(low, high)

# %% TRAINING

# epochs = 10
# lr = 5e-3
# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[ProfilerActivity.CUDA], 
#     profile_memory=False, record_shapes=False) as prof:
#     with record_function("model_inference"):
#         model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution)         
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# exit()
# %% TRAINING



import time
print("START")
start_time = time.time()
epochs = 500
lr = 1e-3
# CONFIGURATION
config = {
    "dataset_name": dataset_name,
    "epochs": epochs,
    "lr": lr,
}
model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution)
stop_time = time.time()
print("STOP")
print(stop_time-start_time)
# exit()

plot_loss(train_losses, test_losses)
plt.savefig("loss.pdf")
# %%
# test_log_likelihood = -test_losses[-1]

# scaler = np.mean(-np.log(train_loader.dataset.train_max - train_loader.dataset.train_min))
# test_log_likelihood_original = test_log_likelihood + scaler
# test_log_likelihood_original
# %% PLOTS

@torch.no_grad()
def plot_2D(model, train_loader, target_distribution, n_samples=2000):
    # TRANSFORMATION EVOLUTION
    n = 2

    # FORWARD X => Z
    fig, axs = plt.subplots(1, 2, figsize=(4*n, 4), sharey=False)
    
    x = train_loader.dataset.tensors[0]
    s = np.random.choice(range(x.shape[0]), n_samples, replace=False)
    x = x[s]
    z, log_dz_dx = model.forward(x)
    X = [x.cpu(), z.cpu()]

    for k in range(n):
        x = X[k]
        axs[k].scatter(x=x[:,0], y=x[:,1], s=1, c="black")
        # sns.rugplot(x=x[:,0], y=x[:,1], ax=axs[k], alpha=0.05, c="black")    
        axs[k].set_title(f"$p(x_{k})$")
        # axs[k].axis("equal")
        axs[k].axis("off")
    
    # BACKWARD Z => X
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
    
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
        g = axs[k].scatter(x=z[:,0], y=z[:,1], c=c, s=1, cmap=cmap)
        # sns.rugplot(x=z[:,0], y=z[:,1], ax=axs[k], alpha=0.05, c="black")    
        axs[k].set_title(f"$p(z_{k})$")
        # axs[k].axis("equal")
        axs[k].axis("off")
        # plt.colorbar(g, ax=axs[k])

from matplotlib.collections import LineCollection
def plot_grid(g, ax, **kwargs):
    segs1 = g
    segs2 = g.transpose(1,0)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

@torch.no_grad()
def plot_evolution_2D(model, train_loader, target_distribution, n_samples=2000):
    # TRANSFORMATION EVOLUTION
    n = len(model.transforms)+1

    # FORWARD X => Z
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
    plt.subplots_adjust(wspace=0)
    
    x = train_loader.dataset.tensors[0].to(train_loader.dataset.device)
    s = np.random.choice(range(x.shape[0]), n_samples, replace=False)
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
    g = torch.stack(torch.meshgrid(a, a), axis=2)
    xgrid = g.reshape(-1,2).to(x.device)
    XG = [xgrid.cpu()]

    for transform in model.transforms:
        zgrid, _ = transform.forward(xgrid)
        xgrid = zgrid
        XG.append(xgrid.cpu())
    
    for k in range(n):
        x = X[k]
        axs[k].scatter(x=x[:,0], y=x[:,1], s=1, c="black")
        # sns.rugplot(x=x[:,0], y=x[:,1], ax=axs[k], alpha=0.05, c="black")
        axs[k].set_title(f"$p(x_{k})$")
        # axs[k].axis("equal")
        axs[k].axis("off")

        xgrid = XG[k].reshape(m, m, 2)
        plot_grid(xgrid, axs[k], color="black", alpha=0.1)
    


    # BACKWARD Z => X
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
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
    g = torch.stack(torch.meshgrid(a, a), axis=2)
    zgrid = g.reshape(-1,2).to(z.device)
    ZG = [zgrid.cpu()]

    for transform in model.transforms[::-1]:
        xgrid, _ = transform.backward(zgrid)
        zgrid = xgrid
        ZG.append(zgrid.cpu())
    
    for k in range(n):
        z, pz = Z[k], PZ[k]
        c = pz.mean(1)
        
        palette = "magma"
        cmap = palette if c.max() == c.min() else palette + "_r"
        # s = axs[k].scatter(x=z[:,0], y=z[:,1], c=c, s=1, cmap=cmap)
        s = axs[k].scatter(x=z[:,0], y=z[:,1], c=c, s=1)
        # axs[k].hist2d(x=z[:,0].numpy(), y=z[:,1].numpy(), bins=40)
        # plt.colorbar(s, ax=axs[k])
        # sns.rugplot(x=z[:,0], y=z[:,1], ax=axs[k], alpha=0.05, c="black")    
        axs[k].set_title(f"$p(z_{k})$")
        # axs[k].axis("equal")
        axs[k].axis("off")

        zgrid = ZG[k].reshape(m, m, 2)
        plot_grid(zgrid, axs[k], color="black", alpha=0.1)

# plot_2D(model, train_loader, target_distribution, n_samples=2000)
plot_evolution_2D(model, train_loader, target_distribution, n_samples=1000)
plt.savefig("rest.pdf")
# %%
