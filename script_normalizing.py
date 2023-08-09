# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta

# %% DATA
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons, make_circles, make_blobs, make_s_curve, make_swiss_roll

class NumpyDataset(Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array.astype(np.float32)
        self.array_min, self.array_max = self.array.min(axis=0), self.array.max(axis=0)
        self.array = normalize_01(self.array)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

def normalize_01(x):
    x -= x.min(axis=0)
    x /= x.max(axis=0)
    return x

def normalize_z(x):
    x -= x.mean()
    x /= x.std()
    return x

def collate_fn(batch):
    batch = np.array(batch, dtype=np.float32)
    batch = normalize_01(batch)
    batch = torch.Tensor(batch)
    return batch



# %% FLOWS 1D

class Flow1d(nn.Module):
    def __init__(self, n_components=5):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
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

    def forward(self, x):
        x_new = self.alpha/2 + (1-self.alpha)*x 
        z = torch.log(x_new) - torch.log(1-x_new)
        log_dz_dx = torch.log(torch.FloatTensor([1-self.alpha])) - torch.log(x_new) - torch.log(1-x_new)
        return z, log_dz_dx
        
class FlowComposable1d(nn.Module):
    def __init__(self, models_list):
        super(FlowComposable1d, self).__init__()
        self.models_list = nn.ModuleList(models_list)

    def forward(self, x):
        z, sum_log_dz_dx = x, 0
        for model in self.models_list:
            z, log_dz_dx = model(z)
            sum_log_dz_dx += log_dz_dx
        return z, sum_log_dz_dx

import cpab
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

   
    def forward(self, x):
        y = self.a*x + self.b
        w = self.T.transform_grid(y, self.theta)[0]
        z = self.c*w + self.d

        dy_dx = self.a
        dw_dy = self.T.gradient_space(y, self.theta)[0]
        dz_dw = self.c
        
        dz_dx = dz_dw * dw_dy * dy_dx
        log_dz_dx = dz_dx.log()
        
        return z, log_dz_dx

    def backward(self, z):
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
        
    def forward(self, x):
        z = self.T.transform_grid(x, self.theta)[0]

        dz_dx = self.T.gradient_space(x, self.theta)[0]
        log_dz_dx = dz_dx.log()
        return z, log_dz_dx

    def backward(self, z):
        x = self.T.transform_grid(z, -self.theta)[0]
        
        dx_dz = self.T.gradient_space(z, -self.theta)[0]
        log_dx_dz = dx_dz.log()

        return x, log_dx_dz

class FlowLinear(nn.Module):
    def __init__(self):
        super(FlowLinear, self).__init__()
        self.a = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, x):
        z = self.a*x + self.b
        log_dz_dx = (torch.ones_like(x)*self.a).log()
        return z, log_dz_dx

    def backward(self, z):
        x = (z - self.b) / self.a
        log_dx_dz = (torch.ones_like(x)/self.a).log()
        return x, log_dx_dz


# %% FLOWS 2D

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_dx

class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_hidden_layers=3, output_size=None):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append( nn.Linear(hidden_size, hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_size, output_size) )
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
        self.cdf = MLP(output_size=n_components*3)

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

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z1, log_dz1_by_dx1 = self.flow_dim1(x1)
        z2, log_dz2_by_dx2 = self.flow_dim2(x2, condition=x1)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        log_dz_dx = torch.cat([log_dz1_by_dx1.unsqueeze(1), log_dz2_by_dx2.unsqueeze(1)], dim=1)
        return z, log_dz_dx

class AffineTransform2D(nn.Module):
    def __init__(self, left, hidden_size=64, num_hidden_layers=2):
        super(AffineTransform2D, self).__init__()
        self.mlp = MLP(2, hidden_size, num_hidden_layers, 2)
        self.mask = torch.FloatTensor([1,0]) if left else torch.FloatTensor([0,1])
        self.mask = self.mask.view(1,-1)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
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

    def backward(self, z):
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
    def __init__(self, left, hidden_size=10, num_hidden_layers=3, tess_size=10, zero_boundary=True):
        super(CPABTransform2D, self).__init__()
        self.T = cpab.Cpab(tess_size, backend="pytorch", device="cpu", zero_boundary=zero_boundary, basis="svd")
        # self.T.params.use_slow = True
        # self.theta = self.T.identity(n_sample=1, epsilon=0)
        # self.theta = nn.Parameter(self.theta, requires_grad=True)
        self.d = self.T.params.d

        self.mlp = MLP(1, hidden_size, num_hidden_layers, self.d)
        # self.mlp.apply(self.mlp.init_weights) # initialize weights
        # self.mask = 0 if left else 1
        self.mask = torch.IntTensor([1,0]) if left else torch.IntTensor([0,1])
        # self.mask = self.mask.view(1,-1)

        # self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.b = nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, x):
        eps = 1e-7
        x = torch.clip(x, eps, 1-eps)

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

    def backward(self, z):
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


# JACOBIAN IS NOT TRIANGULAR (BAD)
class CPABTransform2D_BIS(nn.Module):
    def __init__(self, left, hidden_size=10, num_hidden_layers=3, tess_size=10, zero_boundary=True):
        super(CPABTransform2D_BIS, self).__init__()
        self.T = cpab.Cpab(tess_size, backend="pytorch", device="cpu", zero_boundary=zero_boundary, basis="svd")
        self.d = self.T.params.d

        # self.theta = self.T.identity(n_sample=128, epsilon=0)
        # self.theta = nn.Parameter(self.theta, requires_grad=True)

        self.mlp1 = MLP(1, hidden_size, num_hidden_layers, self.d)
        self.mlp2 = MLP(1, hidden_size, num_hidden_layers, self.d)
        self.mask = torch.IntTensor([1,0]) if left else torch.IntTensor([0,1])


    def forward(self, x):
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

    def backward(self, z):
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

    def forward(self, x):
        z = self.f(x)
        dz_dx = self.k * self.f(x) * (1 - self.f(x))
        log_dz_dx = dz_dx.log()

        return z, log_dz_dx

    def backward(self, z):
        eps = 1e-3
        z = torch.clip(torch.zeros_like(z) + eps, torch.ones_like(z) - eps)
        x = self.g(z)
        dx_dz = 1 / (self.k * z * (1-z))
        log_dx_dz = dx_dz.log()
        log_dx_dz = -torch.log(self.k * z * (1-z))

        return x, log_dx_dz

class NormalizingFlow(nn.Module):
    def __init__(self, affine_transforms):
        super(NormalizingFlow, self).__init__()
        self.transforms = nn.ModuleList(affine_transforms)

    def forward(self, x):
        z, log_dz_dx_sum = x, torch.zeros_like(x)
        for transform in self.transforms:
            z, log_dz_dx = transform(z)
            log_dz_dx_sum += log_dz_dx
        return z, log_dz_dx_sum

    def backward(self, z):
        log_dx_dz_sum = torch.zeros_like(z)
        for transform in self.transforms[::-1]:
            z, log_dx_dz = transform.backward(z)
        return z, log_dx_dz_sum


# n = 1000
# d = 2
# x = torch.normal(0, 1, (n, d))
# x = torch.rand(n, d)

# model = LogitTransform2D(d, True)
# z, log_dz_dx = model.forward(x)

# with torch.no_grad():
#     plt.figure()
#     plt.scatter(x[:,0], x[:,1])
#     plt.axis("equal")
#     plt.figure()
#     plt.scatter(z[:,0], z[:,1])
#     plt.axis("equal")

# %% PIPELINE

def loss_function(model, target_distribution, z, log_dz_dx):
    # print("LOSS: ", torch.norm(z))
    # return torch.norm(z)
    # print("LOSS",  target_distribution.log_prob(z) )
    log_likelihood = target_distribution.log_prob(z) + log_dz_dx
    loss = -log_likelihood.mean()
    # reg = 0.0
    # for p in model.parameters():
    #     reg += 0.01*torch.norm(p)
    #     break
    # reg = torch.norm(model.get_parameter("theta"))
    # loss += reg
    return loss

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        z, log_dz_dx = model(x)
        loss = loss_function(model, target_distribution, z, log_dz_dx)
        # z.retain_grad()#
        # log_dz_dx.retain_grad()#
        # loss.retain_grad()#
        optimizer.zero_grad()
        loss.backward()
        # print("hey", optimizer.param_groups[0]["params"][0].grad)
        # print(list(model.parameters()))
        optimizer.step()
        # print("PARAMETERS:", list(model.parameters()) )
        # print("X:", x)
        # print("Z:", z)
        # print("LOGDZ:", log_dz_dx)
        # print("X GRAD: ", x.grad)
        # print("Z GRAD: ", z.grad)
        # print("LOGDZ GRAD: ", log_dz_dx.grad)
        # print("LOSS GRAD: ", loss.grad)
        # break
    return model
        
def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        z, log_dz_dx = model(x)
        loss = loss_function(model, target_distribution, z, log_dz_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            model = train(model, train_loader, optimizer, target_distribution)
            # break#
            train_loss = eval_loss(model, train_loader, target_distribution)
            test_loss = eval_loss(model, test_loader, target_distribution)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            pbar.set_description(f"Loss {train_loss:.4f}")
            pbar.update()
        pbar.close()
    return model, train_losses, test_losses

# %% PLOTS

@torch.no_grad()
def plot_1D(model, train_loader, target_distribution, sample=True):
    # sampling: known p(z), sample z, obtain x by inverse flow model
    if hasattr(model, "backward") and sample:
        n_samples = 1000
        z = target_distribution.sample([n_samples])
        x, log_dx_dz = model.backward(z)
    else:
        x = torch.FloatTensor(train_loader.dataset.array)
    
    # x = torch.FloatTensor(train_loader.dataset.array)

    # density estimation: known x, obtain z and log_dz_dx, estimate p(x)
    z, log_dz_dx = model(x)
    pz = target_distribution.log_prob(z).exp()
    px = (pz.log() + log_dz_dx).exp()

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

@torch.no_grad()
def plot_2D(model, train_loader, target_distribution, n_samples=2000):
    # TRANSFORMATION EVOLUTION
    n = 2

    # FORWARD X => Z
    fig, axs = plt.subplots(1, 2, figsize=(4*n, 4), sharey=False)
    
    x = train_loader.dataset.array[:n_samples]
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

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()

# %% 1D TEST

def make_uniform(n):
    return np.random.uniform(0, 1, n)

def make_gaussian_mixture(n):
    m = n // 2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(m,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n-m,))
    return np.concatenate([gaussian1, gaussian2])

def make_gaussian(n):
    return np.random.normal(4.0, 1.0, n)

if False:
    np.random.seed(1)
    torch.random.manual_seed(1)
    n_train, n_test = 2000, 1000
    generator = make_gaussian_mixture
    train_data = generator(n_train)
    test_data = generator(n_test)

    train_dataset = NumpyDataset(train_data)
    test_dataset = NumpyDataset(test_data)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # FLOW1D
    model = Flow1d(n_components=5)
    model = FlowLinear()
    model = FlowCPAB(tess_size=10, zero_boundary=False)
    model = FlowCPABLinear(tess_size=10, zero_boundary=True)

    # FLOWCOMPOSITION
    # models_list = [Flow1d(2), LogitTransform(0.1), Flow1d(2), LogitTransform(0.1), Flow1d(2)]
    models_list = [FlowCPAB(10, False), FlowLinear()]
    model = FlowComposable1d(models_list)

    target_distribution = Beta(2.0, 2.0)
    eps = 1e-5
    target_distribution = Uniform(0.0-eps, 1.0+eps)
    target_distribution = Normal(0.0, 1.0)
    epochs = 150
    lr = 5e-3
    model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution)
    plot_loss(train_losses, test_losses)

    # print(model.__str__())
    # list(model.parameters())

    # plot_1D(model, train_loader, target_distribution, sample=True)
    plot_1D(model, train_loader, target_distribution, sample=False)



# %% 2D TEST
###########################################################################
np.random.seed(1)
torch.random.manual_seed(1)
n_train, n_test = 2000, 1000

dataset = "swiss_roll"

if dataset == "moons":
    # MOONS
    train_data, train_labels = make_moons(n_samples=n_train, noise=0.1)
    test_data, test_labels = make_moons(n_samples=n_test, noise=0.1)
elif dataset == "circles":
    # CIRCLES
    train_data, train_labels = make_circles(n_samples=n_train, noise=0.1)
    test_data, test_labels = make_circles(n_samples=n_test, noise=0.1)
elif dataset == "s_curve":
    # S-CURVE
    train_data, train_labels = make_s_curve(n_samples=n_train, noise=0.1)
    test_data, test_labels = make_s_curve(n_samples=n_test, noise=0.1)
    train_data = train_data[:,[0,2]]
    test_data = test_data[:,[0,2]]
elif dataset == "swiss_roll":
    # SWISS-ROLL
    train_data, train_labels = make_swiss_roll(n_samples=n_train, noise=0.5)
    test_data, test_labels = make_swiss_roll(n_samples=n_test, noise=0.5)
    train_data = train_data[:,[0,2]]
    test_data = test_data[:,[0,2]]

########################################
train_dataset = NumpyDataset(train_data)
test_dataset = NumpyDataset(test_data)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# FLOW2D
# model = Flow2d(n_components=5)

# REALNVP
transforms = [
    # CPABTransform2D_BIS(True), CPABTransform2D_BIS(False),
    # CPABTransform2D_BIS(True), CPABTransform2D_BIS(False),
    # CPABTransform2D_BIS(True), CPABTransform2D_BIS(False),
    CPABTransform2D(True), CPABTransform2D(False), 
    CPABTransform2D(True), CPABTransform2D(False), 
    # CPABTransform2D(True), CPABTransform2D(False), 
    # CPABTransform2D(True), CPABTransform2D(False), 
    # AffineTransform2D(True), AffineTransform2D(False),
    # AffineTransform2D(True), AffineTransform2D(False),
    # AffineTransform2D(True), AffineTransform2D(False),
    # LogitTransform2D(2, True),
    ]
model = NormalizingFlow(transforms)

h = 1e-4
low = torch.FloatTensor([0-h, 0-h])
high = torch.FloatTensor([1+h, 1+h])
target_distribution = Uniform(low, high) # for Flow2d

low = torch.FloatTensor([0, 0])
high = torch.FloatTensor([1, 1])
target_distribution = Normal(low, high) # for RealNVP

low = torch.FloatTensor([2, 2])
high = torch.FloatTensor([2, 2])
target_distribution = Beta(low, high) # for RealNVP

# %%
epochs = 75
lr = 1e-3
model, train_losses, test_losses = train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution)
plot_loss(train_losses, test_losses)

# %%
def plot_evolution_2D(model, train_loader, target_distribution):
    # TRANSFORMATION EVOLUTION
    with torch.no_grad():
        n = len(model.transforms)+1

        # FORWARD X => Z
        fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
        
        X = []
        x = torch.FloatTensor(train_loader.dataset.array)
        X.append(x)

        for transform in model.transforms:
            z, _ = transform.forward(x)
            
            # new input x is z
            x = z
            X.append(x)
        
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
        n_samples = 2000
        z = target_distribution.sample([n_samples])
        pz = target_distribution.log_prob(z).exp()
        Z.append(z)
        PZ.append(pz)

        for transform in model.transforms[::-1]:
            x, log_dx_dz = transform.backward(z)
            zcopy, log_dz_dx = transform.forward(x)
            px = (pz.log() + log_dz_dx).exp()
            
            # new input z is x
            z = x
            pz = px
            Z.append(z)
            PZ.append(pz)
        
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

plot_evolution_2D(model, train_loader, target_distribution)
# %%
plot_2D(model, train_loader, target_distribution, False)
plot_2D(model, train_loader, target_distribution, True)
