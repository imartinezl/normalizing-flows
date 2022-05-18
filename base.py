# %%

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from tqdm import tqdm

# %%

class NormalizingFlow(nn.Module):
    def __init__(self, transforms):
        super(NormalizingFlow, self).__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, y=None):
        log_dz_dx_sum = torch.zeros_like(x)
        for transform in self.transforms:
            z, log_dz_dx = transform.forward(x, y)
            log_dz_dx_sum += log_dz_dx
            x = z
        return z, log_dz_dx_sum

    def backward(self, z, y=None):
        log_dx_dz_sum = torch.zeros_like(z)
        for transform in self.transforms[::-1]:
            x, log_dx_dz = transform.backward(z, y)
            log_dx_dz_sum += log_dx_dz
            z = x
        return x, log_dx_dz_sum

def log_prob(model, target_distribution, z, log_dz_dx):
    return target_distribution.log_prob(z).view(z.size(0),-1).sum(1) + log_dz_dx.sum(1)

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    device = train_loader.dataset.device

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        z, log_dz_dx = model(x, y)
        loss = -torch.mean(log_prob(model, target_distribution, z, log_dz_dx)) # MEAN OR SUM?
        loss.backward()
        optimizer.step()
    return model

import math

@torch.no_grad()    
def eval_loss(model, data_loader, target_distribution, log_scaler=0.0):
    model.eval()
    device = data_loader.dataset.device
    total_loss = 0
    logprobs = []

    for i, (x, y) in enumerate(data_loader):
        z, log_dz_dx = model(x, y)
        log_likelihood = log_prob(model, target_distribution, z, log_dz_dx)
        
        total_loss += -torch.mean(log_likelihood)# * x.size(0)
        logprobs.append(log_likelihood + log_scaler)
        # logprobs.append(log_likelihood)

    # total_loss = (total_loss / len(data_loader.dataset)).item()
    total_loss = (total_loss / i).item()
    logprobs = torch.cat(logprobs, dim=0).to(device)
    logprob_mean, logprob_std = logprobs.mean(0).item(), logprobs.var(0).sqrt().item() / math.sqrt(len(data_loader.dataset))
    return total_loss, logprob_mean, logprob_std


def train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_scaler = np.sum(-np.log(train_loader.dataset.train_max - train_loader.dataset.train_min))

    train_losses, test_losses = [], []
    # with tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        model = train(model, train_loader, optimizer, target_distribution)
        train_loss, train_logprob_mean, train_logprob_std = eval_loss(model, train_loader, target_distribution, log_scaler)
        test_loss, test_logprob_mean, test_logprob_std = eval_loss(model, test_loader, target_distribution, log_scaler)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_logprob_mean': train_logprob_mean,
                'train_logprob_std': train_logprob_std,
                'test_loss': test_loss,
                'test_logprob_mean': test_logprob_mean,
                'test_logprob_std': test_logprob_std,
            }, os.path.join(path, "epoch_{}.pth".format(epoch)))
        if test_loss <= min(test_losses):
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_logprob_mean': train_logprob_mean,
                'train_logprob_std': train_logprob_std,
                'test_loss': test_loss,
                'test_logprob_mean': test_logprob_mean,
                'test_logprob_std': test_logprob_std,
            }, os.path.join(path, "epoch_best.pth"))
        # # pbar.set_description(f"Loss: train {train_loss:.4f}; test {test_loss:.4f}")
        # pbar.set_description(f"Loglikelihood: train {train_logprob_mean:.4f}; test {test_logprob_mean:.4f}")
        # pbar.update()
    # pbar.close()
    return model, train_losses, test_losses

def plot_loss(train_losses, test_losses, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(train_losses, label='train_loss')
    ax.plot(test_losses, label='test_loss')
    ax.legend()

