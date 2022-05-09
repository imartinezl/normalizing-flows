# %%

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# %%

class NormalizingFlow(nn.Module):
    def __init__(self, transforms):
        super(NormalizingFlow, self).__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, y=None):
        z, log_dz_dx_sum = x, torch.zeros_like(x)
        for transform in self.transforms:
            z, log_dz_dx = transform(z, y)
            log_dz_dx_sum += log_dz_dx
        return z, log_dz_dx_sum

    def backward(self, z, y=None):
        log_dx_dz_sum = torch.zeros_like(z)
        for transform in self.transforms[::-1]:
            z, log_dx_dz = transform.backward(z, y)
        return z, log_dx_dz_sum

def loss_function(model, target_distribution, z, log_dz_dx):
    log_likelihood = target_distribution.log_prob(z) + log_dz_dx
    loss = -log_likelihood.mean()
    return loss

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    device = train_loader.dataset.device
    for i, data in enumerate(train_loader):
        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(device)
        x = x.view(x.shape[0], -1).to(device)

        z, log_dz_dx = model(x, y)
        loss = loss_function(model, target_distribution, z, log_dz_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
        
def eval_loss(model, data_loader, target_distribution):
    model.eval()
    device = data_loader.dataset.device
    total_loss = 0
    for i, data in enumerate(data_loader):
        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(device)
        x = x.view(x.shape[0], -1).to(device)

        z, log_dz_dx = model(x, y)
        loss = loss_function(model, target_distribution, z, log_dz_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(model, epochs, lr, train_loader, test_loader, target_distribution):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            model = train(model, train_loader, optimizer, target_distribution)
            train_loss = eval_loss(model, train_loader, target_distribution)
            test_loss = eval_loss(model, test_loader, target_distribution)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            pbar.set_description(f"Loss: train {train_loss:.4f}; test {test_loss:.4f}")
            pbar.update()
        pbar.close()
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    return model, train_losses, test_losses

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()

