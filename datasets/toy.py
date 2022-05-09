# %%

import numpy as np
import torch
from torch.utils.data import Dataset

# %%

from sklearn.datasets import make_moons

class MOONS(Dataset):
    def __init__(self, n_samples=25000, noise=0.1, **kwargs):
        self.x, self.y = make_moons(n_samples=n_samples, noise=noise, shuffle=True)
        self.input_size = 2
        self.label_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# %%

from sklearn.datasets import make_circles

class CIRCLES(Dataset):
    def __init__(self, n_samples=25000, noise=0.1, **kwargs):
        self.x, self.y = make_circles(n_samples=n_samples, noise=noise, shuffle=True)
        self.input_size = 2
        self.label_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# %%

from sklearn.datasets import make_s_curve

class SCURVE(Dataset):
    def __init__(self, n_samples=25000, noise=0.1, **kwargs):
        self.x, self.y = make_s_curve(n_samples=n_samples, noise=noise)
        self.input_size = 3
        self.label_size = 3
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# %%
from sklearn.datasets import make_swiss_roll

class SWISSROLL(Dataset):
    def __init__(self, n_samples=25000, noise=0.1, **kwargs):
        self.x, self.y = make_swiss_roll(n_samples=n_samples, noise=noise)
        self.input_size = 3
        self.label_size = 3
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i], self.y[i]
