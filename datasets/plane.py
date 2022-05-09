# %%

import numpy as np
import torch
from torch.utils.data import Dataset

# %%
def make_gaussian(n_samples):
    x1 = torch.randn(n_samples)
    x2 = 0.5 * torch.randn(n_samples)
    return torch.stack((x1, x2)).t()

class GAUSSIAN(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_gaussian(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def make_crescent(n_samples):
    x1 = torch.randn(n_samples)
    x2_mean = 0.5 * x1 ** 2 - 1
    x2_var = torch.exp(torch.Tensor([-2]))
    x2 = x2_mean + x2_var ** 0.5 * torch.randn(n_samples)
    return torch.stack((x2, x1)).t()

class CRESCENT(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_crescent(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def make_crescent_cubed(n_samples):
    x1 = torch.randn(n_samples)
    x2_mean = 0.2 * x1 ** 3
    x2_var = torch.exp(torch.Tensor([-2]))
    x2 = x2_mean + x2_var ** 0.5 * torch.randn(n_samples)
    return torch.stack((x2, x1)).t()

class CRESCENTCUBED(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_crescent_cubed(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def make_sine_wave(n_samples):
    x1 = torch.randn(n_samples)
    x2_mean = torch.sin(5 * x1)
    x2_var = torch.exp(-2 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var ** 0.5 * torch.randn(n_samples)
    return torch.stack((x1, x2)).t()

class SINEWAVE(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_sine_wave(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]


# %%
def make_abs(n_samples):
    x1 = torch.randn(n_samples)
    x2_mean = torch.abs(x1) - 1.
    x2_var = torch.exp(-3 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var ** 0.5 * torch.randn(n_samples)
    return torch.stack((x1, x2)).t()

class ABS(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_abs(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def make_sign(n_samples):
    x1 = torch.randn(n_samples)
    x2_mean = torch.sign(x1) + x1
    x2_var = torch.exp(-3 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var ** 0.5 * torch.randn(n_samples)
    return torch.stack((x1, x2)).t()

class SIGN(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_sign(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def make_two_spirals(n_samples, noise):
    n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * np.pi) / 360
    d1x = -torch.cos(n) * n + torch.rand(n_samples // 2) * noise
    d1y = torch.sin(n) * n + torch.rand(n_samples // 2) * noise
    x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
    return x / 3 + torch.randn_like(x) * 0.1

class TWOSPIRALS(Dataset):
    def __init__(self, n_samples=25000, noise=0.1, **kwargs):
        self.x = make_two_spirals(n_samples=n_samples, noise=noise)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def make_checkerboard(n_samples):
    x1 = torch.rand(n_samples) * 4 - 2
    x2_ = torch.rand(n_samples) - torch.randint(0, 2, [n_samples]).float() * 2
    x2 = x2_ + torch.floor(x1) % 2
    return torch.stack([x1, x2]).t() * 2

class CHECKERBOARD(Dataset):
    def __init__(self, n_samples=25000, **kwargs):
        self.x = make_checkerboard(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def create_circle(num_per_circle, std=0.05):
    u = torch.rand(num_per_circle)
    x1 = torch.cos(2 * np.pi * u)
    x2 = torch.sin(2 * np.pi * u)
    data = 2 * torch.stack((x1, x2)).t()
    data += std * torch.randn(data.shape)
    return data

class FOURCIRCLES(Dataset):
    def __init__(self, n_samples=25000, std=0.05, **kwargs):
        num_per_circle = n_samples // 4
        centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        self.x = torch.cat([create_circle(num_per_circle, std) - torch.Tensor(center) for center in centers])
        self.input_size = 2
        self.n_samples = num_per_circle * 4

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]

# %%
def create_diamond(n_samples, bound, width, std, rotate=True):
    means = np.array([
        (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
        for x in np.linspace(-bound, bound, width)
        for y in np.linspace(-bound, bound, width)
    ])

    covariance_factor = std * np.eye(2)

    index = np.random.choice(range(width ** 2), size=n_samples, replace=True)
    noise = np.random.randn(n_samples, 2)
    data = means[index] + noise @ covariance_factor
    if rotate:
        rotation_matrix = np.array([
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)]
        ])
        data = data @ rotation_matrix
    return data

class DIAMOND(Dataset):
    def __init__(self, n_samples=25000, width=20, bound=2.5, std=0.04, rotate=True, **kwargs):
        self.x = create_diamond(n_samples, bound, width, std, rotate)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]


# %%
from skimage import color, io, transform
def make_face(path, n_samples):
    image = io.imread(path)
    image = color.rgb2gray(image)
    image = transform.resize(image, [200, 200])

    grid = np.array([(x, y) for x in range(image.shape[0]) for y in range(image.shape[1])])
    rotation_matrix = np.array([[0, -1],[1, 0]])
    p = image.reshape(-1) / sum(image.reshape(-1))
    ix = np.random.choice(range(len(grid)), size=n_samples, replace=True, p=p)
    points = grid[ix].astype(np.float32)
    points += np.random.rand(n_samples, 2)  # dequantize
    points /= (image.shape[0])  # scale to [0, 1]

    data = points @ rotation_matrix
    data[:, 1] += 1
    return data

class FACE(Dataset):
    def __init__(self, path, n_samples=25000, **kwargs):
        self.x = make_face(n_samples=n_samples)
        self.input_size = 2
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.x[i]
