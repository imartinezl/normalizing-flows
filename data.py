# %%

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# %% TO REMOVE

class Dataset(torch.utils.data.Dataset):
    def __init__(self, array):
        super().__init__()
        if isinstance(array, np.ndarray):
            self.array = array.astype(np.float32)
        
        self.array = self.normalize_01(self.array)
        self.array = torch.from_numpy(self.array)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

    def normalize_01(self, array):
        self.array_min = array.min(0)
        self.array_min = array.max(0)
        array -= array.min(0)
        array /= array.max(0)
        return array

    def normalize_z(self, array):
        self.array_mean = array.mean(axis=0)
        self.array_std = array.std(axis=0)
        array -= array.mean()
        array /= array.std()
        return array

# %%

def normalize_01(x, a, b):
    return (x-a)/(b-a)

def denormalize_01(y, a, b):
    return a*(1-y) + b*y

# %%


def load_dataset(name, source=None):
    if source is None:
        source = name.lower()
    exec('from datasets.{} import {}'.format(source, name))
    return locals()[name]

def fetch_dataloaders(dataset_name, batch_size, device, toy_train_size=25000, toy_test_size=5000):
    # grab datasets
    if dataset_name in ["GAS", "POWER", "HEPMASS", "MINIBOONE", "BSDS300"]:  
        dataset = load_dataset(dataset_name)()

        # join train and val data again
        train_x = np.concatenate((dataset.trn.x, dataset.val.x), axis=0).astype(np.float32)
        test_x = dataset.tst.x.astype(np.float32)

        train_x = train_x[:toy_train_size]
        test_x = test_x[:toy_test_size]

        # normalize
        train_min = train_x.min(0)
        train_max = train_x.max(0)
        train_x = normalize_01(train_x, train_min, train_max)
        test_x = normalize_01(test_x, train_min, train_max)

        # construct datasets
        train_dataset = TensorDataset(torch.from_numpy(train_x))
        test_dataset  = TensorDataset(torch.from_numpy(test_x))

        input_dims = dataset.n_dims
        label_size = None
        lam = None

    elif dataset_name in ['MNIST']:
        dataset = load_dataset(dataset_name)()

        # join train and val data again
        train_x = np.concatenate((dataset.trn.x, dataset.val.x), axis=0).astype(np.float32)
        train_y = np.concatenate((dataset.trn.y, dataset.val.y), axis=0).astype(np.float32)

        test_x = dataset.tst.x.astype(np.float32)
        test_y = dataset.tst.y.astype(np.float32)

        # normalize
        train_min = train_x.min(0)
        train_max = train_x.max(0)
        train_x = normalize_01(train_x, train_min, train_max)
        test_x = normalize_01(test_x, train_min, train_max)

        # construct datasets
        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        test_dataset  = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        input_dims = dataset.n_dims
        label_size = 10
        lam = dataset.alpha

    elif dataset_name in ['MOONS', 'CIRCLES', 'SCURVE', 'SWISSROLL']:  # use own constructors
        train_toy_dataset = load_dataset(dataset_name, "toy")(toy_train_size)
        test_toy_dataset = load_dataset(dataset_name, "toy")(toy_test_size)

        train_x = train_toy_dataset.x.astype(np.float32)
        test_x = test_toy_dataset.x.astype(np.float32)

        # normalize
        train_min = train_x.min(0)
        train_max = train_x.max(0)
        train_x = normalize_01(train_x, train_min, train_max)
        test_x = normalize_01(test_x, train_min, train_max)

        # construct datasets
        train_dataset = TensorDataset(torch.from_numpy(train_x))
        test_dataset  = TensorDataset(torch.from_numpy(test_x))

        input_dims = train_toy_dataset.input_size
        label_size = train_toy_dataset.label_size
        lam = None

    elif dataset_name in ['GAUSSIAN', 'CRESCENT', 'CRESCENTCUBED', 'SINEWAVE', 'ABS', \
        'SIGN', 'TWOSPIRALS', 'CHECKERBOARD', 'FOURCIRCLES', 'DIAMOND', 'FACE']:  # use own constructors
        train_toy_dataset = load_dataset(dataset_name, "plane")(toy_train_size)
        test_toy_dataset = load_dataset(dataset_name, "plane")(toy_test_size)

        train_x = np.array(train_toy_dataset.x).astype(np.float32)
        test_x = np.array(test_toy_dataset.x).astype(np.float32)

        # normalize
        train_min = train_x.min(0)
        train_max = train_x.max(0)
        train_x = normalize_01(train_x, train_min, train_max)
        test_x = normalize_01(test_x, train_min, train_max)

        # construct datasets
        train_dataset = TensorDataset(torch.from_numpy(train_x))
        test_dataset  = TensorDataset(torch.from_numpy(test_x))

        input_dims = 2
        label_size = None
        lam = None

    else:
        raise ValueError('Unrecognized dataset.')

    # keep input dims, input size and label size
    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam
    train_dataset.train_min = train_min
    train_dataset.train_max = train_max
    train_dataset.device = device

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam
    test_dataset.train_min = train_min
    test_dataset.train_max = train_max
    test_dataset.device = device

    # construct dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader

# %%

datasets = ["POWER", "MOONS", "CIRCLES", "SCURVE", "SWISSROLL", "GAUSSIAN", 
    "CRESCENT", "CRESCENTCUBED", "SINEWAVE", "ABS", "SIGN", "TWOSPIRALS", 
    "CHECKERBOARD", "FOURCIRCLES", "DIAMOND", "FACE", "MNIST", "MINIBOONE", 
    "HEPMASS", "GAS", "BSDS300"]
