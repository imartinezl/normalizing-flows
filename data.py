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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return [self.x[index]]
        return self.x[index], self.y[index]

class CustomSampler():
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__()
        data = dataset.tensors
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
        self.dataset = dataset
        self.x = x
        self.y = y
        self.n = len(self.x)
        self.batch_size = batch_size
        self.n_batches = len(x) // batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n)
            self.x = self.x[idx]
            if self.y is not None:
                self.y = self.y[idx]
        b = self.batch_size
        for i in range(self.n_batches):
            if self.y is None:
                yield self.x[(i*b):((i+1)*b)], None
            else:
                yield self.x[(i*b):((i+1)*b)], self.y[(i*b):((i+1)*b)]



class MultipleGPUSampler():
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__()
        data = dataset.tensors
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data

        self.dataset = dataset
        self.x = x
        self.y = y
        self.n = len(self.x)
        # self.batch_size = batch_size
        # self.n_batches = (len(x) // batch_size) + 1

        # shuffle
        if shuffle:
            self.permutation()
        
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:" + str(j)) for j in range(n_devices)]

        # assign to device
        xb = torch.split(self.x, batch_size)
        if y is not None:
            yb = torch.split(self.y, batch_size)
        self.n_batches = len(xb)
        print("Batch split", self.n_batches )

        self.x = []
        self.y = None if self.y is None else []
        for i in range(self.n_batches):
            dev = devices[i % n_devices]
            self.x.append(xb[i].to(dev))
            if y is not None:
                self.y.append(yb[i].to(dev))
        
        print("Device assigned")

    def permutation(self):
        idx = torch.randperm(self.n)
        self.x = self.x[idx]
        if self.y is not None:
            self.y = self.y[idx]

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        for i in range(self.n_batches):
            if self.y is None:
                yield self.x[i], None
            else:
                yield self.x[i], self.y[i]

          


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

        # construct datasets/home/imartinez/.conda/envs/.venv/lib/python3.8/site-packages/torch/_utils.py
        train_dataset = TensorDataset(torch.from_numpy(train_x).to(device))
        test_dataset  = TensorDataset(torch.from_numpy(test_x).to(device))

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
        train_dataset = TensorDataset(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
        test_dataset  = TensorDataset(torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device))

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
        train_dataset = TensorDataset(torch.from_numpy(train_x).to(device))
        test_dataset  = TensorDataset(torch.from_numpy(test_x).to(device))

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
        train_dataset = TensorDataset(torch.from_numpy(train_x).to(device))
        test_dataset  = TensorDataset(torch.from_numpy(test_x).to(device))

        # train_dataset = CustomDataset(torch.from_numpy(train_x))
        # test_dataset  = CustomDataset(torch.from_numpy(test_x))
              
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
    # kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}
    # kwargs = {}
    # train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    train_loader = CustomSampler(train_dataset, batch_size, shuffle=True)
    test_loader = CustomSampler(train_dataset, batch_size, shuffle=False)
    
    # train_loader = MultipleGPUSampler(train_dataset, batch_size, shuffle=True)
    # test_loader = MultipleGPUSampler(train_dataset, batch_size, shuffle=False)

    return train_loader, test_loader

# %%

datasets = ["POWER", "MOONS", "CIRCLES", "SCURVE", "SWISSROLL", "GAUSSIAN", 
    "CRESCENT", "CRESCENTCUBED", "SINEWAVE", "ABS", "SIGN", "TWOSPIRALS", 
    "CHECKERBOARD", "FOURCIRCLES", "DIAMOND", "FACE", "MNIST", "MINIBOONE", 
    "HEPMASS", "GAS", "BSDS300"]
