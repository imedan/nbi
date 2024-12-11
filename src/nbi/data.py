import copy
import os

import numpy as np
from torch.utils.data import Dataset
import h5py


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = len(x)  # number of datapoints
        self.y = self.y.astype("float32")


class BaseContainer(Dataset):
    def __init__(self, x, y, x_file, f_val=0.2, f_test=0, split="all", process=None):
        # create data partitions
        N = len(x)
        p_train = 1 - f_val - f_test
        p_val = 1 - f_test

        self.trn = Data(x[: int(N * p_train)], y[: int(N * p_train)])
        self.val = Data(
            x[int(N * p_train) : int(N * p_val)], y[int(N * p_train) : int(N * p_val)]
        )
        self.tst = Data(x[int(N * p_val) :], y[int(N * p_val) :])
        self.all = Data(x, y)
        self.set_split(split)
        self.process = process

        self.x_file = x_file
        self.dataset = None

    def set_split(self, split="all"):
        data = getattr(self, split)
        self.split = split
        self.x = data.x
        self.y = data.y

    def get_splits(self):
        train = copy.copy(self)
        val = copy.copy(self)
        test = copy.copy(self)
        train.set_split("trn")
        val.set_split("val")
        test.set_split("tst")
        return train, val, test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i, **kwargs):
        if os.path.isfile(self.x_file):
            if self.dataset is None:
                self.dataset = h5py.File(self.x_file, "r")['x']
            idx = int(self.x[i].split('/')[-1].split('.')[0])  # need this to get correct index in HDF5 file?
            x = self.dataset[idx]
            y = self.y[i]
        elif isinstance(self.x[i], np.str_):
            x, y = np.load(self.x[i], allow_pickle=True), self.y[i]
        else:
            x, y = self.x[i], self.y[i]
        if self.process is not None:
            x, y = self.process(x, y)

        return np.atleast_2d(x), y
