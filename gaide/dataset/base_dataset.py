#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def train_val_split(self, train_ratio: float = 0.9):
        n = len(self)
        shuffled = torch.randperm(n).tolist()
        split = int(train_ratio * n)
        return shuffled[:split], shuffled[split:]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
