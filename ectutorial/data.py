import requests
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SlidingWindowDataset(Dataset):
    def __init__(self, ts, window_size, step_size):
        self.window_size = window_size
        self.ts = ts
        self.step_size = step_size
    
    def __len__(self):
        return (len(self.ts) - self.window_size + 1) // self.step_size
    
    def __getitem__(self, idx):
        return torch.FloatTensor(
            self.ts[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        )

def get_ett_dataset(with_features=False):
    r = requests.get(
        'https://raw.githubusercontent.com/airi-industrial-ai/ETDataset/main/ETT-small/ETTh1.csv'
    )
    open('ETTh1.csv', 'wb').write(r.content)
    ett = pd.read_csv('ETTh1.csv', index_col=0)
    ett.index = pd.to_datetime(ett.index)
    if not with_features:
        return ett

    ett["dom"] = ett.index.day
    ett["dow"] = ett.index.weekday
    ett["hour"] = ett.index.hour
    ett["mnth"] = ett.index.month
    ett['OT-1'] = ett.OT.shift(1)
    ett = ett.iloc[1:]

    return ett
    
