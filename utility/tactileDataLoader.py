import os, sys
from torch.utils.data import Dataset
import numpy as np

class TactileDataLoader(Dataset):
    """
    Load  Tactile Data
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.files = os.listdir(self.file_path)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        data = np.load(self.file_path + file_name, allow_pickle=True).item()
        return data['LR'], data['HR']

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    pass
