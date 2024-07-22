from torch.utils.data import Dataset
from torch import tensor, float, long


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return tensor(self.x.iloc[idx, :].values, dtype=float), self.y.iloc[idx]
