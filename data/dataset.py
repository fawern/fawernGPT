import torch 
from torch.utils.data import Dataset
from typing import List

class LMDataset(Dataset):
    """
    Dataset for language modeling.

    Args:
        ids: List of token ids.
        block_size: Size of the block to use for the dataset.
    """
    def __init__(
        self, 
        ids: List[int],
        block_size: int = 64
    ):
        self.ids = ids
        self.block = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block - 1)
    
    def __getitem__(self, idx):
        x = self.ids[idx:idx+self.block]
        y = self.ids[idx+1:idx+self.block+1]

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            'labels': torch.tensor(y, dtype=torch.long)
        }