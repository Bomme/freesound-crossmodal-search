from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class SingleModalityDataset(Dataset):
    def __init__(
        self,
        items_idx: list[object],
        items: dict[object, object],
        source_identifier_fn: Optional[Callable[[object], object]] = None,
    ):
        self.items_idx = items_idx
        self.items = items
        self.source_identifier_fn = source_identifier_fn or (lambda x: x)

    def get_source_identifier(self, key):
        assert key in self.items, f"The key {key} is not present in the dataset"
        return self.source_identifier_fn(key)

    def __getitem__(self, idx):
        key = self.items_idx[idx]
        comparable = self.get_source_identifier(key)
        return key, self.items[key], comparable

    def __len__(self):
        return len(self.items_idx)


class EmbeddingDataset(SingleModalityDataset):
    def __getitem__(self, idx):
        key, path, comparable = super().__getitem__(idx)
        path = Path(path)
        return key, np.load(path.with_suffix(".npy")), comparable


class PairedModalitiesDataset(Dataset):
    def __init__(
        self, this_dataset: SingleModalityDataset, that_dataset: SingleModalityDataset
    ):
        assert len(this_dataset) == len(
            that_dataset
        ), "Expected that both datasets have the same number of elements"
        self.length = len(this_dataset)
        self.this_dataset = this_dataset
        self.that_dataset = that_dataset

    def __getitem__(self, idx):
        this_key, this, this_comparable = self.this_dataset[idx]
        that_key, that, that_comparable = self.that_dataset[idx]
        return this_key, this, this_comparable, that_key, that, that_comparable

    def __len__(self):
        return self.length


class ThreeSplitsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False)
