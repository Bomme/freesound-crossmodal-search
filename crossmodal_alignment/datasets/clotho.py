"""Dataset classes and loaders for the the Clotho captioning dataset."""

import csv
import os
from pathlib import Path

from typing import Optional, Union

from crossmodal_alignment.datasets import (
    PairedModalitiesDataset,
    ThreeSplitsDataModule,
    SingleModalityDataset,
    EmbeddingDataset,
)


class ClothoEmbeddings(PairedModalitiesDataset):
    @classmethod
    def build(cls, captions_csv_path, data_dir, split: str, num_captions: int = 5):
        data_dir = Path(data_dir)
        file_names = {}
        file_names_idx = []
        captions = {}
        with open(captions_csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                file_name = row["file_name"]
                file_names[file_name] = data_dir / split / file_name
                for i in range(1, num_captions + 1):
                    file_names_idx.append(file_name)
                    key = f"{file_name}_{i}"
                    captions[key] = row[f"caption_{i}"]

        captions_ds = SingleModalityDataset(
            list(captions.keys()), captions, lambda some_key: some_key.rsplit("_", 1)[0]
        )
        embeddings_ds = EmbeddingDataset(file_names_idx, file_names)
        return cls(embeddings_ds, captions_ds)


class ClothoEmbeddingsDatamodule(ThreeSplitsDataModule):
    def __init__(
        self,
        metadata_dir: Union[str, os.PathLike],
        data_dir: Union[str, os.PathLike],
        **kwargs,
    ):
        self.metadata_dir = Path(metadata_dir)
        self.data_dir = Path(data_dir)
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ClothoEmbeddings.build(
            self.metadata_dir / "clotho_captions_development.csv",
            self.data_dir,
            "development",
        )
        self.val_dataset = ClothoEmbeddings.build(
            self.metadata_dir / "clotho_captions_validation.csv",
            self.data_dir,
            "validation",
        )
        self.test_dataset = ClothoEmbeddings.build(
            self.metadata_dir / "clotho_captions_evaluation.csv",
            self.data_dir,
            "evaluation",
        )
