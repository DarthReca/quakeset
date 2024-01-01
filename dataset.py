from typing import Any, Dict, Optional, Set

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset


class EarthQuakeDataModule(pl.LightningDataModule):
    def __init__(self, **hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.batch_size = self.hparams["batch_size"]
        self.file = self.hparams["file"]
        self.magnitudes_file = self.hparams["magnitudes_file"]
        self.num_workers = self.hparams["num_workers"]
        self.fill_nan = self.hparams["fill_nan"]
        self.post_only = self.hparams.get("post_only", False)

        df = pd.read_parquet(self.hparams["split_file"])
        df["resource_id"] = df["resource_id"].str.split("=").str[1]
        self.sample_split = df.set_index("resource_id")

    def setup(self, stage: str = None) -> None:
        get_ids = lambda x: set(
            self.sample_split[self.sample_split["split"] == x].index
        )

        if stage in ("fit", None):
            self.train_dataset = EarthQuakeDataset(
                self.file,
                self.magnitudes_file,
                self.fill_nan,
                ids_set=get_ids("train"),
                post_only=self.post_only,
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = EarthQuakeDataset(
                self.file,
                self.magnitudes_file,
                self.fill_nan,
                ids_set=get_ids("validation"),
                post_only=self.post_only,
            )
        if stage in ("test", None):
            self.test_dataset = EarthQuakeDataset(
                self.file,
                self.magnitudes_file,
                self.fill_nan,
                ids_set=get_ids("test"),
                post_only=self.post_only,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class EarthQuakeDataset(Dataset):
    def __init__(
        self,
        h5_file: str,
        magnitudes_file: str,
        fill_nan: float,
        ids_set: Optional[Set[str]] = None,
        post_only: bool = False,
    ) -> None:
        super().__init__()
        df = pd.read_parquet(magnitudes_file)
        if df.index.name != "resource_id":
            df = df.set_index("resource_id")
        df = df.set_index(df.index.str.split("=").str[1])
        self.magnitudes = df["magnitude"].to_dict()

        self.h5_file = h5_file
        self.sample_ids = []
        with h5py.File(self.h5_file, "r") as f:
            for key, patches in f.items():
                if ids_set is not None and key not in ids_set:
                    continue
                self.sample_ids += [
                    (f"{key}/{p}", 1)
                    for p in patches.keys()
                ]
                self.sample_ids += [
                    (f"{key}/{p}", 0)
                    for p, v in patches.items()
                    if "before" in v
                ]
        self.fill_nan = fill_nan
        self.post_only = post_only

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id, label = self.sample_ids[idx]

        with h5py.File(self.h5_file, "r") as f:
            pre_key = "pre" if label == 1 else "before"
            post_key = "post" if label == 1 else "pre"
            pre_sample = f[sample_id][pre_key][...]
            post_sample = f[sample_id][post_key][...]
        pre_sample = np.nan_to_num(pre_sample, nan=self.fill_nan).transpose(2, 0, 1)
        post_sample = np.nan_to_num(post_sample, nan=self.fill_nan).transpose(2, 0, 1)
        if self.post_only:
            sample = post_sample
        else:
            sample = np.concatenate([pre_sample, post_sample], axis=0, dtype=np.float32)
        resource_id = sample_id.split("/")[0]
        magnitude = self.magnitudes[str(resource_id)] if label == 1 else 0.0

        return {
            "sample": sample,
            "label": label,
            "magnitude": np.float32(magnitude),
        }
