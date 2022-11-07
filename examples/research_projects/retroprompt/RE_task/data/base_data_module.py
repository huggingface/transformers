"""Base DataModule class."""
import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


BATCH_SIZE = 8
NUM_WORKERS = 8


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = Config(vars(args)) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--data_dir", type=str, default="./dataset/dialogue", help="Number of additional processes to load data."
        )
        return parser

    def get_data_config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"num_labels": self.num_labels}

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def train_dataloader(self):
        return DataLoader(
            self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
