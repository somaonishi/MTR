import logging
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset

import data.datasets as datasets
from data.datasets.tabular_dataframe import TabularDataFrame

logger = logging.getLogger(__name__)

# Copied from https://github.com/pfnet-research/deep-table.
# Modified by somaonishi and shoyameguro.
class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str = "binary",
        continuous_columns: Optional[Sequence[str]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        target: Optional[Union[str, Sequence[str]]] = None,
        transform: Optional[Callable] = None,
        unlabel_data_rate: Optional[float] = None,
        seed=42,
    ) -> None:
        """
        Args:
            data (pandas.DataFrame): DataFrame.
            task (str): One of "binary", "multiclass", "regression".
                Defaults to "binary".
            continuous_cols (sequence of str, optional): Sequence of names of
                continuous features (columns). Defaults to None.
            categorical_cols (sequence of str, optional): Sequence of names of
                categorical features (columns). Defaults to None.
            target (str, optional): If None, `np.zeros` is set as target.
                Defaults to None.
            transform (callable): Method of converting Tensor data.
                Defaults to None.
        """
        super().__init__()
        self.task = task
        self.num = data.shape[0]
        self.categorical_columns = categorical_columns if categorical_columns else []
        self.continuous_columns = continuous_columns if continuous_columns else []

        if unlabel_data_rate is not None:
            if task == "regression":
                stratify = None
            else:
                stratify = data[target]
            data, unlabeled_data = train_test_split(
                data, test_size=unlabel_data_rate, stratify=stratify, random_state=seed
            )
            self.num = data.shape[0]
            self.unlabeled_num = unlabeled_data.shape[0]
            logger.info(f"labeled data size: {self.num}")
            logger.info(f"unlabeled data size: {self.unlabeled_num}")

        if self.continuous_columns:
            self.continuous = data[self.continuous_columns].values
            if unlabel_data_rate is not None:
                self.unlabeled_continuous = unlabeled_data[self.continuous_columns].values

        if self.categorical_columns:
            self.categorical = data[categorical_columns].values
            if unlabel_data_rate is not None:
                self.unlabeled_categorical = unlabeled_data[categorical_columns].values

        if target:
            self.target = data[target].values
            if isinstance(target, str):
                self.target = self.target.reshape(-1, 1)
        else:
            self.target = np.zeros((self.num, 1))

        self.transform = transform
        self.unlabel_data_rate = unlabel_data_rate

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            dict[str, Tensor]:
                The returned dict has the keys {"target", "continuous", "categorical"}
                and its values. If no continuous/categorical features, the returned value is `[]`.
        """
        if self.task == "multiclass":
            x = {
                "target": torch.LongTensor(self.target[idx]),
                "continuous": Tensor(self.continuous[idx]) if self.continuous_columns else [],
                "categorical": LongTensor(self.categorical[idx]) if self.categorical_columns else [],
            }
        elif self.task in {"binary", "regression"}:
            x = {
                "target": torch.Tensor(self.target[idx]),
                "continuous": Tensor(self.continuous[idx]) if self.continuous_columns else [],
                "categorical": LongTensor(self.categorical[idx]) if self.categorical_columns else [],
            }
        else:
            raise ValueError(f"task: {self.task} must be 'multiclass' or 'binary' or 'regression'")

        if self.transform is not None:
            x = self.transform(x)

        if hasattr(self, "unlabeled_num"):
            unlabel_idx = np.random.randint(0, self.unlabeled_num)
            unlabel = {
                "continuous": Tensor(self.unlabeled_continuous[unlabel_idx]) if self.continuous_columns else [],
                "categorical": LongTensor(self.unlabeled_categorical[unlabel_idx]) if self.categorical_columns else [],
            }
            return x, unlabel

        else:
            return x


class TabularDatamodule:
    def __init__(
        self,
        dataset: TabularDataFrame,
        transform: Optional[Callable] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        batch_size: int = 128,
        num_workers: int = 3,
        seed: int = 42,
        val_size: float = 0.1,
    ) -> None:
        # self.dataset = dataset
        self.__num_categories = dataset.num_categories()
        self.categorical_columns = dataset.categorical_columns
        self.continuous_columns = dataset.continuous_columns
        self.cat_cardinalities = dataset.cat_cardinalities(True)
        self.target = dataset.target_columns
        self.d_out = dataset.dim_out

        dataframes = dataset.processed_dataframes(val_size=val_size, seed=seed)
        self.train = dataframes["train"]
        self.val = dataframes["val"]
        self.test = dataframes["test"]

        for k, v in dataframes.items():
            logger.info(f"{k} dataset shape: {v.shape}")

        self.task = dataset.task
        self.transform = transform
        self.train_sampler = train_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.task == "regression":
            self.y_std = dataset.y_std
        
        self.seed = seed

    @property
    def num_categories(self) -> int:
        return self.__num_categories

    @property
    def num_continuous_features(self) -> int:
        return len(self.continuous_columns)

    @property
    def num_categorical_features(self) -> int:
        return len(self.categorical_columns)

    def dataloader(self, mode: str, batch_size: Optional[int] = None, transform=None, unlabel_data_rate=None):
        assert mode in {"train", "val", "test"}
        if not hasattr(self, mode):
            return None
        data = getattr(self, mode)

        if mode == "test":
            transform = None

        if transform is None:
            transform = self.transform

        dataset = TabularDataset(
            data=data,
            task=self.task,
            categorical_columns=self.categorical_columns,
            continuous_columns=self.continuous_columns,
            target=self.target,
            transform=transform,
            unlabel_data_rate=unlabel_data_rate,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_size if batch_size is not None else self.batch_size,
            shuffle=True if mode == "train" else False,
            num_workers=self.num_workers,
            sampler=self.train_sampler if mode == "train" else None,
        )


def get_datamodule(config) -> TabularDatamodule:
    if type(config.data) == int:
        dataset = datasets.OpenmlDataset(data_id=config.data, config=config)
    else:
        dataset = getattr(datasets, config.data)(config=config)

    return TabularDatamodule(dataset, batch_size=config.batch_size, seed=config.seed, val_size=config.val_size)
