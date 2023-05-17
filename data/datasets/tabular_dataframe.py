import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torchvision.datasets.utils import check_integrity

from .utils import quantiletransform

logger = logging.getLogger(__name__)


# Copied from https://github.com/pfnet-research/deep-table.
# Modified by somaonishi and shoyameguro.
class TabularDataFrame(object):
    """Base class for datasets"""

    def __init__(self, config, download: bool = False) -> None:
        """
        Args:
            root (str): Path to the root of datasets for saving/loading.
            download (bool): If True, you must implement `self.download` method
                in the child class. Defaults to False.
        """
        self.config = config
        self.root = config.data_dir
        if download:
            self.download()

    @property
    def mirrors(self) -> None:
        pass

    @property
    def resources(self) -> None:
        pass

    @property
    def raw_folder(self) -> str:
        """The folder where raw data will be stored"""
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self, fpath: Sequence[Tuple[str, Union[str, None]]]) -> bool:
        """
        Args:
            fpath (sequence of tuple[str, (str or None)]): Each value has the format
                [file_path, (md5sum or None)]. Checking if files are correctly
                stored in `self.raw_folder`. If `md5sum` is provided, checking
                the file itself is valid.
        """
        return all(check_integrity(os.path.join(self.raw_folder, path[0]), md5=path[1]) for path in fpath)

    def download(self) -> None:
        """
        Implement this function if the dataset is downloadable.
        See :func:`~deep_table.data.datasets.adult.Adult` for an example implementation.
        """
        raise NotImplementedError

    def cat_cardinalities(self, use_unk: bool = True) -> Optional[List[int]]:
        """List of the numbers of the categories of each column.

        Args:
            use_unk (bool): If True, each feature (column) has "unknown" categories.

        Returns:
            List[int], optional: List of cardinalities. i-th value denotes
                the number of categories which i-th column has.
        """
        cardinalities = []
        df_train = self.get_dataframe(train=True)
        df_train_cat = df_train[self.categorical_columns]
        cardinalities = df_train_cat.nunique().values.astype(int)
        if use_unk:
            cardinalities += 1
        cardinalities_list = cardinalities.tolist()
        return cardinalities_list

    def get_dataframe(self, train: bool = True) -> pd.DataFrame:
        """
        Args:
            train (bool): If True, the returned value is `pd.DataFrame` for train.
                If False, the returned value is `pd.DataFrame` for test.

        Returns:
            `pd.DataFrame`
        """
        if train:
            return self.train
        else:
            return self.test

    def get_classify_dataframe(self, val_size, seed) -> Dict[str, pd.DataFrame]:
        df_train = self.get_dataframe(train=True)
        df_test = self.get_dataframe(train=False)
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_size,
            stratify=df_train[self.target_columns],
            random_state=seed,
        )
        if self.config.train_size < 1:
            logger.info(f"Change train set size {len(df_train)} -> {int(len(df_train) * self.config.train_size)}.")
            df_train, _ = train_test_split(
                df_train,
                train_size=self.config.train_size,
                stratify=df_train[self.target_columns],
                random_state=seed,
            )

        classify_dfs = {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }

        if len(classify_dfs["val"]) > 20000:
            logger.info("validation size reduction: {} -> 20000".format(len(classify_dfs["val"])))
            classify_dfs["val"], _ = train_test_split(
                classify_dfs["val"],
                train_size=20000,
                stratify=classify_dfs["val"][self.target_columns],
                random_state=seed,
            )
        return classify_dfs

    def get_regression_dataframe(self, val_size, seed) -> Dict[str, pd.DataFrame]:
        df_train = self.get_dataframe(train=True)
        df_test = self.get_dataframe(train=False)
        df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=seed)
        if self.config.train_size < 1:
            logger.info(f"Change train set size {len(df_train)} -> {int(len(df_train) * self.config.train_size)}.")
            df_train, _ = train_test_split(
                df_train,
                train_size=self.config.train_size,
                random_state=seed,
            )

        regression_dfs = {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }
        self.y_mean = regression_dfs["train"][self.target_columns].to_numpy().mean()
        self.y_std = regression_dfs["train"][self.target_columns].to_numpy().std()
        for key in regression_dfs.keys():
            regression_dfs[key] = self._regression_encoder(regression_dfs[key])

        if len(regression_dfs["val"]) > 20000:
            logger.info("validation size reduction: {} -> 20000".format(len(regression_dfs["val"])))
            regression_dfs["val"], _ = train_test_split(regression_dfs["val"], train_size=20000, random_state=seed)
        return regression_dfs

    def processed_dataframes(self, val_size, seed) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            dict[str, DataFrame]: The value has the keys "train", "val" and "test".
        """
        if self.task == "regression":
            dfs = self.get_regression_dataframe(val_size=val_size, seed=seed)
        else:
            dfs = self.get_classify_dataframe(val_size=val_size, seed=seed)
        # preprocessing
        if self.categorical_columns != []:
            categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                dfs["train"][self.categorical_columns]
            )
        # only apply DL model.
        if self.continuous_columns != []:
            continuous_encoder = quantiletransform(dfs["train"][self.continuous_columns], seed=seed)

        for key in dfs.keys():
            if self.categorical_columns != []:
                dfs[key][self.categorical_columns] = (
                    categorical_encoder.transform(dfs[key][self.categorical_columns]) + 1
                )
            if self.continuous_columns != []:
                dfs[key][self.continuous_columns] = continuous_encoder.transform(dfs[key][self.continuous_columns])
        return dfs

    def _regression_encoder(self, df):
        df[self.target_columns] = (df[self.target_columns] - self.y_mean) / self.y_std
        return df

    def num_categories(self, use_unk: bool = True) -> int:
        """Total numbers of categories

        Args:
            use_unk (bool): If True, the returned value is calculated
                as there are unknown categories.
        """
        return sum(self.cat_cardinalities(use_unk=use_unk))
