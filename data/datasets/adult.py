import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .tabular_dataframe import TabularDataFrame


class Adult(TabularDataFrame):
    dim_out = 1

    all_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    continuous_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    target_columns = ["income"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        df = pd.read_csv(os.path.join(self.raw_folder, "adult.csv"))
        df.columns = self.all_columns
        df["income"] = df["income"].replace({"<=50K": 1, ">50K": 0})
        self.train, self.test = train_test_split(
            df, test_size=0.2, stratify=df["income"], random_state=self.config.seed
        )
