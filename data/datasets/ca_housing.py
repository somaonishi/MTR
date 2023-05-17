from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .tabular_dataframe import TabularDataFrame


class CAHousing(TabularDataFrame):
    dim_out = 1

    all_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedHouseVal",
    ]

    continuous_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    categorical_columns = []

    target_columns = ["MedHouseVal"]

    task = "regression"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        df = fetch_california_housing(as_frame=True).frame
        self.train, self.test = train_test_split(df, test_size=0.2, random_state=self.config.seed)

    def download(self) -> None:
        pass

    # def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
    #     if train:
    #         return self.train
    #     else:
    #         return self.test

    # def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
    #     df_train = self.raw_dataframe(train=True)
    #     df_test = self.raw_dataframe(train=False)
    #     df_train, df_val = train_test_split(df_train, **kwargs)
    #     dfs = {
    #         "train": df_train,
    #         "val": df_val,
    #         "test": df_test,
    #     }
    #     # preprocessing
    #     # only apply DL model.
    #     sc = quantiletransform(df_train[self.continuous_columns], seed=self.config.seed)

    #     self.y_mean = df_train[self.target_columns].to_numpy().mean()
    #     self.y_std = df_train[self.target_columns].to_numpy().std()
    #     for key in dfs.keys():
    #         dfs[key][self.continuous_columns] = sc.transform(dfs[key][self.continuous_columns])
    #         dfs[key] = self._label_encoder(dfs[key])
    #     return dfs

    # def _label_encoder(self, df):
    #     df[self.target_columns] = (df[self.target_columns] - self.y_mean) / self.y_std
    #     return df
