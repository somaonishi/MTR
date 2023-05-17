import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .tabular_dataframe import TabularDataFrame

exceptions_binary = [45062]
exceptions_multiclass = [44135]


def get_task_and_dim_out(data_id, df, columns, cate_indicator, target_col):
    target_idx = columns.index(target_col)

    if data_id in exceptions_binary:
        task = "binary"
        dim_out = 1
    elif data_id in exceptions_multiclass:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    elif cont_checker(df, target_col, cate_indicator[target_idx]):
        task = "regression"
        dim_out = 1
    elif int(df[target_col].nunique()) == 2:
        task = "binary"
        dim_out = 1
    else:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    return task, dim_out


def cont_checker(df, col, is_cate):
    return not is_cate and df[col].dtype != bool and df[col].dtype != object


def cate_checker(df, col, is_cate):
    return is_cate or df[col].dtype == bool or df[col].dtype == object


def get_columns_list(df, columns, cate_indicator, target_col, checker):
    return [col for col, is_cate in zip(columns, cate_indicator) if col != target_col and checker(df, col, is_cate)]


def print_dataset_details(dataset: openml.datasets.OpenMLDataset, data_id):
    df, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")
    print(dataset.name)
    print(dataset.openml_url)
    print(df)

    target_col = dataset.default_target_attribute
    print("Nan count", df.isna().sum().sum())
    print("cont", get_columns_list(df, columns, cate_indicator, target_col, cont_checker))
    print("cate", get_columns_list(df, columns, cate_indicator, target_col, cate_checker))
    print("target", target_col)

    task, dim_out = get_task_and_dim_out(data_id, df, columns, cate_indicator, target_col)
    print(f"task: {task}")
    print(f"dim_out: {dim_out}")
    exit()


class OpenmlDataset(TabularDataFrame):
    def __init__(self, data_id, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        dataset = openml.datasets.get_dataset(data_id)
        if config.show_data_detail:
            print_dataset_details(dataset, data_id)

        df, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")

        self.all_columns = columns
        target_col = dataset.default_target_attribute
        self.continuous_columns = get_columns_list(df, columns, cate_indicator, target_col, cont_checker)
        self.categorical_columns = get_columns_list(df, columns, cate_indicator, target_col, cate_checker)

        self.task, self.dim_out = get_task_and_dim_out(data_id, df, columns, cate_indicator, target_col)

        self.target_columns = [target_col]
        if self.task != "regression":
            df[target_col] = LabelEncoder().fit_transform(df[target_col])
            self.train, self.test = train_test_split(
                df, test_size=0.2, stratify=df[target_col], random_state=self.config.seed
            )
        else:
            self.train, self.test = train_test_split(df, test_size=0.2, random_state=self.config.seed)
