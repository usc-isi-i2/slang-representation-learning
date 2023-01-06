from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils import clean_text

REGISTRY = {}


def register_dataset(cls):
    REGISTRY[cls.__name__] = cls


def get_dataset(name, datapath):
    return REGISTRY[name](datapath)


class EvalDataset(ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_dataset(cls)

    @abstractmethod
    def read_data(self):
        pass


class OffenseEval(EvalDataset):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    def read_data(self):
        path = os.path.join(self.data_path, "olid-training-v1.0.tsv")
        data_df = pd.read_csv(path, delimiter="\t", dtype=str)
        data_df = data_df.drop(["subtask_b", "subtask_c"], axis=1)
        data_df["cleaned_tweet"] = data_df["tweet"].apply(lambda x: clean_text(x))
        le = LabelEncoder()
        data_df["label"] = le.fit_transform(train_df["subtask_a"])

        np.random.seed(112)
        df_train, df_val, df_test = np.split(data_df.sample(frac=1, random_state=42),
                                             [int(.8 * len(data_df)), int(.9 * len(data_df))])

        return df_train, df_val, df_test


class HateEval(EvalDataset):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    def read_data(self):
        path = os.path.join(self.data_path, "Data/#2 Development-English-A/train_dev_en_merged.tsv")
        data_df = pd.read_csv(path, delimiter="\t", dtype=str)
        data_df = data_df.drop(["TR", "AG"], axis=1)
        data_df["cleaned_tweet"] = data_df["text"].apply(lambda x: clean_text(x))
        data_df["label"] = data_df["HS"].apply(lambda x: int(x))

        np.random.seed(112)
        df_train, df_val, df_test = np.split(data_df.sample(frac=1, random_state=42),
                                             [int(.8 * len(data_df)), int(.9 * len(data_df))])

        return df_train, df_val, df_test


class SemEval(EvalDataset):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    def read_data(self):
        path = os.path.join(self.data_path, "train/")
        train_files = glob.glob(os.path.join(path, "*.txt"))

        train_df = pd.concat(
            (pd.read_csv(f, delimiter="\t", header=None, names=["id", "target", "tweet"], dtype=str, index_col=False)
             for f in train_files), ignore_index=True)
        train_df["id"] = train_df["id"].astype("str")
        train_df["target"] = train_df["target"].astype("str")
        train_df["tweet"] = train_df["tweet"].astype("str")

        train_df = train_df.loc[train_df["target"].isin(["positive", "negative", "neutral"])]

        train_df["cleaned_tweet"] = train_df["tweet"].apply(lambda x: clean_text(x))
        le = LabelEncoder()
        train_df["label"] = le.fit_transform(train_df["target"])

        np.random.seed(112)
        df_train, df_val = np.split(train_df.sample(frac=1, random_state=43), [int(0.9 * len(train_df))])

        test_df = pd.read_csv(os.path.join(self.data_path, "/test/SemEval2017-task4-test.subtask-A.english.txt"),
                              sep="\t", header=None, names=["id", "target", "tweet"])
        test_df["cleaned_tweet"] = test_df["tweet"].apply(lambda x: clean_text(x))
        test_df["label"] = le.fit_transform(test_df["target"])

        return df_train, df_val, test_df


if __name__ == '__main__':
    print(REGISTRY.keys())
    get_dataset(name="OffenseEval", datapath=None).read_data()
    get_dataset(name="HateEval", datapath=None).read_data()
