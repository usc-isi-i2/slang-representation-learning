import fasttext
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from eval_data import get_dataset

from utils import read_ud, scrape_OSD
import json


def get_ft_embed(ft_model, text):
    return ft_model.get_sentence_vector(text)


def ft_evaluate(ft_model_path, eval_ds_name, eval_ds_path):
    ft_model = fasttext.load_model(ft_model_path)
    df_train, df_val, df_test = get_dataset(eval_ds_name, eval_ds_path)

    X_train = df_train["cleaned_tweet"].apply(lambda x: get_ft_embed(ft_model, x)).to_list()
    X_train = np.asarray(X_train)
    Y_train = df_train["label"].to_numpy()

    X_val = df_val["cleaned_tweet"].apply(lambda x: get_ft_embed(ft_model, x)).to_list()
    X_val = np.asarray(X_val)
    Y_val = df_val["label"].to_numpy()

    X_test = df_test["cleaned_tweet"].apply(lambda x: get_ft_embed(ft_model, x)).to_list()
    X_test = np.asarray(X_test)
    Y_test = df_test["label"].to_numpy()

    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)

    # Val Stats
    preds = clf.predict(X_val)
    print(classification_report(Y_val, preds))

    # Test Stats
    preds = clf.predict(X_test)
    print(classification_report(Y_test, preds))


def train_ft(ds_name="UD", ds_path="./data/ud_2020/all_definitions.dat"):

    if ds_name == "UD":
        fname = read_ud(ds_path, fname="ud.dat")

    if ds_name == "OSD":
        if not os.path.exists("data/osd.json"):
            word_usage_dict = scrape_OSD()
        else:
            with open("data/osd.json", "r") as f:
                word_usage_dict = json.load(f)
        examples = [x for ex in word_usage_dict.values() for x in ex]
        df = pd.DataFrame(examples)
        df.to_csv('osd.dat', index=False)
        fname = "osd.dat"

    model = fasttext.train_unsupervised(fname, epoch=10, minn=2, dim=300)

    model.save_model("ft_trained.bin")

    return model
