import fasttext
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from eval_data import get_dataset

from utils import read_ud, scrape_OSD
import json
import argparse


def get_ft_embed(ft_model, text):
    return ft_model.get_sentence_vector(text)


def ft_evaluate(ft_model, eval_ds_name, eval_ds_path):
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


def train_ft(ds_name="UD", ds_path="./data/ud_2020/all_definitions.dat", out_model_dir="./models"):

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
        df.to_csv('data/osd.dat', index=False)
        fname = "data/osd.dat"

    if ds_name == "UD+OSD":
        if not os.path.exists(fname):
            ud_path = "./data/ud_2020/all_definitions.dat"
            df = pd.read_csv(ud_path, on_bad_lines="skip", delimiter="|", dtype=str)
            df["example"] = df["example"].astype(str)
            df["example"] = df["example"].apply(lambda x: clean_ud(x))
            if not os.path.exists("data/osd.json"):
                word_usage_dict = scrape_OSD()
            else:
                with open("data/osd.json", "r") as f:
                    word_usage_dict = json.load(f)
            examples = [x for ex in word_usage_dict.values() for x in ex]
            osd_df = pd.DataFrame(examples)
            df.append(osd_df)
            df.to_csv("data/ud+osd.dat")
            fname = "data/ud_osd.dat"

    model = fasttext.train_unsupervised(fname, epoch=10, minn=2, dim=300)
    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
    model.save_model(os.path.join(out_model_dir, "ft_trained.bin"))

    return model

def main(eval_ds_name, eval_ds_path,ft_model_path=None, train=True,ds_name="UD", ds_path="./data/ud_2020/all_definitions.dat",out_model_dir="./models" ):
    if train:
       ft_model = train_ft(ds_name, ds_path,out_model_dir=out_model_dir)
    elif ft_model_path is not None:
        ft_model = fasttext.load_model(ft_model_path)
    else:
        print("Model Path not given. Training is set to False.")
        return
    ft_evaluate(ft_model, eval_ds_name, eval_ds_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-ename','--eval_name', required=True, type=str, description="Name of Evaluation Dataset")
    parser.add_argument('-epath','--eval_path', required=True, description="Path to Evaluation Dataset")
    parser.add_argument('--train', type=bool, default=False,description="Train FastText model")
    parser.add_argument('--train_ds_name', description="Name of training dataset")
    parser.add_argument('--train_ds_path', description="Path to training dataset")
    parser.add_argument('--model', description="Path to fast text model", default=None)
    parser.add_argument('--out_dir', description="Path to store fast text model", default="./models")
    args = parser.parse_args()
    
    main(args.eval_name, args.eval_path, args.model, args.train, args.train_ds_name, args.train_ds_path, args.out_dir)