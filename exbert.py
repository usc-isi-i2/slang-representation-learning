import pandas as pd
from transformers import AutoTokenizer, RobertaModel
import json
import re
from utils import read_osd
from eval_data import get_dataset
from BertClassifier import BertClassifier, train, evaluate, evaluate_semeval


def main(eval_ds_name, eval_data_path, model_name="vinai/bertweet-large")
    word_usage_dict = read_osd()
    osd_words = osd_words.keys()
    osd_words = [x for x in osd_words if " " not in x]
    osd_words = [x for x in osd_words if not re.search(r'\d', x)]
    print(len(osd_words))
    all_words = list(set(osd_words))
    all_words = [x.lower() for x in all_words]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pmodel = RobertaModel.from_pretrained(model_name)

    print("[ BEFORE ] tokenizer vocab size:", len(tokenizer))
    added_tokens = tokenizer.add_tokens(all_words)
    print("[ AFTER ] tokenizer vocab size:", len(tokenizer))
    print()
    model.resize_token_embeddings(len(tokenizer))

    df_train, df_val, df_test = get_dataset(eval_ds_name, eval_data_path).read_data()
    model = BertClassifier(bert_model=pmodel)
    train(model, tokenizer, df_train, df_val)

    if eval_ds_name == "OffenseEval" or eval_ds_name == "HateEval":
        evaluate(model, tokenizer, df_test)

    elif eval_ds_name=="SemEval":
        evaluate_semeval(model, tokenizer, df_test)