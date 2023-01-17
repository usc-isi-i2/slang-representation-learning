from eval_data import get_dataset
from BertClassifier import BertClassifier, train, evaluate, evaluate_semeval
from transformers import BertTokenizer, BertModel, AutoTokenizer, RobertaModel
import argparse


def finetune(eval_ds_name, eval_data_path, model_name="bert-base-uncased"):

    df_train, df_val, df_test = get_dataset(eval_ds_name, eval_data_path).read_data()

    if "bert-base" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        pmodel = BertModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pmodel = RobertaModel.from_pretrained(model_name)

    model = BertClassifier(bert_model=pmodel)
    train(model, tokenizer, df_train, df_val)

    if eval_ds_name == "OffenseEval" or eval_ds_name == "HateEval":
        evaluate(model, tokenizer, df_test)

    if eval_ds_name == "SemEval":
        evaluate_semeval(model, tokenizer, df_test)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-ename','--eval_name', required=True, type=str, description="Name of Evaluation Dataset")
    parser.add_argument('-epath','--eval_path', required=True, description="Path to Evaluation Dataset")
    parser.add_argument('--model_name', default="bert-base-uncased", description="Name of BERT based model")

    finetune(args.eval_name, args.eval_path, args.model_name)
