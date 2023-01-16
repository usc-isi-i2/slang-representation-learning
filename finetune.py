from eval_data import get_dataset
from BertClassifier import BertClassifier, train, evaluate, evaluate_semeval
from transformers import BertTokenizer, BertModel


def finetune(eval_ds_name, eval_data_path, model_name="bert-base-uncased"):

    df_train, df_val, df_test = get_dataset(eval_ds_name, eval_data_path).read_data()

    tokenizer = BertTokenizer.from_pretrained(model_name)
    pmodel = BertModel.from_pretrained(model_name)

    model = BertClassifier(bert_model=pmodel)
    train(model, tokenizer, df_train, df_val)

    if eval_ds_name == "OffenseEval" or eval_ds_name == "HateEval":
        evaluate(model, tokenizer, df_test)

    if eval_ds_name == "SemEval":
        evaluate_semeval(model, tokenizer, df_test)
    return
