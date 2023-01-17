# Slang Representation Learning

  

In this work, we focus on understanding slang representations and knowledge sources that rely on slang un-

derstanding.

  

We evaluate slang representations from fasttext and bert models on three tasks:

1. Sentiment Analysis (SemEval)

2. Offense Detection (OffenseEval)

3. Hate Speech Detection (HateEval)

We use two knowledge sources:
1. Urban Dictionary : Please contact us for the data  
2. Online Slang Dictionary: Data is scraped from the web source. 

To run evaluation fasttext model:
```
python fasttext.py --model /path/to/ft_model --eval_name SemEval --eval_path /path/to/data_dir
```
To train and evaluate the trained fasttext model:
```
python fasttext.py --train --train_ds_name UD --train_ds_path /path/to/ud --eval_name SemEval --eval_path /path/to/data_dir --out_dir /path/to_save/trained_model
```
train_ds_name can be either UD or OSD or UD+OSD.

To finetune BERT based models :
```
python finetune.py  eval_name SemEval --eval_path /path/to/data_dir --model_name vinai/bertweet-large
```
We use model_name `bert-base-uncased` and `vinai/bertweet-large` for our experiments.


To retrain the models:
```
python retrain.py
```

Please download the datasets from respective challenges. 
