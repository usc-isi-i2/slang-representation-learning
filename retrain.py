import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    RobertaModel,
    RobertaForMaskedLM
)
from utils import set_seed, preprocess


class RetrainDataset(Dataset):
    def __init__(self, df, tokenizer, block_size=512):
        lines = df["clean_text"].tolist()
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_prob=0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def retrain(train_dataset, pretrain_model, pretrain_tokenizer, train_bs=32, max_steps=2000, num_epochs=2,
            lr=5e-5, adam_epsilon=1e-8, model_name_or_path=None, out_dir="./output/",seed=110, save_steps=1000):

    def collate(examples):
        if pretrain_tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=pretrain_tokenizer.pad_token_id)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pretrain_model = pretrain_model.to(device)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_bs, collate_fn=collate)

    if max_steps > 0:
        t_total = max_steps
        num_epochs = max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) // num_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=t_total)

    if (model_name_or_path and os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_name_or_path, "scheduler.pt"))):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))


    global_step = 0
    epochs_trained = 0
    step_cur_epoch = 0

    if model_name_or_path and os.path.exists(model_name_or_path):
        checkpoint_suffix = model_name_or_path.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // len(train_dataloader)
        step_cur_epoch = global_step % len(train_dataloader)

    tr_loss, log_loss = 0.0, 0.0

    pretrain_model.zero_grad()

    set_seed(seed)

    for _ in trange(epochs_trained, num_epochs, desc="Epochs"):
        for step, train_batch in tqdm(train_dataloader):
            if step_cur_epoch > 0:
                step_cur_epoch -=1
                continue

            inputs, labels = mask_tokens(train_batch, pretrain_tokenizer)
            inputs = inputs.to(device)
            labels = labels.to(device)

            pretrain_model.train()

            outputs = pretrain_model(inputs, labels)
            loss = outputs[0]

            loss.mean().backward()

            tr_loss += loss.mean().item()

            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1)

            optimizer.step()
            scheduler.step()
            pretrain_model.zero_grad()
            global_step += 1

            if global_step % save_steps == 0:
                checkpoint_prefix = "checkpoint"
                output_dir = os.path.join(out_dir, "{}-{}".format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.mkdirs(output_dir)
                pretrain_model.save_pretrained(output_dir)
                pretrain_tokenizer.save_pretrained(output_dir)

                print("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                print("Saving optimizer and scheduler states to %s", output_dir)

    return global_step, tr_loss / global_step


def evaluate(eval_dataset, pretrain_model, pretrain_tokenizer, prefix="", out_dir="./eval_output", eval_bs=32):

    def collate(examples):
        if pretrain_tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=pretrain_tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=collate)

    eval_loss = 0.0
    nb_eval_steps = 0

    pretrain_model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")
    model = RobertaForMaskedLM.from_pretrained("vinai/bertweet-large")

    slang_files = ["twitter/slang_2010_tweets.csv", "twitter/slang_2020_tweets.csv"]

    slang_df = pd.concat((pd.read_csv(f) for f in slang_files), ignore_index=True)
    print("Number of entries in slang: ", len(slang_df))

    slang_df = slang_df.drop(columns=["id", "year", "month", "day", "author_id"])
    slang_df = slang_df.loc[:, ~slang_df.columns.str.contains('^Unnamed')]
    slang_df = slang_df.dropna()
    print("Number of entries in slang: ", len(slang_df))

    slang_df["clean_text"] = slang_df["text"].apply(lambda x: preprocess(x))

    df_train, df_test = train_test_split(slang_df, shuffle=True, random_state=42, test_size=0.2)

    train_dataset = RetrainDataset(df_train, tokenizer)
    eval_dataset = RetrainDataset(df_test, tokenizer)

    train(train_dataset, model, tokenizer)
