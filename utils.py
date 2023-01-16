import html
import nltk
import re
import string
import random
import torch
import os
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def clean_text(text: str) -> str:
    text = text.rstrip()
    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)

    text = ' '.join(text.split())

    text = re.sub("@[A-Za-z0-9]+", "", text)  # Remove @ sign

    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)  # Remove http links

    text = " ".join(text.split())

    text = re.sub(emoji_pattern, '', text)

    text = text.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text

    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.isalpha())

    return text


def clean_ud(txt):
    txt = txt.lower()
    return txt.translate(str.maketrans('', '', string.punctuation))


def read_ud(ud_path, fname):
    if not os.path.exists(fname):
        df = pd.read_csv(ud_path, on_bad_lines="skip", delimiter="|", dtype=str)
        df["example"] = df["example"].astype(str)
        df["example"] = df["example"].apply(lambda x: clean_ud(x))
        df.to_csv(fname, columns=["example"], header=False, index=False)

    return fname

def read_osd(osd_fname = "data/osd.json", write_to_file=False, dfname="./data/osd.dat"):
    if not os.path.exists(osd_fname):
        word_usage_dict = scrape_OSD()
    else:
        with open(osd_fname, "r") as f:
            word_usage_dict = json.load(f)

    if write_to_file:
        examples = [x for ex in word_usage_dict.values() for x in ex]
        df = pd.DataFrame(examples)
        df.to_csv(df_name, index=False)

    return word_usage_dict


def scrape_OSD(osd_fname="data/osd.json"):
    word_list_page = "http://onlineslangdictionary.com/word-list/0-z/"

    resp = requests.get(word_list_page)

    if resp.status_code == 200:
        data = resp.text
    else:
        print("Unable to reach page, Status Code: ", resp.status_code)
        return

    soup = BeautifulSoup(data, "html.parser")
    words = []
    for a in soup.findAll("a"):
        if "meaning-definition-of" in a["href"]:
            words.append(a.text)

    urls = [(word, "http://onlineslangdictionary.com/meaning-definition-of/" + word.replace(" ", "%20")) for word in
            words]
    word_usage_dict = {}
    for wrd, url in tqdm(urls):
        word_page = requests.get(url)
        if word_page.status_code == 200:
            word_data = word_page.content
            soup = BeautifulSoup(word_data, "html.parser")
            bq = soup.find_all("blockquote", {"class": "sentence"})
            if bq is not None and len(bq) > 0:
                word_usage_dict[wrd] = [b.get_text().replace("[", "").replace("]", "") for b in bq]

    with open(osd_fname, "w") as f:
        json.dump(word_usage_dict, f)

    return word_usage_dict


def remove_redundant_punct(text, redundant_punct_pattern):
    text_ = text
    result = re.search(redundant_punct_pattern, text)
    dif = 0
    while result:
        sub = result.group()
        sub = sorted(set(sub), key=sub.index)
        sub = ' ' + ''.join(list(sub)) + ' '
        text = ''.join((text[:result.span()[0] + dif], sub, text[result.span()[1] + dif:]))
        text_ = ''.join((text_[:result.span()[0]], text_[result.span()[1]:])).strip()
        dif = abs(len(text) - len(text_))
        result = re.search(redundant_punct_pattern, text_)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text):
    regex_url_step1 = r'(?=http)[^\s]+'
    regex_url_step2 = r'(?=www)[^\s]+'
    regex_url = r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    regex_mention = r'@[\w\d]+'
    regex_email = r'\S+@\S+'
    redundant_punct_pattern = r'([!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘]{2,})'

    text = str(text)
    processing_tweet = re.sub('ـ', '', text)
    processing_tweet = processing_tweet.lower()
    processing_tweet = re.sub('[«»]', ' " ', processing_tweet)
    processing_tweet = re.sub(regex_url_step1, '[link]', processing_tweet)
    processing_tweet = re.sub(regex_url_step2, '[link]', processing_tweet)
    processing_tweet = re.sub(regex_url, '[link]', processing_tweet)
    processing_tweet = re.sub(regex_email, '[email]', processing_tweet)
    processing_tweet = re.sub(regex_mention, '[user]', processing_tweet)
    processing_tweet = re.sub('…', r'\.', processing_tweet).strip()
    processing_tweet = remove_redundant_punct(processing_tweet, redundant_punct_pattern)
    processing_tweet = re.sub(r'\[ link \]|\[ link\]|\[link \]', ' [link] ', processing_tweet)
    processing_tweet = re.sub(r'\[ email \]|\[ email\]|\[email \]', ' [email] ', processing_tweet)
    processing_tweet = re.sub(r'\[ user \]|\[ user\]|\[user \]', ' [user] ', processing_tweet)
    processing_tweet = re.sub("(.)\\1{2,}", "\\1", processing_tweet)
    ####processing_tweet=strip_emoji(processing_tweet)

    search = ['_', '\\', '\n', '-', ',', '/', '.', '\t', '?', '!', '+', '*', '\'', '|', '#', '$', '%']
    replace = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    # remove numbers
    processing_tweet = re.sub(r'\d+', '', processing_tweet)
    processing_tweet = ' '.join(re.sub("[\n\.\,\"\!\?\:\;\-\=\؟]", " ", processing_tweet).split())
    processing_tweet = ' '.join(re.sub("[\_]", " ", processing_tweet).split())
    processing_tweet = re.sub(r'[^\x00-\x7F]+', ' ', processing_tweet)

    for i in range(0, len(search)):
        processing_tweet = processing_tweet.replace(search[i], replace[i])

    return processing_tweet.strip()


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print(memory_available)
    return np.argmax(memory_available)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
