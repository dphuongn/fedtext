import os
import os.path as op
import sys
import time

from datasets import load_dataset
import numpy as np
import pandas as pd
from packaging import version
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import urllib


checkpoint = "distilbert-base-uncased"

def download_dataset():
    # The AG News dataset will be downloaded using the datasets library, so this function can be removed
    pass

def load_dataset_into_dataframe():
    raw_datasets = load_dataset("fancyzhx/ag_news")
    df = pd.DataFrame(raw_datasets['train'])

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    print("Class distribution:")
    print(df['label'].value_counts())

    return df

def partition_dataset(df):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    df_train = df_shuffled.iloc[:96_000]
    df_val = df_shuffled.iloc[96_000:108_000]
    df_test = df_shuffled.iloc[108_000:]

    if not op.exists("data"):
        os.makedirs("data")
    df_train.to_csv(op.join("data", "train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(op.join("data", "val.csv"), index=False, encoding="utf-8")
    df_test.to_csv(op.join("data", "test.csv"), index=False, encoding="utf-8")

class AGNewsDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows

def get_dataset_ag():
    files = ("test.csv", "train.csv", "val.csv")
    download = True

    for f in files:
        if not os.path.exists(op.join("data", f)):
            download = False

    if download is False:
        df = load_dataset_into_dataframe()
        partition_dataset(df)

    df_train = pd.read_csv(op.join("data", "train.csv"))
    df_val = pd.read_csv(op.join("data", "val.csv"))
    df_test = pd.read_csv(op.join("data", "test.csv"))

    return df_train, df_val, df_test

def tokenization_ag():
    ag_news_dataset = load_dataset(
        "csv",
        data_files={
            "train": op.join("data", "train.csv"),
            "validation": op.join("data", "val.csv"),
            "test": op.join("data", "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    ag_news_tokenized = ag_news_dataset.map(tokenize_text, batched=True, batch_size=None)
    ag_news_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return ag_news_tokenized

def setup_dataloaders_ag(ag_news_tokenized):
    train_dataset = AGNewsDataset(ag_news_tokenized, partition_key="train")
    val_dataset = AGNewsDataset(ag_news_tokenized, partition_key="validation")
    test_dataset = AGNewsDataset(ag_news_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=4
    )
    return train_loader, val_loader, test_loader

