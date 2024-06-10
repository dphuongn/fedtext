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

def load_dataset_into_dataframe(dataset):
    
    if dataset == 'ag':
        raw_datasets = load_dataset("fancyzhx/ag_news")
        df = pd.DataFrame(raw_datasets['train'])
    
    elif dataset == 'sogou_news':
        raw_datasets = load_dataset("community-datasets/sogou_news")
        df = pd.DataFrame(raw_datasets['train'])
        
    else:
        raise ValueError("Invalid dataset name")

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    print("Class distribution:")
    print(df['label'].value_counts())

    return df

def partition_dataset(df, dataset):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    if dataset == 'ag':
        df_train = df_shuffled.iloc[:84_000]
        df_val = df_shuffled.iloc[84_000:96_000]
        df_test = df_shuffled.iloc[96_000:]
        
    elif dataset == 'sogou_news':
        df_train = df_shuffled.iloc[:315_000]
        df_val = df_shuffled.iloc[315_000:360_000]
        df_test = df_shuffled.iloc[360_000:]
        
    else:
        raise ValueError("Invalid dataset name")
        
        
    data_path = f"data/{dataset}"

    if not op.exists(data_path):
        os.makedirs(data_path)
    df_train.to_csv(op.join(data_path, "train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(op.join(data_path, "val.csv"), index=False, encoding="utf-8")
    df_test.to_csv(op.join(data_path, "test.csv"), index=False, encoding="utf-8")

    
# class AGNewsDataset(Dataset):
#     def __init__(self, dataset_dict, partition_key="train"):
#         self.partition = dataset_dict[partition_key]

#     def __getitem__(self, index):
#         return self.partition[index]

#     def __len__(self):
#         return self.partition.num_rows
    
    
class CustomDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows

def get_dataset_new(dataset):
    files = ("test.csv", "train.csv", "val.csv")
    download = True
    
    data_path = f"data/{dataset}"

    for f in files:
        if not os.path.exists(op.join(data_path, f)):
            download = False

    if download is False:
        df = load_dataset_into_dataframe(dataset)
        partition_dataset(df, dataset)

    df_train = pd.read_csv(op.join(data_path, "train.csv"))
    df_val = pd.read_csv(op.join(data_path, "val.csv"))
    df_test = pd.read_csv(op.join(data_path, "test.csv"))

    return df_train, df_val, df_test

def tokenization_new(dataset):
    
    data_path = f"data/{dataset}"
    
    dataset_loaded = load_dataset(
        "csv",
        data_files={
            "train": op.join(data_path, "train.csv"),
            "validation": op.join(data_path, "val.csv"),
            "test": op.join(data_path, "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if dataset == 'ag':
        def tokenize_text(batch):
            return tokenizer(batch["text"], truncation=True, padding=True)
        
    elif dataset == 'sogou_news':
        def tokenize_text(batch):
            return tokenizer(batch["content"], truncation=True, padding=True)
        
    else:
        raise ValueError("Invalid dataset name")

    dataset_tokenized = dataset_loaded.map(tokenize_text, batched=True, batch_size=None)
    dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return dataset_tokenized

def setup_dataloaders_new(dataset_tokenized):
    train_dataset = CustomDataset(dataset_tokenized, partition_key="train")
    val_dataset = CustomDataset(dataset_tokenized, partition_key="validation")
    test_dataset = CustomDataset(dataset_tokenized, partition_key="test")

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

