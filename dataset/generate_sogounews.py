import numpy as np
import os
import sys
import random
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from utils.dataset_utils_new import check, process_dataset, separate_data, separate_data_pfl, split_data, save_file, separate_data_few_shot_iid, separate_data_few_shot_pat_non_iid

random.seed(1)
np.random.seed(1)

dir_path = "agnews_hf"
if not dir_path.endswith('/'):
    dir_path += '/'

num_classes = 4  # AG News has 4 classes

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def process_tokenized_dataset(dataset):
    dataset = dataset.map(tokenize_function, batched=True)
    input_ids = np.array(dataset['input_ids'])
    attention_masks = np.array(dataset['attention_mask'])
    labels = np.array(dataset['label'])
    return input_ids, attention_masks, labels

# Allocate data to users
def generate_agnews(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl):
        return

    trainset = load_dataset("fancyzhx/ag_news", split='train', cache_dir=dir_path+"rawdata")
    testset = load_dataset("fancyzhx/ag_news", split='test', cache_dir=dir_path+"rawdata")

    dataset_input_ids = []
    dataset_attention_masks = []
    dataset_labels = []
    
    train_input_ids, train_attention_masks, train_labels = process_tokenized_dataset(trainset)
    test_input_ids, test_attention_masks, test_labels = process_tokenized_dataset(testset)

    dataset_input_ids.extend(train_input_ids)
    dataset_input_ids.extend(test_input_ids)
    dataset_attention_masks.extend(train_attention_masks)
    dataset_attention_masks.extend(test_attention_masks)
    dataset_labels.extend(train_labels)
    dataset_labels.extend(test_labels)

    dataset_input_ids = np.array(dataset_input_ids)
    dataset_attention_masks = np.array(dataset_attention_masks)
    dataset_labels = np.array(dataset_labels)

    if pfl:
        X, y, statistic = separate_data_pfl((dataset_input_ids, dataset_attention_masks, dataset_labels), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)

        train_data, test_data = split_data(X, y)

        for idx, test_dict in enumerate(train_data):
            print(f'train data: {idx}')
            print(f'train data shape: {len(train_data[idx]["y"])}')
        for idx, test_dict in enumerate(test_data):
            print(f'test data: {idx}')
            print(f'test data shape: {len(test_dict["x"])}')
    
    elif few_shot:
        if not niid:
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_iid((dataset_input_ids, dataset_attention_masks, dataset_labels), 
                                                        num_clients, num_classes, n_shot)
        else:
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_pat_non_iid((dataset_input_ids, dataset_attention_masks, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        
    else:
        train_data, test_data, statistic, statistic_test = separate_data((dataset_input_ids, dataset_attention_masks, dataset_labels), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)
        
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha, few_shot, n_shot, pfl)


if __name__ == "__main__":
    # Check if the minimum number of arguments is provided
    if len(sys.argv) < 7:
        print("Usage: script.py num_clients niid balance partition alpha few_shot [n_shot]")
        sys.exit(1)

    # Parse arguments
    try:
        num_clients = int(sys.argv[1])
    except ValueError:
        print("Invalid input for num_clients. Please provide an integer value.")
        sys.exit(1)

    niid = sys.argv[2].lower() == "noniid"
    balance = sys.argv[3].lower() == "balance"
    partition = sys.argv[4]
            
    # Alpha is required only for non-IID data with "dir" partition
    alpha = None
    if niid and partition == "dir":
        if len(sys.argv) < 6 or sys.argv[5] == "-":
            print("Alpha parameter is required for non-IID 'dir' partitioned data.")
            sys.exit(1)
        try:
            alpha = float(sys.argv[5])
        except ValueError:
            print("Invalid input for alpha. Please provide a float value.")
            sys.exit(1)
    elif len(sys.argv) >= 6 and sys.argv[5] != "-":
        # Optional alpha for other cases
        try:
            alpha = float(sys.argv[5])
        except ValueError:
            print("Invalid input for alpha. Please provide a float value or '-' for default.")
            sys.exit(1)

    few_shot = sys.argv[6].lower() in ["true", "fs"]

    n_shot = None
    if few_shot:
        if len(sys.argv) < 8:
            print("n_shot parameter is required for few_shot mode.")
            sys.exit(1)
        try:
            n_shot = int(sys.argv[7])
        except ValueError:
            print("Invalid input for n_shot. Please provide an integer value.")
            sys.exit(1)
            
    pfl = sys.argv[8].lower() == "pfl"
    
    # Print all parsed arguments
    print(f"Running script with the following parameters:")
    print(f"num_clients: {num_clients}")
    print(f"niid: {niid}")
    print(f"balance: {balance}")
    print(f"partition: {partition}")
    print(f"alpha: {alpha}")
    print(f"few_shot: {few_shot}")
    print(f"n_shot: {n_shot}")
    print(f"pfl: {pfl}")

    generate_agnews(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl)
