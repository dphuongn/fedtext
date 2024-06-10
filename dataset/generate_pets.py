# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
from datasets import load_dataset
import torchvision.transforms as transforms
from utils.dataset_utils_new import check, process_dataset, separate_data, separate_data_pfl, split_data, save_file, separate_data_few_shot_iid, separate_data_few_shot_pat_non_iid


random.seed(1)
np.random.seed(1)

dir_path = "pets_hf"
if not dir_path.endswith('/'):
    dir_path += '/'
    
# dir_path = os.path.join('pets_hf', 'rawdata')
    
num_classes = 37

def resize_and_normalize(example):
    image = example['image']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to (224, 224)
        transforms.ToTensor(),        # Convert to tensor and normalize to [0, 1]
    ])
    image = transform(image)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert single channel to three channels
    example['image'] = image
    return example

# Process dataset using Hugging Face datasets
def process_hf_dataset(data):
    data = data.map(resize_and_normalize, batched=False)
    images = torch.stack([example['image'].numpy() for example in data])
    labels = torch.tensor([example['label'] for example in data])
    return images, labels

def resize_and_normalize(example):
    image = example['image']
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to (28, 28)
        transforms.ToTensor(),        # Convert to tensor and normalize to [0, 1]
    ])
    image = transform(image)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert single channel to three channels
    example['image'] = image
    return example

# Process dataset using Hugging Face datasets
def process_hf_dataset(data):
    data = data.map(resize_and_normalize, batched=False)
    images = torch.stack([example['image'] for example in data])
    labels = torch.tensor([example['label'] for example in data])
    return images.numpy(), labels.numpy()

# Allocate data to users
def generate_pets(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl):
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize all images to 224x224
        transforms.ToTensor() # Then convert them to tensor
    ])

    trainset = load_dataset("visual-layer/vl-oxford-iiit-pets", split='train', cache_dir=dir_path+"rawdata")
    
    
    dataset_image = []
    dataset_label = []

    # train_images, train_labels = process_dataset(trainset)
    
    train_images, train_labels = process_hf_dataset(trainset)
    
    # dataset_image.extend(train_images)
    # dataset_image.extend(test_images)
    # dataset_label.extend(train_labels)
    # dataset_label.extend(test_labels)
    # dataset_image = np.array(dataset_image)
    # dataset_label = np.array(dataset_label)
    
    dataset_image.extend(train_images)
    dataset_label.extend(train_labels)
    
    if pfl:
        X, y, statistic = separate_data_pfl((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)
        
        train_data, test_data = split_data(X, y)
        
        for idx, test_dict in enumerate(train_data):
            print(f'train data: {idx}')
            print(f'train data shape: {len(train_data[idx]["y"])}')
        for idx, test_dict in enumerate(test_data):
            print(f'test data: {idx}')
            print(f'test data shape: {len(test_dict["x"])}')
    

    elif few_shot:  # Add a parameter or a condition to trigger few-shot scenario
        if not niid:
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_iid((dataset_image, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        else:
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_pat_non_iid((dataset_image, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        
    else:

        train_data, test_data, statistic, statistic_test = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)
        # train_data, test_data = split_data(X, y)
        
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

    generate_pets(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl)