
import numpy as np
import os
import torch
from PIL import Image
from tqdm.auto import tqdm
import evaluate

from torchvision.transforms import ToPILImage

# from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding


checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def read_data(dataset, idx, is_train=True):
    if is_train:
        current_directory = os.getcwd()
        # print("Current Working Directory:", current_directory)
        
        # train_data_dir = os.path.join('../../dataset', dataset, 'train/')
        train_data_dir = os.path.join('../dataset', dataset, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file {train_file} not found.")
        
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        
        print(f"Loaded train data from {train_file}")
        print(f"Data length: {len(train_data)}")
        # print(f"Keys in the loaded train data: {train_data.keys()}")  # Print the keys

        
        return train_data['x']

    else:
        # test_data_dir = os.path.join('../../dataset', dataset, 'test/')
        test_data_dir = os.path.join('../dataset', dataset, 'test/')
        test_file = test_data_dir + str(idx) + '.npz'
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file {test_file} not found.")
        
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        
        print(f"Loaded test data from {test_file}")
        print(f"Data length: {len(test_data)}")
        # print(f"Keys in the loaded test data: {test_data.keys()}")  # Print the keys

        
        return test_data['x']
    

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]  # Use .item() instead
    

    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Example usage to check train data for client 0
    dataset = 'ag_news'  # replace with your dataset name
    train_data = read_data(dataset, idx=0, is_train=True)
    print("Type of train data:", type(train_data))  # Print the type of train data
    # print("Train data example:", train_data)  # Print the contents to check the structure

    # Example usage to check test data for client 0
    test_data = read_data(dataset, idx=0, is_train=False)
    print("Type of test data:", type(test_data))  # Print the type of test data
    # print("Test data example:", test_data)  # Print the contents to check the structure

    
    
    
    
#     current_directory = os.getcwd()
#     print("Current Working Directory:", current_directory)
    
#     # train_data = read_client_data_clip('digit5', 1, processor, class_names, device, is_train=True)
    
#     dataset = 'ag_news'  # replace with your dataset name
#     train_data = read_data(dataset, idx=0, is_train=True)
#     print("Train data example:", train_data[:1])  # Print the first element to check the structure

#     test_data = read_data(dataset, idx=0, is_train=False)
#     print("Test data example:", test_data[:1])  # Print the first element to check the structure

    
    test_dataloader = DataLoader(test_data, batch_size=8, collate_fn=data_collator)
    
    batch = next(iter(test_dataloader))
    print(batch)
    
#     x = next(iter(test_dataloader))
#     print(x[0].shape)
#     print(x[1].shape)
    