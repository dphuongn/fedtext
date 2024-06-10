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

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_data
from torch.utils.data import Subset

from transformers import AutoTokenizer, DataCollatorWithPadding

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        # for layer in self.model.children():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         self.has_BatchNorm = True
        #         break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, 
        #     gamma=args.learning_rate_decay_gamma
        # )
        self.learning_rate_decay = args.learning_rate_decay


#     def load_train_data(self, batch_size=None):
#         if batch_size == None:
#             batch_size = self.batch_size
#         train_data = read_client_data(self.dataset, self.id, is_train=True)
#         return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

#     def load_test_data(self, batch_size=None):
#         if batch_size == None:
#             batch_size = self.batch_size
#         test_data = read_client_data(self.dataset, self.id, is_train=False)
#         return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # train_data = read_client_data_clip(self.dataset, self.id, self.processor, self.class_names, self.device, is_train=True)
        
        train_data = read_data(self.dataset, self.id, is_train=True)
        
        train_subset_size = int(len(train_data) * self.train_data_fraction)
        train_indices = np.random.choice(len(train_data), train_subset_size, replace=False)
        train_subset = Subset(train_data, train_indices)
        
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        
        return train_dataloader

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
            
        # test_data = read_client_data_clip(self.dataset, self.id, self.processor, self.class_names, self.device, is_train=False)
        
        test_data = read_data(self.dataset, self.id, is_train=False)
        
        test_subset_size = int(len(test_data) * self.test_data_fraction)
        test_indices = np.random.choice(len(test_data), test_subset_size, replace=False)
        test_subset = Subset(test_data, test_indices)
        
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
        
        return test_dataloader
        
    def set_parameters(self, model):
        print(f'model: {model}')
        print(f'self.model: {self.model}')
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()
    
    
    


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))