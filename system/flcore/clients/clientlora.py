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
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_data, accuracy
from torch.utils.data import Subset

from flcore.trainmodel.distilbert_model import *

from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler

import evaluate

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class clientLORA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.lora_params = args.lora_params
        
        print(f'self.lora_params: {self.lora_params}')
        
        self.num_labels = args.num_labels
        
        self.model_object = DistilBertModelWithLoRA(model_id=args.model_id, home_dir=args.home_dir, lora_params=self.lora_params, num_labels=self.num_labels).to(args.device)
        
        # self.clip_model_object = copy.deepcopy(args.model)
        # self.model = copy.deepcopy(args.model.model)
        
        self.model = self.model_object.model
        
        # self.processor = self.clip_model_object.processor
        
        # self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()
        
        self.loss = nn.CrossEntropyLoss()
        
        self.train_data_fraction = args.train_data_fraction
        self.test_data_fraction = args.test_data_fraction
        
        # self.class_names = args.class_names
        
        self.optimizer = torch.optim.Adam([p for name, p in self.model.named_parameters() if p.requires_grad], lr=self.learning_rate,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
        
        
        # print(f"print LoRA parameters before training:")
        # for name, param in self.clip_model.named_parameters():
        #     # Check if the parameter's parent module is a LoRALayer
        #     if 'lora' in name:
        #         print(f"{name}: {param.data}")
        
#         num_param = self.clip_model_object.count_parameters()
#         print("Trained parameters in model: {:,}".format(num_param))
        
#         clip_model_size = self.clip_model_object.calculate_model_size(self.clip_model)
#         print('Size of clip model: {:.3f} MB'.format(clip_model_size))
        
#         lora_state_dict = self.clip_model_object.get_lora_state_dict()
#         lora_state_dict_size = self.clip_model_object.calculate_state_dict_size(lora_state_dict) 
#         print('Size of lora state edict: {:.3f} MB'.format(lora_state_dict_size))

        
        
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

    def train(self):
        trainloader = self.load_train_data()
        # self.clip_model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            
            
        num_training_steps = max_local_epochs * len(trainloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        
        
        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for epoch in range(max_local_epochs):
            for batch in trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

        end = time.time()
        elapsed = end-start
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        
        # print LoRA parameters
        # print(f"print LoRA parameters after training:")
        # for name, param in self.clip_model.named_parameters():
        #     # Check if the parameter's parent module is a LoRALayer
        #     if 'lora' in name:
        #         print(f"{name}: {param.data}")
            
            
            
            

        # self.clip_model.cpu()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            
            
            
    # added ------------------------------------------------------
    
    
    
    def test_metrics(self):
        testloaderfull = self.load_test_data()

        metric = evaluate.load("accuracy")
        self.model.eval()
        
            
        for batch in testloaderfull:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"].cpu())

        result = metric.compute()
        
        print(f'metric.compute(): {result}')
        
        accuracy_value = result['accuracy']

        
        return accuracy_value, 0, 0
        
        
                
                
#         with torch.no_grad():
#             top1_1, top5_1, test_num = 0., 0., 0.

#             # for i, (images, target, texts) in enumerate(tqdm(testloaderfull)):
#             for i, (images, target, texts) in enumerate(testloaderfull):
#                 images = images
#                 target = target.to(self.device)
#                 texts = texts

#                 # predict
#                 image_features = self.clip_model.get_image_features(images)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)

#                 # measure accuracy of 1 template
#                 zeroshot_weights_1 = return_zeroshot_weight(self.dataset, self.clip_model, self.processor, self.class_names, self.device)
#                 logits = self.logit_scale * image_features @ zeroshot_weights_1
#                 # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
#                 acc1 = accuracy(logits, target, topk=(1,))
#                 top1_1 += acc1[0]
#                 # top5_1 += acc5

#                 test_num += images.size(0)

#         top1_1 = (top1_1 / test_num) * 100
#         top5_1 = (top5_1 / test_num) * 100 

#         print(f"accuracy of 1 template:")
#         print(f"Top-1: {top1_1:.2f}, Top-5: {top5_1:.2f}")
    
#         return top1_1, test_num, 0
    
    
    def set_parameters(self, lora_state_dict):
        # Updating LoRA parameters from another state dict 
        
        # Assuming `self.clip_model_object` has a method `set_lora_state_dict` that accepts a state dict of parameters
        self.model_object.set_lora_state_dict(lora_state_dict)
        # self.clip_model_object.set_lora_parameters(lora_params_list)
    # ------------------------------------------------------------
    
    
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    
    
#     client = clientAVGC(, 
#                             id=i, 
#                             train_samples=len(train_data), 
#                             test_samples=len(test_data), 
#                             train_slow=train_slow, 
#                             send_slow=send_slow)
#             self.clients.append(client)
    
    
    