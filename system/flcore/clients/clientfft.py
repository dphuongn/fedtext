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
# from utils.data_utils import read_client_data, read_client_data_clip, return_zeroshot_weight, accuracy
from torch.utils.data import Subset

from flcore.trainmodel.distilbert_model import *

from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class clientFFT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.num_labels = args.num_labels
        
        self.model_object = BertModelFFT(model_id=args.model_id, home_dir=args.home_dir, num_labels=self.num_labels).to(args.device)
        
        self.model = self.model_object.model         # bert model
        
        self.loss = nn.CrossEntropyLoss()
        
        self.train_data_fraction = args.train_data_fraction
        self.test_data_fraction = args.test_data_fraction
        
        self.optimizer = torch.optim.Adam([p for name, p in self.model.named_parameters() if p.requires_grad], lr=self.learning_rate,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
        

    def train(self):
        trainloader = self.load_train_data()
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
    
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def set_parameters(self, model):
        self.model_object.set_fft_parameters(model)
    
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
    
    
    