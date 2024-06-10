import argparse
import os
import shutil
import time
from functools import partial

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoModelForSequenceClassification
import torch
import torch.nn as nn

from local_dataset_utilities import tokenization, setup_dataloaders, get_dataset
# from local_dataset_utilities_ag import tokenization_ag, setup_dataloaders_ag, get_dataset_ag
# from local_dataset_utilities_sogou import tokenization_sogou, setup_dataloaders_sogou, get_dataset_sogou
from local_dataset_utilities_new import tokenization_new, setup_dataloaders_new, get_dataset_new
from local_model_utilities import CustomLightningModule

# torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
HOME = '/work/LAS/jannesar-lab/dphuong/'
checkpoint = "distilbert-base-uncased"


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
class LoRALayer(nn.Module):
    def __init__(self, 
         in_dim, 
         out_dim, 
         rank: int, 
         alpha: int, 
         dropout: float, 
         # merge_weights: bool
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # self.merge_weights = merge_weights

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else (lambda x: x)

        # Scaling
        # self.scaling = self.alpha / self.rank
        self.scaling = self.alpha

        # Merged flag
        # self.merged = False

    def forward(self, x):
        # if self.rank > 0 and not self.merged:
        if self.rank > 0:
            x = self.dropout(x)
            x = self.scaling * (x @ self.W_a @ self.W_b)
        return x
            
class LinearWithLoRA(nn.Module):
    def __init__(self, 
         linear, 
         rank: int = 0, 
         alpha: int = 1, 
         dropout: float = 0.0, 
         # merge_weights: bool = True,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha, 
            dropout, 
            # merge_weights
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LoRA parameters configuration')
    parser.add_argument('--lora_r', type=int, default=8, help='Rank for LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.00, help='Dropout for LoRA layers')
    
    parser.add_argument('--lora_query', type=str2bool, default=False, help='Apply LoRA to query')
    parser.add_argument('--lora_key', type=str2bool, default=False, help='Apply LoRA to key')
    parser.add_argument('--lora_value', type=str2bool, default=False, help='Apply LoRA to value')
    parser.add_argument('--lora_projection', type=str2bool, default=False, help='Apply LoRA to projection')
    parser.add_argument('--lora_mlp', type=str2bool, default=False, help='Apply LoRA to MLP')
    parser.add_argument('--lora_head', type=str2bool, default=False, help='Apply LoRA to head')
    
    parser.add_argument('--device', type=int, default=0, help='Specify GPU device index')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Enable/disable progress bars')
    
    parser.add_argument('-data', "--dataset", type=str, default="pets")
    parser.add_argument('-sd', "--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")
        quit()
        
    
    # Set the seed
    torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    
    
    if args.dataset == 'aclImdb':
        df_train, df_val, df_test = get_dataset()
        imdb_tokenized = tokenization()
        train_loader, val_loader, test_loader = setup_dataloaders(imdb_tokenized)
    
        num_labels = 2
        
        results = 'aclImdb'
        
    elif args.dataset == 'ag_news':
        df_train, df_val, df_test = get_dataset_new(args.dataset)
        new_tokenized = tokenization_new(args.dataset)
        train_loader, val_loader, test_loader = setup_dataloaders_new(new_tokenized)
    
        num_labels = 4
        
        results = 'ag_news'
        
    elif args.dataset == 'sogou_news':
        
        df_train, df_val, df_test = get_dataset_new(args.dataset)
        dataset_tokenized = tokenization_new(args.dataset)
        train_loader, val_loader, test_loader = setup_dataloaders_new(dataset_tokenized)
    
        num_labels = 5
        
        results = 'sogou_news'
        
    else:
        raise ValueError("Invalid dataset name")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=args.lora_r, alpha=args.lora_alpha)
    
    
    assign_lora = partial(
            LinearWithLoRA,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    
    
        
    for layer in model.distilbert.transformer.layer:
        if args.lora_query:
            layer.attention.q_lin = assign_lora(layer.attention.q_lin)
        if args.lora_key:
            layer.attention.k_lin = assign_lora(layer.attention.k_lin)
        if args.lora_value:
            layer.attention.v_lin = assign_lora(layer.attention.v_lin)
        if args.lora_projection:
            layer.attention.out_lin = assign_lora(layer.attention.out_lin)
        if args.lora_mlp:
            layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
            layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
    if args.lora_head:
        model.pre_classifier = assign_lora(model.pre_classifier)
        model.classifier = assign_lora(model.classifier)
    

    # print("Total number of trainable parameters:", count_parameters(model))
    

    lightning_model = CustomLightningModule(model, num_labels)
    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="val_acc"
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name=f"my-model-{args.dataset}-{args.device}") 

    trainer = L.Trainer(
        max_epochs=1,
        callbacks=callbacks,
        accelerator="gpu",
        precision="16-mixed",
        devices=[int(args.device)],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=args.verbose
    )
    
    start = time.time()

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)


    # Print all argparse settings
    print("------------------------------------------------")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)

    # Print settings and results
    with open(f"{results}.txt", "a") as f:
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")        
        for arg in vars(args):
            s = f'{arg}: {getattr(args, arg)}'
            print(s), f.write(s+"\n")
            
        args.dataset
        
        s = f"Dataset: {args.dataset}"
        print(s), f.write(s + "\n")
            
        s = f"Total number of trainable parameters: {count_parameters(model)}"
        print(s), f.write(s + "\n")

        s = f"Train acc: {train_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Val acc:   {val_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Test acc:  {test_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")    

    # Cleanup
    log_dir = f"logs/my-model-{args.dataset}-{args.device}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        