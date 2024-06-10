import argparse
import os
import shutil
import time
from functools import partial

from transformers import CLIPProcessor, CLIPModel

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from local_dataset_utilities import setup_dataloaders, zeroshot_classifier, accuracy

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
# HOME = '/export/work/yusx/phuong'
HOME = '/work/LAS/jannesar-lab/dphuong/'
model_id = "openai/clip-vit-base-patch32"


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune all parameters configuration')
    parser.add_argument('--data_fraction', type=float, default=0.001, help='Fraction of training data to use')
    parser.add_argument('--device', type=int, default=0, help='Specify GPU device index')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Enable/disable progress bars')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")
        quit()
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    model = CLIPModel.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    
    train_dataloader, test_dataloader = setup_dataloaders(processor, device, args.data_fraction)


    num_params = count_parameters(model)
    print("Total number of trainable parameters:", num_params)
    
    
    # Prepare the optimizer
    # the lr is smaller, more safe for fine tuning to new dataset
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if p.requires_grad], lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    
    
    # Specify the loss function
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    
    model.to(device)
    

    start = time.time()


    # model.train()
    num_epochs = 1
    start = time.time()

    # Train the model
    for epoch in range(num_epochs):
        with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:  # Initialize pbar here
            for batch in pbar:    

                images, target, texts = batch

                # texts is a dictionary, extract the required tensors
                input_ids = texts['input_ids'].squeeze(1) # Remove the extra dimension
                attention_mask = texts['attention_mask'].squeeze(1) # Remove the extra dimension
                
                image_features = model.get_image_features(images).float()
                
                text_features = model.get_text_features(input_ids=input_ids, 
                                                            attention_mask=attention_mask).float()
                
                
                image_features = image_features / \
                    image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                    text_features.norm(dim=1, keepdim=True)
                
                # logit_scale = model.model.logit_scale.exp()
                logit_scale = model.state_dict()['logit_scale'].exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()


                # Compute loss
                # ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
                
                targets = torch.eye(logits_per_image.size(0), dtype=torch.float32, device=logits_per_image.device)
                total_loss = nn.functional.binary_cross_entropy_with_logits(logits_per_image, targets)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
                
    end = time.time()
    elapsed = end-start
    print(f"Time elapsed {elapsed/60:.2f} min")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    
    zeroshot_weights_1 = zeroshot_classifier(model, processor, device)
        
    
    with torch.no_grad():
        top1_1, top5_1, n = 0., 0., 0.
        
        for i, (images, target, texts) in enumerate(tqdm(test_dataloader)):
            images = images
            target = target.to(device)
            texts = texts
            # print(image.shape)
            # break

            # predict
            image_features = model.get_image_features(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # measure accuracy of 1 template
            logits = 100. * image_features @ zeroshot_weights_1
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1_1 += acc1
            top5_1 += acc5


            n += images.size(0)

    top1_1 = (top1_1 / n) * 100
    top5_1 = (top5_1 / n) * 100 


    # Print settings and results
    with open("results_all.txt", "a") as f:
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")        
        s = f"Total number of trainable parameters: {num_params}"    
        print(s), f.write(s+"\n")
    
        s = f"Time elapsed: {elapsed/60:.2f} min"
        print(s), f.write(s+"\n")
        
        print(f"Time elapsed {elapsed/60:.2f} min")
        
        s = f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
        print(s), f.write(s+"\n")
        
        for arg in vars(args):
            s = f'{arg}: {getattr(args, arg)}'
            print(s), f.write(s+"\n")

        s = f"Top-1 test acc: {top1_1:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Top-5 test acc: {top5_1:2.2f}%"
        print(s), f.write(s+"\n")
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")    

    # Cleanup
    # log_dir = f"logs/my-model-{args.device}"
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)