import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from UNet import UNet
from scheduler import DDPMScheduler, DDIMScheduler
from dataset import get_dataloader
import tqdm.notebook as tqdm
import argparse
from torch_ema import ExponentialMovingAverage
import os

device = "cuda"

def train(pipeline, dataloader, num_epochs=100, save_path="results/train1/"):
    model = pipeline.model
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    losses = []
    pipeline.train()
    epoch_bar = tqdm.tqdm(range(num_epochs), desc="Epochs", leave=True)
    for epoch in epoch_bar:
        epoch_losses = []
        batch_bar = tqdm.tqdm(dataloader, desc="Batches", leave=False)
        for batch in batch_bar:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                
                loss = pipeline.get_loss(images, labels)
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                
                ema.update()
                
                avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
                batch_bar.set_postfix({"avg_loss": avg_loss})
                
        avg_loss_epoch = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss_epoch)
        epoch_bar.set_postfix({"epoch_avg_loss": avg_loss_epoch})
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exists_ok=True)
    pipeline.save_model(save_path+"model.pt")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--conditional", type=bool, default=True)
    argparser.add_argument("--num_epochs", type=int, default=100)
    argparser.add_argument("--scheduler", type=str, default="DDIM")
    argparser.add_argument("--save_path", type=str, default="results/")
    args = argparser.parse_args()
    
    conditional = args.conditional
    scheduler_type = args.scheduler
    if scheduler_type == "DDIM":
        scheduler = DDIMScheduler(device=device)
    elif scheduler_type == "DDPM":
        scheduler = DDPMScheduler(device=device)
    
    model = UNet(device=device)
    pipeline = DiffusionPipeline(model, scheduler)
    dataloader = get_dataloader()
    model = train(pipeline, dataloader)