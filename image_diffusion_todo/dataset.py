import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
import PIL.Image as Image
import os

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im

def save_images(images, output_dir="generated_images", prefix="image"):
    os.makedirs(output_dir, exist_ok=True)

    images = images * 0.5 + 0.5

    for i, image_tensor in enumerate(images):
        # Convert the tensor to a PIL image
        grid_im = image_tensor.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
        grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
        
        # Construct the filename and save the image
        filename = f"{prefix}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        grid_im.save(filepath)
        
        print(f"Saved {filepath}")


def get_dataloader(split="train", batch_size=32):
    dataset = load_dataset("imagefolder", data_dir="data", split=split, num_proc=10)
    preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    )
    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"image": images, "label": examples["label"]}
    dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader