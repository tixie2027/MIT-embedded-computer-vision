import cv2
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from capture_image import capture_images
from demo import get_lora_model




import torch
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import compressai
from Iwildcam_Pretrain import CompressaiWrapper
from lora_modules import LoRAConv2d, LoRALinear, LoRAConvTranspose2d
import torch.nn as nn
import torch.optim as optim
import cv2
from IPython.display import clear_output, display
import time







device = torch.device("cpu")


print('checkpoint 1')

NUM_IMAGES = 4
images = capture_images(NUM_IMAGES).to(device)

fig, axes = plt.subplots(2, NUM_IMAGES // 2, figsize=(15, 6))
for i in range(NUM_IMAGES):
    img = images[i].permute(1, 2, 0).cpu().numpy()
    img = img.clip(0, 1)
    axes[i % 2, i // 2].imshow(img)
    axes[i % 2, i // 2].axis('off')

plt.tight_layout()
plt.show(block=False)
plt.pause(2)  # Show for 2 seconds
plt.close()
print('IMAGES HAVE BEEN PROCESSED')


model = get_lora_model()
model.to(device)
model.train()
print('checkpoiuint 2')

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.MSELoss()

batch_size = 2
num_epochs = 10

lora_params_initial = {
    n: p.detach().cpu().clone() 
    for n, p in model.named_parameters() if "delta" in n.lower()
}
print("initial lora params:", lora_params_initial.keys())

# plot over time
lora_param_changes = []
losses = []

fig, axes = plt.subplots(2, 5, figsize=(15,8.5))
loss_ax = axes[0,0]
weights_ax = axes[1,0]
orig_axes = axes[0,1:]
recon_axes = axes[1,1:]
current_epoch = 0

for epoch in range(num_epochs):
    print("current epoch:", current_epoch)
    current_epoch += 1

    current_batch = 0
    for i in range(0, images.size(0), batch_size):
        print("current batch:", current_batch)
        current_batch += 1
        
        img = images[i:i+batch_size].to(device)
        
        # out = model(img)
        x = img
        y = model.g_a(x)
        z = model.h_a(torch.abs(y))
        z_hat, z_likelihoods = model.entropy_bottleneck(z)
        scales_hat = model.h_s(z_hat)

        # this op not supported on mps
        y = y.to("cpu")
        scales_hat = scales_hat.to("cpu")
        model = model.to("cpu")
        if y.shape != scales_hat.shape:
            # Resize scales_hat to match y's spatial dimensions
            scales_hat = torch.nn.functional.interpolate(scales_hat, size=y.shape[2:], mode='nearest')


        y_hat, y_likelihoods = model.gaussian_conditional(y, scales_hat)

        # put stuff back on mps for backward pass and loss
        y_hat = y_hat.to(device)
        model = model.to(device)
        
        x_hat = model.g_s(y_hat) #["x_hat"]
        
        loss = criterion(x_hat, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # get lora update norms
        current_lora_params = {
            n: p.detach().cpu().clone() 
            for n, p in model.named_parameters() if "delta_" in n.lower()
        }
        diff_norm = 0.0
        for n in lora_params_initial:
            diff_norm += (current_lora_params[n] - lora_params_initial[n]).pow(2).sum().item()
        diff_norm = diff_norm**0.5
        lora_param_changes.append(diff_norm)

        # update display
        clear_output(wait=True)

        # loss plot
        loss_ax.clear()
        loss_ax.plot(losses)
        loss_ax.set_title("Training Loss")
        loss_ax.set_xlabel("Iteration")
        loss_ax.set_ylabel("MSE Loss")

        # lora plot
        weights_ax.clear()
        weights_ax.plot(lora_param_changes)
        weights_ax.set_title("LoRA Weight Changes")
        weights_ax.set_xlabel("Iteration")
        weights_ax.set_ylabel("L2 Norm of ΔWeights")

        for j in range(min(4, img.size(0))):
            orig_img_np = img[j].detach().cpu().permute(1,2,0).numpy().clip(0,1) # use middle 4 images
            x_hat_np = x_hat[j].detach().cpu().permute(1,2,0).numpy().clip(0,1) # use middle 4 images

            orig_axes[j].clear()
            orig_axes[j].imshow(orig_img_np)
            orig_axes[j].axis('off')
    
            recon_axes[j].clear()
            recon_axes[j].imshow(x_hat_np)
            recon_axes[j].axis('off')

        recon_axes[0].set_title("Reconstructed Images")
        orig_axes[0].set_title("Original Images")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5)  # Show for 2 seconds
        plt.close()
        # plt.pause(0.001)

print("done")