import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from capture_image import capture_images
import numpy as np
import os
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from compressai.zoo import bmshj2018_hyperprior
from Iwildcam_Pretrain import Autoencoder, IWildCamDataset, CompressaiWrapper
from lora_modules import LoRAConv2d, LoRALinear, LoRAConvTranspose2d
import pytorch_lightning as pl

#pretrained_filename = '8-epoch.04-val_loss.0.01.ckpt'

def pad_to_multiple_of(x, base=64):
    """Pads a 4D tensor (N, C, H, W) to the next multiple of `base`."""
    _, _, h, w = x.shape
    h_pad = (base - h % base) % base
    w_pad = (base - w % base) % base
    return torch.nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)

def main():
    # Create directory for saving images
    os.makedirs("captured_images", exist_ok=True)

    # Capture batch of images as tensor: [N, C, H, W]
    images = capture_images() 
    print(f"[DEBUG] Raw captured shape: {images.shape}")

    # Step 2: Pad image dimensions to be multiples of 64
    images = pad_to_multiple_of(images, base=64)
    print(f"[DEBUG] Padded image shape: {images.shape}")
    for i, img_tensor in enumerate(images):
        save_image(img_tensor, f"captured_images/image_{i}.png")

    # Instantiate the wrapper
    wrapper = CompressaiWrapper()  
    model = wrapper.model

    # Now this works
    wrapper.model.update()  # Needed before compress()
    out = wrapper.compress(images)
    data = out["strings"][0][0].hex(), out["strings"][0][1].hex(), out["strings"][1][0].hex(), out["strings"][1][1].hex()
   
    with open("embeddings.txt", "w") as f:
        f.write(",".join(data))

    # Step 5: Calculate bit sizes
    compressed_bits = sum(len(s) for s in out["strings"][0]) * 8
    original_bits = images.shape[-1] * images.shape[-2] * 3 * 8

    # Step 6: Report compression
    print("out shape", out['shape'])
    print(f"\n[✓] Original size: {original_bits / 8:.1f} bytes")
    print(f"[✓] Compressed size: {compressed_bits / 8:.1f} bytes")
    print(f"[✓] Compression ratio: {original_bits / compressed_bits:.2f}x\n")

    # Optional: save reconstructed image
    recon = model.decompress(out["strings"], out["shape"])
    x_hat = recon["x_hat"].clamp(0, 1)
    save_image(x_hat[0], "reconstructed.png")
    print("[✓] Reconstructed image saved as reconstructed.png")


if __name__ == "__main__":
    main()