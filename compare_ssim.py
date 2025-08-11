import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import torch
import cv2
import numpy as np
from finetune import get_drawn_box_mask
# Directories
input_dir = "outputs/input_padded"
recon_dir = "outputs/reconstructed_original"
lora_dir  = "outputs/reconstructed_lora"

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32)

def compute_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def compute_metrics(img1, img2):
    return {
        "ssim": ssim(img1, img2, data_range=255, channel_axis=-1),
        "mse": compute_mse(img1, img2),
        "psnr": psnr(img1, img2, data_range=255)
    }
'''
def main():
    image_names = sorted(os.listdir(input_dir))

    original_metrics = {"ssim": 0, "mse": 0, "psnr": 0}
    lora_metrics = {"ssim": 0, "mse": 0, "psnr": 0}

    count = 0
    for name in image_names:
        input_path = os.path.join(input_dir, name)
        recon_path = os.path.join(recon_dir, name)
        lora_path  = os.path.join(lora_dir, name)

        if not (os.path.exists(recon_path) and os.path.exists(lora_path)):
            continue

        input_img = load_image(input_path)
        recon_img = load_image(recon_path)
        lora_img  = load_image(lora_path)

        orig = compute_metrics(input_img, recon_img)
        lora = compute_metrics(input_img, lora_img)

        for k in original_metrics:
            original_metrics[k] += orig[k]
            lora_metrics[k] += lora[k]

        count += 1

    if count == 0:
        print("No matching image sets found.")
        return

    print("📊 Average Metrics over", count, "images")
    print("Original Recon:")
    print(f"  SSIM : {original_metrics['ssim']/count:.4f}")
    print(f"  MSE  : {original_metrics['mse']/count:.2f}")
    print(f"  PSNR : {original_metrics['psnr']/count:.2f} dB")
    print()
    print("LoRA Recon:")
    print(f"  SSIM : {lora_metrics['ssim']/count:.4f}")
    print(f"  MSE  : {lora_metrics['mse']/count:.2f}")
    print(f"  PSNR : {lora_metrics['psnr']/count:.2f} dB")
'''



def apply_mask(img, mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()  # Convert to NumPy

    if mask.ndim == 3 and mask.shape[0] == 1:
        # Convert [1, H, W] → [H, W, 1]
        mask = np.transpose(mask, (1, 2, 0))
    elif mask.ndim == 2:
        # Convert [H, W] → [H, W, 1]
        mask = np.expand_dims(mask, axis=-1)

    return img * mask  # Broadcasting works: [H, W, 3] * [H, W, 1]

def cv_show(tensor, window="img"):
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3:            # [C,H,W] → [H,W,C]
        img = np.transpose(img, (1,2,0))
    img = (img * 255).astype(np.uint8)
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_names = sorted(os.listdir(input_dir))

    # Foreground (animal) metrics
    fg_orig = {"ssim": 0, "mse": 0, "psnr": 0}
    fg_lora = {"ssim": 0, "mse": 0, "psnr": 0}

    # Background metrics
    bg_orig = {"ssim": 0, "mse": 0, "psnr": 0}
    bg_lora = {"ssim": 0, "mse": 0, "psnr": 0}

    count = 0
    for name in image_names:
        input_path = os.path.join(input_dir, name)
        recon_path = os.path.join(recon_dir, name)
        lora_path  = os.path.join(lora_dir, name)

        if not (os.path.exists(recon_path) and os.path.exists(lora_path)):
            continue

        input_img = load_image(input_path)           # [H, W, 3], float32
        recon_img = load_image(recon_path)
        lora_img  = load_image(lora_path)

        # Get foreground mask (1 = animal), then create background mask (1 = background)
        input_tensor = torch.from_numpy(input_img / 255.0).permute(2, 0, 1).float()  # scale for mask function
        fg_mask = get_drawn_box_mask(input_tensor)                      # [H, W], float32
        bg_mask = 1.0 - fg_mask

        # Apply masks
        input_fg = apply_mask(input_img, fg_mask)
        recon_fg = apply_mask(recon_img, fg_mask)
        lora_fg  = apply_mask(lora_img, fg_mask)


        input_bg = apply_mask(input_img, bg_mask)
        recon_bg = apply_mask(recon_img, bg_mask)
        lora_bg  = apply_mask(lora_img, bg_mask)

        # Compute metrics
        fg_orig_metrics = compute_metrics(input_fg, recon_fg)
        fg_lora_metrics = compute_metrics(input_fg, lora_fg)
        bg_orig_metrics = compute_metrics(input_bg, recon_bg)
        bg_lora_metrics = compute_metrics(input_bg, lora_bg)

        for k in fg_orig:
            fg_orig[k] += fg_orig_metrics[k]
            fg_lora[k] += fg_lora_metrics[k]
            bg_orig[k] += bg_orig_metrics[k]
            bg_lora[k] += bg_lora_metrics[k]

        count += 1

    if count == 0:
        print("No matching image sets found.")
        return

    def print_metrics(name, metrics):
        print(f"{name}:")
        print(f"  SSIM : {metrics['ssim']/count:.4f}")
        print(f"  MSE  : {metrics['mse']/count:.2f}")
        print(f"  PSNR : {metrics['psnr']/count:.2f} dB")

    print(f"📊 Average Metrics over {count} images")
    print("=== Foreground (Animal) ===")
    print_metrics("Original Recon", fg_orig)
    print_metrics("LoRA Recon", fg_lora)
    print()
    print("=== Background (Unmasked) ===")
    print_metrics("Original Recon", bg_orig)
    print_metrics("LoRA Recon", bg_lora)


if __name__ == "__main__":
    main()
