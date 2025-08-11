import torch
from torch import nn
from Iwildcam_Pretrain import CompressaiWrapper
from torchvision.utils import save_image
from lora_modules import LoRAConv2d
import os
from finetune import replace_conv_with_lora, lora_config, patch_first_conv2d_input_channels
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from compressai.zoo import bmshj2018_hyperprior
from maskedmodel import MaskConditionedHyperprior
from finetunenorm import get_drawn_box_mask
from torchvision.transforms import GaussianBlur
from get_lora_model import get_lora_model


IMAGE_DIR = 'preview'

def load_and_pad_image(image_path, base=64):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # [1, 3, H, W]

    # Pad to multiple of 64
    h, w = image.shape[2], image.shape[3]
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    image_padded = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return image, image_padded

def main():
    image_dir = IMAGE_DIR
    image_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Create output folders
    os.makedirs("outputs/input", exist_ok=True)
    os.makedirs("outputs/input_padded", exist_ok=True)
    os.makedirs("outputs/reconstructed_original", exist_ok=True)
    os.makedirs("outputs/reconstructed_lora", exist_ok=True)

    # Load models
    wrapper = CompressaiWrapper()
    wrapper.model.update()



    wrapper_lora = get_lora_model()
    #wrapper_lora = CompressaiWrapper().model
    wrapper_lora.load_state_dict(torch.load("lora_encoder_finetuned.pth", map_location="cpu"))
    wrapper_lora.update()
    wrapper_lora.eval()
    #patch_first_conv2d_input_channels(wrapper_lora.model.g_a, new_in_channels=4)
    #wrapper_lora = CompressaiWrapper()
    #replace_conv_with_lora(wrapper_lora.model.g_a, lora_config)
   # replace_conv_with_lora(wrapper_lora.model.h_a, lora_config)
  



    total_ratio_original = 0
    total_ratio_lora = 0

    for i, image_name in enumerate(tqdm(image_names, desc="Processing images")):
        image_path = os.path.join(image_dir, image_name)
        image, image_padded = load_and_pad_image(image_path)
        masks = torch.stack([get_drawn_box_mask(img) for img in image_padded])  # [B, 1, H, W]
        blur = GaussianBlur(kernel_size=15, sigma=2)
        blurred = blur(image_padded)  # takes a [B, C, H, W] Tensor and blurs each image


        mask_fg   = masks.expand_as(image_padded)
        batch_pre = image_padded * mask_fg + blurred * (1 - mask_fg)

        # Save input and padded input
        save_image(image, f"outputs/input/{image_name}")
        save_image(image_padded, f"outputs/input_padded/{image_name}")

        # Original model
        out = wrapper.model.compress(batch_pre)
        compressed_size = sum(len(s[0]) for s in out["strings"])
        original_size = image_padded.numel()
        compression_ratio = original_size / compressed_size
        total_ratio_original += compression_ratio
        recon = wrapper.model.decompress(out["strings"], out["shape"])
        recon_image = recon["x_hat"].clamp(0, 1)
        save_image(recon_image[0], f"outputs/reconstructed_original/{image_name}")


        # LoRA model
        out_lora = wrapper_lora.compress(batch_pre)
        compressed_size_lora = sum(len(s[0]) for s in out_lora["strings"])
        compression_ratio_lora = original_size / compressed_size_lora
        total_ratio_lora += compression_ratio_lora

        recon_lora = wrapper_lora.decompress(out_lora["strings"], out_lora["shape"])
        recon_lora_image = recon_lora["x_hat"].clamp(0, 1)
        save_image(recon_lora_image[0], f"outputs/reconstructed_lora/{image_name}")

    num_images = len(image_names)
    avg_ratio_original = total_ratio_original / num_images
    avg_ratio_lora = total_ratio_lora / num_images

    print("✅ All reconstructions saved to 'outputs/' directory.")
    print(f"📊 Average Compression Ratio - Original: {avg_ratio_original:.2f}, LoRA: {avg_ratio_lora:.2f}")

if __name__ == "__main__":
    main()