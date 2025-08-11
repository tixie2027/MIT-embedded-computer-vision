import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
torch.set_anomaly_enabled(True)
IMAGE_DIR = 'preview'
from distortion import RateDistortionLoss
from torchvision.transforms import GaussianBlur
from get_lora_model import get_lora_model
from torchvision.transforms import ToPILImage

class SimpleImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = [os.path.join(image_folder, f)
                            for f in os.listdir(image_folder)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_drawn_box_mask(image_tensor, color=[1.0, 1.0, 0.0], tolerance=0.2):

    """
    Args:
        image_tensor: [3, H, W], pixel values in [0,1]
        color: RGB color of box (default: yellow)
        tolerance: how exact the color match must be
    Returns:
        mask: [1, H, W], 1 inside box, 0 outside
    """
    c, h, w = image_tensor.shape
    img = image_tensor.permute(1, 2, 0)  # [H, W, 3]

    # find all pixels matching the box color
    color_tensor = torch.tensor(color, dtype=img.dtype, device=img.device)
    color_diff = torch.abs(img - color_tensor)
    match = (color_diff < tolerance).all(dim=2)  

    # Step 2: find bounding rectangle of all matched pixels
    y_indices, x_indices = torch.where(match)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # No box found — return zero mask
        return torch.zeros((1, h, w), dtype=torch.float32, device=img.device)
    x1, x2 = x_indices.min().item(), x_indices.max().item()
    y1, y2 = y_indices.min().item(), y_indices.max().item()

    # Step 3: build mask
    mask = torch.zeros((1, h, w), dtype=torch.float32, device=img.device)
    mask[:, y1:y2+1, x1:x2+1] = 1.0
    return mask

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            old_weight = child.weight
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode
            )

            # Initialize new conv weights with old weights (first 3 channels only)
            with torch.no_grad():
                new_conv.weight[:, :old_weight.shape[1]] = old_weight
                if child.bias is not None:
                    new_conv.bias = child.bias

            setattr(module, name, new_conv)
            print(f"✅ Patched {name} to accept {new_in_channels} input channels.")
            break  # Only patch the first conv
        else:
            patch_first_conv2d_input_channels(child, new_in_channels)

def train(model, dataloader, optimizer, device, epochs=5, gain=5):

    model.train()
    model.to(device)
    to_pil = ToPILImage()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            batch = batch.to(device)
            masks = torch.stack([get_drawn_box_mask(img) for img in batch])  # [B,1,H,W]

            # create this once, outside the loop
            blur = GaussianBlur(kernel_size=15, sigma=2)
            blurred = blur(batch)  # takes a [B, C, H, W] Tensor and blurs each image

            # composite as before
            mask_fg   = masks.expand_as(batch)
            batch_pre = batch * mask_fg + blurred * (1 - mask_fg)

            if batch_idx == 0 and epoch == 0:
                # original
                orig_pil   = to_pil(batch[0].cpu())
                # blurred‐background composite
                blur_pil   = to_pil(batch_pre[0].cpu())
                os.makedirs("preview_outputs", exist_ok=True)
                orig_pil.save("preview_outputs/original_ref.png")
                blur_pil.save("preview_outputs/blurred_ref.png")
                print("Saved preview_outputs/original_ref.png and blurred_ref.png")

            # 4) feed *batch_pre* into your model instead of batch
            result = model.forward(batch_pre)
    
            criterion = RateDistortionLoss(lmbda=0.0018, metric="mse")
            loss_out = criterion(result, batch_pre, masks, w_animal=1, w_background=0.001)

            loss = loss_out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(dataloader):.6f}")
        print(loss_out["bpp_loss"], loss_out["mse_loss"])


# Main
if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = get_lora_model()

    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    dataset = SimpleImageDataset(IMAGE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for name, param in model.named_parameters():
        print(f"Parameter {name}: requires_grad={param.requires_grad}")

    # Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params,  lr=1e-4)
    train(model, dataloader, optimizer, device, epochs=10)

    # Save final weights
    torch.save(model.state_dict(), "lora_encoder_finetuned.pth")
    print("✅ Finished training and saved to lora_encoder_finetuned.pth")