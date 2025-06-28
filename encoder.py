import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from capture_image import capture_images
#from encode_and_save import encode_and_save
from torchvision import transforms as T
import numpy as np

# -- your existing constants and Encoder class here --
HEIGHT = 480
WIDTH = 640

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(num_input_channels, c_hid, 3, padding=1, stride=2),  #  (HxW)->(H/2 x W/2)
            act_fn(),
            nn.Conv2d(c_hid, c_hid, 3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, 3, padding=1, stride=2),              # -> (H/4 x W/4)
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, 3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, 3, padding=1, stride=2),            # -> (H/8 x W/8)
            act_fn(),
            nn.Flatten(),
            nn.Linear(2*c_hid * (HEIGHT * WIDTH) // (4 ** 3), latent_dim),
            # if you need to dequantize:
            torch.quantization.DeQuantStub(),
        )
    
    def forward(self, x):
        return self.net(x)

# -- new helper function to batch‐encode and save to TXT --

def save_latents_hex(latents: torch.Tensor,
                     filename: str = "embeddings.txt"):
    """
    Save a batch of latent vectors as FP16 bit‐patterns in hex,
    4 rows per vector (16 entries each), with 'XXXX' appended
    to the last row of each vector.

    Output for one vector (64 dims) will be 4 lines like:

      a1b2,3c4d,...,5e6f        <-- 16 hex codes
      ...
      7a8b,9c0d,...,eef0        <-- 16 hex codes
      1234,5678,...,abcd,XXXX   <-- 16 hex codes + marker
    """
    # 1) cast to float16, detach, move to CPU
    fp16 = latents.half().detach().cpu().numpy()       # shape (N, D), dtype=float16

    # 2) reinterpret floats as uint16 bit‐patterns
    bits = fp16.view(np.uint16)                       # same shape, dtype=uint16

    N, D = bits.shape
    chunk_size = 16
    assert D % chunk_size == 0, "Dim must be divisible by 16"
    num_chunks = D // chunk_size

    with open(filename, "w") as f:
        for vec in bits:
            for i in range(num_chunks):
                chunk = vec[i*chunk_size:(i+1)*chunk_size]
                # 16 hex codes
                hex_row = ",".join(f"{x:04x}" for x in chunk)
                # append marker on the last of the 4 rows
                if i == num_chunks - 1:
                    hex_row += ",XXXX"
                f.write(hex_row + "\n")

    print(f"Saved {N} vectors (each {D}-dim) as hex to '{filename}'")




def main():
    # capture batch
    images = capture_images()
    # load or instantiate your encoder
    encoder = Encoder(num_input_channels=3, base_channel_size=64, latent_dim=64) 
    latents = encoder(images)
    print(latents)
    print(latents.type)
    save_latents_hex(latents, "embeddings.txt")

if __name__ == "__main__":
    main()