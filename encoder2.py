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


# -- your existing constants and Encoder class here --
HEIGHT = 448
WIDTH = 640


import pickle

def save_out_raw(out, filename="compressed_output.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(out, f)
    print(f"[✓] Saved `out` exactly as-is to {filename}")






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

    # Step 3: Load pretrained CompressAI model
    model = bmshj2018_hyperprior(quality=3, pretrained=True).eval()

    # Step 4: Compress image
    out = model.compress(images)  # Handles everything internally
    print(out)
 
    #torch.save(out, "embeddings.pt")
    #print("[✓] Saved full `out` dictionary to embeddings.pt")
    import struct

    def save_out_exactly(out, filename="embeddings.txt"):
        with open(filename, "w") as f:
            f.write(f"# shape: {tuple(out['shape'])}\n")
            f.write("# hex-encoded float32 values (4 bytes per entry)\n\n")

            for i, row in enumerate(out["strings"]):
                f.write(f"# row {i}\n")
                for j, barray in enumerate(row):
                    padded = barray + b'\x00' * ((4 - len(barray) % 4) % 4)
                    floats = []
                    for k in range(0, len(padded), 4):
                        chunk = padded[k:k+4]
                        if len(chunk) < 4:
                            continue
                        # Unpack to float32, then re-pack to 4-byte hex
                        float_bytes = struct.pack("f", struct.unpack("f", chunk)[0])
                        floats.append(f"0x{float_bytes.hex()}")
                    f.write(" ".join(floats) + "\n")
                f.write("\n")
        print(f"[✓] Saved float32 hex to {filename}")

    save_out_exactly(out, "embeddings.txt")

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