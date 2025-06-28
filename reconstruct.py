import torch
import numpy as np
import matplotlib.pyplot as plt
from decoder import Decoder

HEIGHT = 480
WIDTH = 640
LATENT_DIM = 64
NUM_INPUT_CHANNELS = 3
BASE_CHANNEL_SIZE = 32

# Initialize decoder
decoder = Decoder(NUM_INPUT_CHANNELS, BASE_CHANNEL_SIZE, LATENT_DIM)

ckpt = torch.load("8-epoch.04-val_loss.0.01 (1).ckpt", map_location="cpu")
state_dict = ckpt['state_dict']  # May vary depending on how you saved

# Filter out only decoder keys
decoder_state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}

# Load into your Decoder
decoder.load_state_dict(decoder_state_dict)

decoder.eval()

# Read 1 image worth of hex from embeddings.txt
with open("embeddings.txt", "r") as f:
    lines = [line.strip() for line in f if "XXXX" not in line]
    image_lines = lines[0:4]  # 4 lines = 1 image

# Convert hex to float16
hex_values = ",".join(image_lines).split(",")
float_array = np.array([np.frombuffer(bytes.fromhex(h), dtype=np.float16)[0] for h in hex_values], dtype=np.float32)
latent_tensor = torch.tensor(float_array).unsqueeze(0)  # (1, 128)

# Decode
with torch.no_grad():
    output = decoder(latent_tensor).squeeze().permute(1, 2, 0).numpy()
output = np.clip(output, 0, 1)

# Display
plt.imshow(output)
plt.title("Decoded Image 1")
plt.axis('off')
plt.show()
