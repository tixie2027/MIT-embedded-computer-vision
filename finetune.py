from Iwildcam_Pretrain import Autoencoder, IWildCamDataset, CompressaiWrapper
from lora_modules import LoRAConv2d, LoRALinear, LoRAConvTranspose2d


import torch.nn as nn

def replace_with_lora(module, lora_config):
    for name, child in module.named_children():
        # Replace Conv2d
        if isinstance(child, nn.Conv2d):
            new = LoRAConv2d(base_module=child, lora_config=lora_config)
            setattr(module, name, new)
        else:
            replace_with_lora(child, lora_config)


lora_config = {
    "alpha": 16,
    "rank": 4,
    "rank_for": "kernel",   # or "channels"
    "delta_bias": True,
    "beta": 0.5,
    "precision": None,      # if you want quantization
}

# Load model and apply LoRA
wrapper = CompressaiWrapper()
model = wrapper.model
print(model)
replace_with_lora(model, lora_config)

print(model)


