from Iwildcam_Pretrain import CompressaiWrapper
import torch, torch.nn as nn
from lora_modules import LoRAConv2d

loraize_encoder = True  # no reason really to do this unless we want to reduce memory footprint
loraize_decoder = False 

lora_config = {
    torch.nn.Conv2d: {
        'cls': LoRAConv2d,
        'config': {
            'alpha': 16,
            'rank': 8,
            'rank_for': 'channels',
            'delta_bias': False,}}}


def get_lora_model():
    # 1) load the pre-trained model
    model = CompressaiWrapper().model

    # 2) freeze absolutely everything
    for p in model.parameters():
        p.requires_grad = False

    # 3) LoRA-ize only if desired
    if loraize_encoder:
        # analysis (encoder) transform
        for i, module in enumerate(model.g_a.children()):
            if type(module) in lora_config:
                lora_cls    = lora_config[type(module)]['cls']
                lora_params = lora_config[type(module)]['config']
                wrapped     = lora_cls(module, lora_params)
                wrapped.enable_adapter()
                model.g_a[i] = wrapped

        # hyper-analysis (hyper-encoder) transform
        for i, module in enumerate(model.h_a.children()):
            if type(module) in lora_config:
                lora_cls    = lora_config[type(module)]['cls']
                lora_params = lora_config[type(module)]['config']
                wrapped     = lora_cls(module, lora_params)
                wrapped.enable_adapter()
                model.h_a[i] = wrapped

    return model

model = get_lora_model()
print(model)