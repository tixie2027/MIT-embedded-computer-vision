# import all necessary libraries
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


USE_MPS = True

if USE_MPS and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend")
else:
    device = torch.device("cpu")
print("Device:", device)



loraize_encoder = False  # no reason really to do this unless we want to reduce memory footprint
loraize_decoder = True 

lora_config = {
    torch.nn.Conv2d: {
        'cls': LoRAConv2d,
        'config': {
            'alpha': 8,
            'rank': 4,
            'rank_for': 'channels',
            'delta_bias': False # True # TODO what does this do
        }
    },
    torch.nn.Linear: {
        'cls': LoRALinear,
        'config': {
            'rank': 4,
            'alpha': 2,
            'delta_bias':  False # True # TODO what does this do
        }
    },
    torch.nn.ConvTranspose2d: {
        'cls': LoRAConvTranspose2d,
        'config': {
            'alpha': 8,
            'rank': 4,
            'rank_for': 'channels',
            'delta_bias': False # True # TODO what does this do
        }
    }
}

def get_lora_model():
    # load two copies so we can lora-ize one
    # _model = CompressaiWrapper(compressai.zoo.bmshj2018_hyperprior(quality=1, metric="mse", pretrained=True)).model
    model = CompressaiWrapper(compressai.zoo.bmshj2018_hyperprior(quality=1, metric="mse", pretrained=True)).model

    print(model)
    
    # freeze decoder stuff that won't be transmitted
    print("Freezing:")
    for name, param in model.g_s.named_parameters():
        print(name)
        param.requires_grad = False
    for name, param in model.h_s.named_parameters():
        print(name)
        param.requires_grad = False

    print("Fine-tuning:")
    for i, module in enumerate(model.g_s.children()): 
        if type(module) in lora_config.keys():
            print("lora-izing", module)
            lora_cls = lora_config[type(module)]['cls']
            lora_params = lora_config[type(module)]['config']
            model.g_s[i] = lora_cls(module, lora_params) # automatically freezes old parameters
            model.g_s[i].enable_adapter()                # but we need to turn on the adapter path

    for i, module in enumerate(model.h_s.children()): 
        if type(module) in lora_config.keys():
            print("lora-izing", module)
            lora_cls = lora_config[type(module)]['cls']
            lora_params = lora_config[type(module)]['config']
            model.h_s[i] = lora_cls(module, lora_params) # automatically freezes old parameters
            model.h_s[i].enable_adapter()                # but we need to turn on the adapter path

    # del _model # get rid of the clone
    return model




