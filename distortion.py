import math

import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim
from torch.nn import functional as F
from compressai.registry import register_criterion


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all", lambda_gain=2.0):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.lambda_gain = lambda_gain
        self.return_type = return_type

    
    
    def forward(self, output, target, masks, w_animal=1.0, w_background=1.0):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values())
        out["bpp_loss"] = bpp_loss

        '''
        bpp_loss = 0.0
        for likelihoods in output["likelihoods"].values():
                mask_resized = F.interpolate(masks, size=likelihoods.shape[2:], mode='nearest') # resize
            
                # broadcast to all channels
                if mask_resized.shape[1] == 1:
                    mask_resized = mask_resized.expand(-1, likelihoods.size(1), -1, -1)

                bits = torch.log(likelihoods)
            
            
                bits_animal = (bits * mask_resized).sum()
                bits_background = (bits * (1 - mask_resized)).sum()
                weighted_bits = bits_animal * w_animal + bits_background * w_background

                #pixels_animal = mask_resized.sum()
               # pixels_background = (1 - mask_resized).sum()
                #weighted_pixels = w_animal * pixels_animal + w_background * pixels_background

                bpp_loss += weighted_bits / (-math.log(2) * num_pixels)'''

        out["bpp_loss"] = bpp_loss
        

        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            animal_loss = F.mse_loss(output['x_hat'] * masks, target * masks, reduction='none')
            background_loss = F.mse_loss(output['x_hat'] * (1 - masks), target * (1 - masks), reduction='none')
            recon_loss = animal_loss * w_animal + background_loss * w_background

            out['mse_loss'] = (recon_loss).sum()/(w_animal * masks.sum() + w_background * (1 - masks).sum())
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
    
    '''
    def forward(self, output, target, masks):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # --- BPP Loss ---
        bpp_loss = sum(
            torch.log(likelihoods + 1e-9).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values())

        # --- Dynamic lambda map ---
        lambda_map = self.lmbda * (1 + self.lambda_gain * masks)

        # --- Distortion ---
        if isinstance(self.metric, nn.MSELoss):
            mse_map = F.mse_loss(output["x_hat"], target, reduction="none").mean(1, keepdim=True)
            weighted_distortion = (lambda_map * mse_map).sum() / (lambda_map.sum() + 1e-9)
            distortion_loss = 255**2 * weighted_distortion
        else:  # ms-ssim
            ms_ssim_loss = self.metric(output["x_hat"], target, data_range=1)
            distortion_loss = 1 - ms_ssim_loss

        # --- Total loss ---
        total_loss = distortion_loss + bpp_loss

        out = {
            "loss": total_loss,
            "bpp_loss": bpp_loss.item(),
            "mse_loss": distortion_loss.item()
        }
        return out if self.return_type == "all" else out[self.return_type]'''
    



