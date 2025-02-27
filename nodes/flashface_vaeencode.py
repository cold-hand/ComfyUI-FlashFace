import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from ..flashface.all_finetune.config import cfg
from ..ldm import sd_v1_vae

class FlashFaceVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "pixels": ("IMAGE", ),
                    "vae": ("VAE", ),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "FlashFace"

    def encode(self, pixels, vae):
        # FlashFace expects a different format than standard ComfyUI
        # So we need to properly transform the image tensor
        
        # Make sure input is in the correct format
        if pixels.ndim == 4:
            pixels = pixels.squeeze(0)
        
        # Normalize pixels to [-1, 1]
        x = pixels.permute(2, 0, 1).unsqueeze(0)
        x = x * 2.0 - 1.0
        
        # Move to GPU and encode
        x = x.to(device="cuda")
        
        # Use FlashFace VAE to encode
        with torch.no_grad():
            latent = vae.encode(x)
        
        # Format latent for compatibility
        latent_result = {"samples": latent * cfg.ae_scale}
        
        # Add a noise_mask field to indicate this isn't an empty latent
        latent_result["noise_mask"] = torch.ones((1, 1, latent.shape[2], latent.shape[3]), 
                                                device=latent.device)
        
        return (latent_result,) 