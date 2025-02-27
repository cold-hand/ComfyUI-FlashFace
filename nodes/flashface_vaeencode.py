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
            # Encode the image using the VAE
            latent = vae.encode(x)
            
            # Debug print
            print(f"VAE encode returned type: {type(latent)}")
            
            # Handle different return types
            if isinstance(latent, tuple):
                # Extract the first element from the tuple (usually the actual latent)
                print(f"Tuple length: {len(latent)}")
                latent = latent[0]  # Most VAEs return the latent as the first element
                print(f"Using first element of tuple: {type(latent)}")
            elif not isinstance(latent, torch.Tensor):
                # This might happen if vae.encode returns a dictionary or other object
                print(f"WARNING: VAE encode returned {type(latent)}, trying to extract tensor")
                # Try to get the tensor from whatever was returned
                if hasattr(latent, 'sample'):
                    latent = latent.sample
                elif isinstance(latent, dict) and 'samples' in latent:
                    latent = latent['samples']
                else:
                    raise ValueError(f"Unable to extract tensor from VAE encode result: {type(latent)}")
            
            # Make sure we have a tensor at this point
            if not isinstance(latent, torch.Tensor):
                raise ValueError(f"Failed to convert VAE encode result to tensor: {type(latent)}")
                
            # Scale the latent
            scaled_latent = latent * float(cfg.ae_scale)
            
            # Format latent for compatibility
            latent_result = {"samples": scaled_latent}
            
            # Add a noise_mask field to indicate this isn't an empty latent
            latent_result["noise_mask"] = torch.ones((scaled_latent.shape[0], 1, scaled_latent.shape[2], scaled_latent.shape[3]), 
                                                    device=scaled_latent.device)
        
        return (latent_result,) 