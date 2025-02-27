import copy
import random

import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageOps, ImageSequence
from PIL import Image, ImageDraw

from ..flashface.all_finetune.config import cfg
from ..flashface.all_finetune.utils import Compose, PadToSquare, seed_everything
from ..ldm.models.retinaface import retinaface
from ..ldm.ops.solvers import __all__ as solvers

padding_to_square = PadToSquare(224)

retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

retinaface = retinaface(pretrained=True, device='cuda').eval().requires_grad_(False)

def get_padding(width, height, max_dim=None):
    if max_dim is None:
        max_dim = max(width, height)
    
    h_padding = (max_dim - width) / 2
    v_padding = (max_dim - height) / 2
    
    l_pad = int(h_padding)
    t_pad = int(v_padding)
    r_pad = int(h_padding + 0.5)
    b_pad = int(v_padding + 0.5)
    
    return (l_pad, t_pad, r_pad, b_pad)

class FlashFaceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "reference_faces": ("PIL_IMAGE", {}),
                "latent": ("LATENT", {}),
                "vae": ("VAE", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "sampler": (['ddim', 'euler', 'euler_ancestral', 'dpm_2', 'dpm_2_ancestral',],),
                "steps": ("INT", {"default": 35}),
                "text_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "reference_feature_strength": ("FLOAT", {"default": 1.2, "min": 0.7, "max": 1.4, "step": 0.05}),
                "reference_guidance_strength": ("FLOAT", {"default": 3.2, "min": 1.8, "max": 4.0, "step": 0.1}),
                "step_to_launch_face_guidance": ("INT", {"default": 750, "min": 0, "max": 1000, "step": 50}),
                "auto_detect_face": ("BOOLEAN", {"default": False}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_bbox_x1": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_y1": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_x2": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_y2": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                # "height": ("INT", {"default": 768, "min": 8, "max": 16000}),
                # "width": ("INT", {"default": 768, "min": 8, "max": 16000}),
                # "num_samples": ("INT", {"default": 1}),
            },
            "optional": {
                "mask": ("MASK", {}),  # Make mask an optional input type
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")  # Return the model and the image
    FUNCTION = "generate"
    CATEGORY = "FlashFace"

    def generate(self, model, positive, negative, reference_faces, latent, vae, seed, sampler, steps, text_guidance_strength,
                 reference_feature_strength, reference_guidance_strength, step_to_launch_face_guidance, auto_detect_face,
                 denoise_strength, face_bbox_x1, face_bbox_y1, face_bbox_x2, face_bbox_y2, mask=None):

        # get number of samples, height and width from the latent image
        num_samples, _, height, width = latent["samples"].shape
        height = height * 8
        width = width * 8

        seed_everything(seed)

        print(f'detected {len(reference_faces)} faces')
        if len(reference_faces) == 0:
            raise Exception('No face detected in the reference images, please upload images with clear face')

        face_transforms = Compose(
            [T.ToTensor(),
             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        lambda_feat_before_ref_guidance = 0.85

        # If auto_detect_face is enabled, we need to decode the latent to get the source image
        if auto_detect_face:
            print("Using latent for automatic face detection")
            
            try:
                # Check if this is a ComfyUI VAE encoded latent or from FlashFace VAE
                with torch.no_grad():
                    # Use the FlashFace VAE for decoding (the one passed to this node)
                    latent_for_detection = latent["samples"].to('cuda')
                    
                    # Attempt to decode with FlashFace VAE
                    try:
                        # Only normalize empty latents
                        if "noise_mask" not in latent:
                            latent_for_detection_copy = latent_for_detection.clone()
                            # This is needed for empty latents
                            latent_for_detection_copy = latent_for_detection_copy.normal_()
                            decoded_images = vae.decode(latent_for_detection_copy / cfg.ae_scale)
                        else:
                            # For pre-encoded latents
                            decoded_images = vae.decode(latent_for_detection / cfg.ae_scale)
                            
                        # Convert to PIL for face detection
                        source_img = (decoded_images[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                        source_pil = Image.fromarray(source_img)
                        
                        # Apply RetinaFace detection
                        img_tensor = retinaface_transforms(source_pil).unsqueeze(0).to('cuda')
                        boxes, kpts = retinaface.detect(img_tensor, min_thr=0.6)
                        
                        if len(boxes[0]) > 0:
                            # Get the first detected face
                            scale = 640 / max(source_pil.size)
                            left, top, _, _ = get_padding(round(scale * source_pil.width),
                                                        round(scale * source_pil.height), 640)
                            
                            # Adjust bounding box coordinates
                            box = boxes[0][0].clone()
                            box[0] -= left
                            box[2] -= left
                            box[1] -= top
                            box[3] -= top
                            
                            box[:4] /= scale
                            
                            # Convert to normalized coordinates (0-1)
                            face_bbox_x1 = float(box[0] / source_pil.width)
                            face_bbox_y1 = float(box[1] / source_pil.height)
                            face_bbox_x2 = float(box[2] / source_pil.width)
                            face_bbox_y2 = float(box[3] / source_pil.height)
                            
                            print(f"Detected face at: [{face_bbox_x1:.2f}, {face_bbox_y1:.2f}, {face_bbox_x2:.2f}, {face_bbox_y2:.2f}]")
                        else:
                            print("No face detected in source image, using default bbox parameters")
                    except Exception as e:
                        print(f"Error decoding latent with FlashFace VAE: {e}")
                        print("Falling back to manual bbox parameters")
            except Exception as e:
                print(f"Error during face detection: {e}")
                print("Falling back to manual bbox parameters")

        # Process the face_bbox
        face_bbox = [face_bbox_x1, face_bbox_y1, face_bbox_x2, face_bbox_y2]
        H = height
        W = width
        if isinstance(face_bbox, str):
            face_bbox = eval(face_bbox)
        normalized_bbox = face_bbox
        face_bbox = [
            int(normalized_bbox[0] * W),
            int(normalized_bbox[1] * H),
            int(normalized_bbox[2] * W),
            int(normalized_bbox[3] * H)
        ]
        max_size = max(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])

        if mask is not None:
            mask_tensor = mask.float().cuda()
            if mask_tensor.ndim == 2:  # Ensure mask tensor has 3 dimensions
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = F.resize(mask_tensor, (H // 8, W // 8))
            empty_mask = mask_tensor.repeat(num_samples, 1, 1)
        else:
            empty_mask = torch.zeros((H, W))

            empty_mask[face_bbox[1]:face_bbox[1] + max_size,
            face_bbox[0]:face_bbox[0] + max_size] = 1

            empty_mask = empty_mask[::8, ::8].cuda()
            empty_mask = empty_mask[None].repeat(num_samples, 1, 1)

        padding_to_square = PadToSquare(224)
        pasted_ref_faces = []
        show_refs = []
        for ref_img in reference_faces:
            ref_img = ref_img.convert('RGB')
            ref_img = padding_to_square(ref_img)
            to_paste = ref_img

            to_paste = face_transforms(to_paste)
            pasted_ref_faces.append(to_paste)

        faces = torch.stack(pasted_ref_faces, dim=0).to('cuda')

        ref_z0 = cfg.ae_scale * torch.cat([
            vae.sample(u, deterministic=True)
            for u in faces.split(cfg.ae_batch_size)
        ])

        # Unpack model and diffusion
        model, diffusion = model
        model.share_cache['num_pairs'] = len(faces)
        model.share_cache['ref'] = ref_z0
        model.share_cache['similarity'] = torch.tensor(reference_feature_strength).cuda()
        model.share_cache['ori_similarity'] = torch.tensor(reference_feature_strength).cuda()
        model.share_cache['lamda_feat_before_ref_guidance'] = torch.tensor(lambda_feat_before_ref_guidance).cuda()
        model.share_cache['ref_context'] = negative.repeat(len(ref_z0), 1, 1)
        model.share_cache['masks'] = empty_mask
        model.share_cache['classifier'] = reference_guidance_strength
        model.share_cache['step_to_launch_face_guidance'] = step_to_launch_face_guidance
        diffusion.classifier = reference_guidance_strength

        progress = 0.0
        diffusion.progress = 0

        positive = positive[None].repeat(num_samples, 1, 1, 1).flatten(0, 1)
        positive = {'context': positive}

        negative = {
            'context': negative[None].repeat(num_samples, 1, 1, 1).flatten(0, 1)
        }

        latent_image = latent["samples"].to('cuda')
        has_noise_mask = "noise_mask" in latent
        
        # Apply denoise strength for img2img operations
        if has_noise_mask:
            # This is an encoded image latent
            print(f"Using encoded image latent with denoise strength: {denoise_strength}")
            # Calculate noise timestep based on denoise strength
            t_enc = int(steps * (1.0 - denoise_strength))
            print(f"Starting denoising from timestep {t_enc}/{steps}")
            
            if t_enc < steps:
                # Only add noise if denoising is not set to 0
                noise = torch.randn_like(latent_image)
                # Determine sigma based on timestep
                sigmas = diffusion.sigmas
                sigma = sigmas[t_enc]
                # Add scaled noise to the latent
                noised_latent = latent_image + noise * sigma
                # Set latent_image to the noised version
                latent_image = noised_latent
                # Adjust sampling steps based on denoise strength
                steps_to_run = steps - t_enc
            else:
                # Use the latent directly if denoise is 0
                steps_to_run = steps
            
            print(f"Will run for {steps_to_run} steps")
        else:
            # This is an empty latent - use normal noise
            print("Using empty latent with full noise")
            latent_image = latent_image.normal_()
            steps_to_run = steps
        
        # Check if model contains an image and blend it with the mask
        if mask is not None:
            mask_resized = F.resize(mask_tensor, latent_image.shape[-2:])
            latent_image = latent_image * (1 - mask_resized) + mask_resized * latent_image

        # sample
        with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
            z0 = diffusion.sample(solver=sampler,
                                  noise=latent_image,
                                  model=model,
                                  model_kwargs=[positive, negative],
                                  steps=steps_to_run,
                                  guide_scale=text_guidance_strength,
                                  guide_rescale=0.5,
                                  show_progress=True,
                                  discretization=cfg.discretization)

        imgs = vae.decode(z0 / cfg.ae_scale)

        # output
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(
            0, 255).astype(np.uint8)

        # convert to PIL image
        imgs_pil = [Image.fromarray(img) for img in imgs]

        torch_imgs = []
        for img in imgs_pil:
            img_tensor = F.to_tensor(img)
            # Ensure the data type is correct
            img_np = img_tensor.permute(1, 2, 0).unsqueeze(0)
            torch_imgs.append(img_np)
        torch_imgs = torch.cat(torch_imgs, dim=0)

        # Store the generated image in the model's share cache
        model.share_cache['generated_image'] = torch_imgs

        # Return the model and the image separately via share cache
        return model, torch_imgs
