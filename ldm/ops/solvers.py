import torch
from tqdm.auto import trange

from comfy.k_diffusion.sampling import sample_euler as comfy_sample_euler
from comfy.k_diffusion.sampling import sample_euler_ancestral as comfy_sample_euler_ancestral
from comfy.k_diffusion.sampling import sample_dpm_2 as comfy_sample_dpm_2
from comfy.k_diffusion.sampling import sample_dpm_2_ancestral as comfy_sample_dpm_2_ancestral
from comfy.k_diffusion.sampling import sample_dpm_fast as comfy_sample_dpm_fast

__all__ = [
    'sample_ddim',
    'sample_euler',
    'sample_euler_ancestral',
    'sample_dpm_2',
    'sample_dpm_2_ancestral',
    'sample_dpm_fast'
]


@torch.no_grad()
def sample_ddim(noise, model, sigmas, eta=0., show_progress=True):
    """DDIM solver steps."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        noise_factor = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2 *
                              (1 - (1 - sigmas[i] ** 2) /
                               (1 - sigmas[i + 1] ** 2)))
        d = (x - (1 - sigmas[i] ** 2) ** 0.5 * denoised) / sigmas[i]
        x = (1 - sigmas[i + 1] ** 2) ** 0.5 * denoised + \
            (sigmas[i + 1] ** 2 - noise_factor ** 2) ** 0.5 * d
        if sigmas[i + 1] > 0:
            x += noise_factor * torch.randn_like(x)
    return x


@torch.no_grad()
def sample_euler(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_euler(model, noise, sigmas, s_churn=0.75)

    return noise


@torch.no_grad()
def sample_euler_ancestral(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_euler_ancestral(model, noise, sigmas, s_noise=.97, eta=1.8)

    return noise


@torch.no_grad()
def sample_dpm_2(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_dpm_2(model, noise, sigmas, s_churn=0.75)

    return noise


@torch.no_grad()
def sample_dpm_2_ancestral(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_dpm_2_ancestral(model, noise, sigmas, s_noise=.97, eta=1.8)

    return noise

@torch.no_grad()
def sample_dpm_fast(noise, model, sigmas, show_progress=True):
    sigma_min = sigmas[-1]
    if sigma_min == 0:
        sigma_min = sigmas[-2]
    total_steps = len(sigmas) - 1
    noise = comfy_sample_dpm_fast(model, noise, sigma_min, sigmas[0],total_steps, s_noise=.97, eta=1.8)

    return noise