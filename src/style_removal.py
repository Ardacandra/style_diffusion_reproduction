import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from src.helper import *

def make_ddim_timesteps(num_timesteps_total, T_remov):
    """
    Make evenly spaced descending list of timesteps for DDIM reverse process.
    """
    assert T_remov>=1 and T_remov<=num_timesteps_total
    timesteps = np.linspace(0, num_timesteps_total-1, T_remov, dtype=int)
    timesteps = timesteps[::-1] #reverse for descending order
    return timesteps

@torch.no_grad()
def ddim_reverse_deterministic(x_t, model, diffusion, ddim_timesteps, device):
    """
    Perform DDIM deterministic reverse diffusion from x_t to estimate x0.
    Args:
        x_t: noised input tensor at timestep t with shape [B, C, H, W]
        model: UNet diffusion model that predicts noise for input (x, t)
        diffusion: diffusion process object
        ddim_timesteps: list/array of timesteps for DDIM reverse process
        device: torch device
    
    Returns:
        x0_est: estimated clean image tensor with shape [B, C, H, W]
    """
    x = x_t
    B = x.shape[0]

    #retrieve alphas from diffusion process
    alphas_cumprod = diffusion.alphas_cumprod

    for i in range(len(ddim_timesteps)-1):
        t = int(ddim_timesteps[i])
        t_prev = int(ddim_timesteps[i+1])
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        t_prev_tensor = torch.full((B,), t_prev, dtype=torch.long, device=device)

        #predict noise eps with pre-trained model
        out = model(x, t_tensor)
        # If model predicts both mean and variance
        if out.shape[1] == 6:
            eps, logvar = torch.split(out, 3, dim=1)
        else:
            eps = out
            
        #fetch alpha bars
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod[t_prev]

        #compute x0_est
        x0_pred = (x - np.sqrt(1.0 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)

        #deterministic DDIM update (eta = 0)
        x = np.sqrt(alpha_bar_prev) * x0_pred + np.sqrt(1.0 - alpha_bar_prev) * eps

    # Final step: move all the way to t=0 estimation if last timestep isn't 0
    last_t = int(ddim_timesteps[-1])
    if last_t != 0:
        t_tensor = torch.full((B,), last_t, dtype=torch.long, device=device)
        out = model(x, t_tensor)
        # If model predicts both mean and variance
        if out.shape[1] == 6:
            eps, logvar = torch.split(out, 3, dim=1)
        else:
            eps = out

        alpha_bar_t = alphas_cumprod[last_t]
        x0_pred = (x - np.sqrt(1.0 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)
        x = x0_pred  # move to estimated x0

    return x

if __name__ == "__main__":
    #Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    IMAGE_SIZE = 256
    T_DIFFUSION = 1000
    T_REMOV = 100 #Larger T_remov → stronger style removal (more style details removed). 
    IMAGE_PATH = "data/content/"
    OUTPUT_DIR = "output/"
    OUTPUT_PREFIX = "style_removal__"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #get sample image
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(('.jpg', '.jpeg'))]
    image = Image.open(os.path.join(IMAGE_PATH, image_files[0])).convert('RGB')
    plt.imshow(image)
    plt.title("Original Image")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "original.png"), bbox_inches='tight', dpi=300)

    #apply color removal
    image_luma = rgb_to_luma_601(image)
    plt.imshow(image_luma, cmap='gray')
    plt.title("Image After Color Removal")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "color_removed.png"), bbox_inches='tight', dpi=300)

    #forward diffusion ODE/inversion to obtain latents
    options = model_and_diffusion_defaults()
    options.update({
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': T_DIFFUSION,
        'image_size': IMAGE_SIZE,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': False,
        'use_scale_shift_norm': True,
    })

    model, diffusion = create_model_and_diffusion(**options)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)

    #convert luma image into tensor
    x0 = prepare_image_as_tensor(Image.fromarray(image_luma), image_size=IMAGE_SIZE, device=DEVICE)

    #forward diffusion
    t = torch.tensor([diffusion.num_timesteps - 1]).to(DEVICE)
    x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))

    image_noised = x_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_noised = ((image_noised + 1) / 2).clip(0, 1)  # scale back to [0,1
    plt.imshow(image_noised)
    plt.title("Noised Image After Forward Diffusion")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "noised_image.png"), bbox_inches='tight', dpi=300)

    #reverse diffusion with fewer steps (DDIM)
    ddim_timesteps = make_ddim_timesteps(T_DIFFUSION, T_REMOV)
    x0_est = ddim_reverse_deterministic(x_t, model, diffusion, ddim_timesteps, device=DEVICE)

    image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_recon = ((image_recon + 1) / 2).clip(0, 1)  # scale back to [0,1
    plt.imshow(image_recon)
    plt.title("Reconstructed Image After Style Removal")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "recon_image.png"), bbox_inches='tight', dpi=300)