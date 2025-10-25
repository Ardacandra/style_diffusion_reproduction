import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from src.helper import *

def ddim_deterministic(
    x_start,
    model,
    diffusion,
    ddim_timesteps,
    device,
    logger=None,
    requires_grad=False,
):
    """
    DDIM deterministic diffusion (forward or reverse).

    Args:
        x_start: starting tensor (x0 for forward, x_t for reverse)
        model: UNet diffusion model
        diffusion: diffusion process (contains alphas_cumprod)
        ddim_timesteps: list of timesteps
        device: torch device
        logger: optional logger
        requires_grad: if False, disables gradient tracking (default)
    Returns:
        x_out: resulting tensor (xt for forward, x0 for reverse)
    """
    
    grad_context = torch.enable_grad if requires_grad else torch.no_grad
    with grad_context():
        if logger is not None:
            logger.info(f"Starting DDIM diffusion with {len(ddim_timesteps)} steps.")

        x = x_start.clone()
        B = x.shape[0]
        alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device)

        for i in range(len(ddim_timesteps) - 1):
            t = int(ddim_timesteps[i])
            t_next = int(ddim_timesteps[i + 1])

            if logger is not None:
                logger.info(f"DDIM step {i+1}/{len(ddim_timesteps)-1}: {t} -> {t_next}")

            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

            # Predict noise
            out = model(x, t_tensor)
            # If model predicts mean and variance, only take the mean
            eps = out[:, :3] if out.shape[1] == 6 else out

            # Fetch alpha bars
            alpha_bar_t = alphas_cumprod[t_tensor].view(B, 1, 1, 1)
            alpha_bar_next = alphas_cumprod[t_next].view(B, 1, 1, 1)

            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_ab_next = torch.sqrt(alpha_bar_next)
            sqrt_one_minus_ab_next = torch.sqrt(1.0 - alpha_bar_next)

            # Compute predicted x0
            x0_pred = (x - sqrt_one_minus_ab_t * eps) / sqrt_ab_t

            # Deterministic DDIM update
            x = sqrt_ab_next * x0_pred + sqrt_one_minus_ab_next * eps

        if logger is not None:
            logger.info(f"DDIM diffusion completed.")
    return x

if __name__ == "__main__":
    #Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    IMAGE_SIZE = 256
    S_FOR = 40
    S_REV = 30
    IMAGE_PATH = "data/style/van_gogh/"
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
        'diffusion_steps': S_FOR,
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

    t = torch.tensor([diffusion.num_timesteps - 1]).to(DEVICE)
    ddim_timesteps_forward = np.linspace(0, diffusion.num_timesteps - 1, S_FOR, dtype=int)
    x_t = ddim_deterministic(x0, model, diffusion, ddim_timesteps_forward, DEVICE)

    image_noised = x_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_noised = ((image_noised + 1) / 2).clip(0, 1)  # scale back to [0,1]
    plt.imshow(image_noised)
    plt.title("Noised Image After Forward Diffusion")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "noised_image.png"), bbox_inches='tight', dpi=300)

    #reverse diffusion with fewer steps (DDIM)
    ddim_timesteps_backward = np.linspace(0, S_FOR-1, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    assert ddim_timesteps_backward[-1]==0
    x0_est = ddim_deterministic(x_t, model, diffusion, ddim_timesteps_backward, device=DEVICE)

    image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_recon = ((image_recon + 1) / 2).clip(0, 1)  # scale back to [0,1]
    plt.imshow(image_recon[..., 0], cmap='gray')
    plt.title("Reconstructed Image After Style Removal")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "recon_image.png"), bbox_inches='tight', dpi=300)