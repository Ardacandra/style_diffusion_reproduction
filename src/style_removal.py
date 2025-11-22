import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from models.improved_ddpm.script_util import i_DDPM

from src.helper import rgb_to_luma_601, prepare_image_as_tensor
from src.diffusion import ddim_deterministic, get_linear_alphas_cumprod

if __name__ == "__main__":
    #Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    CHECKPOINT_PATH = "models/checkpoints/512x512_diffusion.pt"
    IMAGE_SIZE = 512

    DIFFUSION_NUM_TIMESTEPS = 1000
    DIFFUSION_BETA_START = 0.0001
    DIFFUSION_BETA_END = 0.02

    T_REMOV = 603
    S_FOR = 40
    S_REV = 6

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
    model = i_DDPM("IMAGENET", IMAGE_SIZE)
    init_ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(init_ckpt)
    model.to(DEVICE)

    #get alphas_cumprod
    alphas_cumprod = get_linear_alphas_cumprod(
        timesteps=DIFFUSION_NUM_TIMESTEPS,
        beta_start=DIFFUSION_BETA_START,
        beta_end=DIFFUSION_BETA_END
    )

    #convert luma image into tensor
    x0 = prepare_image_as_tensor(Image.fromarray(image_luma), image_size=IMAGE_SIZE, device=DEVICE)

    #forward diffusion (DDIM)
    ddim_timesteps_forward = np.linspace(0, T_REMOV, S_FOR, dtype=int)
    x_t = ddim_deterministic(x0, model, alphas_cumprod, ddim_timesteps_forward, DEVICE)

    image_noised = x_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_noised = ((image_noised + 1) / 2).clip(0, 1)  # scale back to [0,1]
    plt.imshow(image_noised)
    plt.title("Noised Image After Forward Diffusion")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "noised_image.png"), bbox_inches='tight', dpi=300)

    #reverse diffusion (DDIM)
    ddim_timesteps_backward = np.linspace(0, T_REMOV, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    assert ddim_timesteps_backward[-1]==0
    x0_est = ddim_deterministic(x_t, model, alphas_cumprod, ddim_timesteps_backward, device=DEVICE)

    image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_recon = ((image_recon + 1) / 2).clip(0, 1)  # scale back to [0,1]
    plt.imshow(image_recon[..., 0], cmap='gray')
    plt.title("Reconstructed Image After Style Removal")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "recon_image.png"), bbox_inches='tight', dpi=300)