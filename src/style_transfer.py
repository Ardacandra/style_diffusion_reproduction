import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import copy
import logging
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from src.helper import *
from src.style_removal import ddim_deterministic

def style_reconstruction_loss(I_ss: torch.Tensor, I_s: torch.Tensor) -> torch.Tensor:
    """
    Compute the style reconstruction loss between reconstructed and reference style images.

    Args:
        I_ss (torch.Tensor): The reconstructed style image from the diffusion model.
        I_s (torch.Tensor): The original style reference image.

    Returns:
        loss (torch.Tensor): A scalar tensor representing the loss.
    """
    return F.mse_loss(I_ss, I_s)

def style_disentanglement_loss(I_ci: torch.Tensor, I_cs: torch.Tensor, I_ss: torch.Tensor, I_s: torch.Tensor) -> torch.Tensor:
    """
    Compute the style disentanglement loss.

    Args:
        I_ci (torch.Tensor): Original content image latent or embedding.
        I_cs (torch.Tensor): Style-modified content image (decoded from diffusion model).
        I_ss (torch.Tensor): Reconstructed style image from the style latent.
        I_s  (torch.Tensor): Original style reference image.

    Returns:
        loss (torch.Tensor): A scalar tensor representing the loss.
    """
    return torch.tensor(0.0, requires_grad=True, device=I_ci.device)

def style_diffusion_fine_tuning(
    style_tensor: torch.Tensor,
    style_latent: torch.Tensor,
    content_latents: list,
    model: nn.Module,
    diffusion,
    s_rev: int,
    k: int,
    k_s: int,
    lr: float,
    device: str,
    logger=None,
):
    """
    Fine-tune a diffusion model using alternating style reconstruction and style disentanglement objectives.

    This function implements a simplified training loop derived from a research pseudocode.
    It alternates between optimizing for:
        1. Style reconstruction (reproducing the style image from its latent)
        2. Style disentanglement (ensuring style transfer doesn't override content structure)

    Args:
        style_tensor (torch.Tensor): The reference style image tensor.
        style_latent (torch.Tensor): The latent representation of the style image.
        content_latents (list[torch.Tensor]): List of latent representations of content images.
        model (nn.Module): The diffusion model to fine-tune.
        diffusion: A diffusion process object that provides 'alphas_cumprod'.
        s_rev (int): Number of reverse diffusion steps.
        k (int): Number of fine-tuning outer iterations.
        k_s (int): Number of inner steps for style reconstruction loss optimization.
        lr (float): Learning rate for fine-tuning.
        device (str): Device identifier, e.g., "cuda" or "cpu".
        logger (logging.logger): optional logger
    Returns:
        model_finetuned (nn.Module): fine-tuned diffusion model
    """
    
    if logger is not None:
        logger.info(f"Starting style transfer fine-tuning...")

    #initialize fine-tuned model
    model_finetuned = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model_finetuned.parameters(), lr=lr)

    #training loop
    for iter in range(k):
        if logger is not None:
            logger.info(f"Starting fine-tuning iteration {iter+1}...")

        #optimize the style reconstruction loss
        I_s = style_tensor.clone().to(device)
        for i in range(k_s):
            if logger is not None:
                logger.info(f"Starting style reconstruction iteration {i+1}...")

            x_t = style_latent.clone().to(device)
            for s in reversed(range(1, s_rev)):
                if logger is not None:
                    logger.info(f"DDIM step: {s} -> {s-1}")

                t = torch.full((x_t.size(0),), s, device=device, dtype=torch.long)

                # Use DDIM deterministic reverse diffusion
                ddim_timesteps_backward = np.linspace(s-1, s, 2, dtype=int)
                ddim_timesteps_backward = ddim_timesteps_backward[::-1]

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=ddim_timesteps_backward,
                    device=device,
                    requires_grad=True,
                )

                #style reconstruction loss evaluation
                I_ss = x_t_prev.clone().to(device)
                loss_sr = style_reconstruction_loss(I_ss, I_s)
                optimizer.zero_grad()
                loss_sr.backward()
                optimizer.step()

                x_t = x_t_prev.detach()
        
        #optimize the style disentanglement loss
        for i in range(len(content_latents)):
            if logger is not None:
                logger.info(f"Starting style disentanglement for sample number {i+1}...")

            x_t = content_latents[i].clone().to(device)
            for s in reversed(range(1, s_rev)):
                if logger is not None:
                    logger.info(f"DDIM step: {s} -> {s-1}")
            
                t = torch.full((x_t.size(0),), s, device=device, dtype=torch.long)

                # Use DDIM deterministic reverse diffusion
                ddim_timesteps_backward = np.linspace(s-1, s, 2, dtype=int)
                ddim_timesteps_backward = ddim_timesteps_backward[::-1]

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=ddim_timesteps_backward,
                    device=device,
                    requires_grad=True,
                )
                
                #style disentanglement loss evaluation
                I_ci = content_latents[i].clone().to(device)
                I_cs = x_t_prev.clone().to(device)
                loss_sd = style_disentanglement_loss(I_ci, I_cs, I_ss, I_s)
                optimizer.zero_grad()
                loss_sd.backward()
                optimizer.step()

                x_t = x_t_prev.detach()
    
    if logger is not None:
        logger.info("Style transfer fine-tuning completed.")
    return model_finetuned

if __name__ == "__main__":
    #Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    IMAGE_SIZE = 256
    S_FOR = 40
    S_REV = 30

    K = 5
    K_S = 50
    LR = 0.00005
    N_CONTENT_SAMPLE = 5

    CONTENT_LATENTS_PATH = "output/test_run/content_latents/"
    STYLE_ORIGINAL_PATH = "data/style/van_gogh/000.jpg"
    STYLE_LATENT_PATH = "output/test_run/style_latents/style.pt"

    OUTPUT_DIR = "output/"
    OUTPUT_PREFIX = "style_transfer__"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR, f"style_transfer.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info(f"Starting example usage of style transfer...")

    # #load model
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
    model.to(DEVICE)

    #get sample style original and latent
    original_style = Image.open(STYLE_ORIGINAL_PATH)
    original_style_tensor = prepare_image_as_tensor(original_style, image_size=IMAGE_SIZE, device=DEVICE)

    style_latent = torch.load(STYLE_LATENT_PATH, map_location=DEVICE, weights_only=True)
    
    #get sample content latents
    content_latent_files = [f for f in os.listdir(CONTENT_LATENTS_PATH) if f.lower().endswith(('.pt'))]
    sample_content_latent_files = content_latent_files[:N_CONTENT_SAMPLE]

    content_latents = []
    for file in sample_content_latent_files:
        content_latents.append(
            torch.load(os.path.join(CONTENT_LATENTS_PATH, file), map_location=DEVICE, weights_only=True)
        )
    
    logger.info(f"original style tensor shape: {original_style_tensor.shape}")
    logger.info(f"style latent shape: {style_latent.shape}")
    logger.info(f"content latents sample count: {len(content_latents)}")
    logger.info(f"content latent shape: {content_latents[0].shape}")

    #apply style diffusion fine-tuning
    model_finetuned = style_diffusion_fine_tuning(
        original_style_tensor,
        style_latent,
        content_latents,
        model,
        diffusion,
        S_REV,
        K,
        K_S,
        LR,
        DEVICE,
        logger=logger,
    )
    torch.save(model_finetuned.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}finetuned_style_model.pt"))
