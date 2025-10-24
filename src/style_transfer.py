import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import copy
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from src.helper import *

def style_reconstruction_loss(I_ss: torch.Tensor, I_s: torch.Tensor) -> torch.Tensor:
    """
    Compute the style reconstruction loss between reconstructed and reference style images.

    Args:
        I_ss (torch.Tensor): The reconstructed style image from the diffusion model.
        I_s (torch.Tensor): The original style reference image.

    Returns:
        loss (torch.Tensor): A scalar tensor representing the loss.
    """
    return torch.tensor(0.0, requires_grad=True, device=I_ss.device)

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
    device: str
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
    """
    #get diffusion variables
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device)

    #initialize fine-tuned model
    model_finetune = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model_finetune.parameters(), lr=lr)

    #training loop
    for iter in range(k):
        #optimize the style reconstruction loss
        I_s = style_tensor.clone().to(device)
        for i in range(k_s):
            x_t = style_latent.clone().to(device)
            for s in reversed(range(1, s_rev)):
                t = torch.full((x_t.size(0),), s, device=device, dtype=torch.long)

                #DDIM backward step
                eps_pred = model_finetune(x_t, t)
                x_t_prev = (
                    torch.sqrt(alphas_cumprod[s - 1]) * model_finetune.decode(x_t, t)
                    + torch.sqrt(1 - alphas_cumprod[s - 1]) * eps_pred
                )

                #style reconstruction loss evaluation
                I_ss = model_finetune.decode(x_t, t)
                loss_sr = style_reconstruction_loss(I_ss, I_s)
                optimizer.zero_grad()
                loss_sr.backward()
                optimizer.step()

                x_t = x_t_prev.detach()
        
        #optimize the style disentanglement loss
        for i in range(len(content_latents)):
            x_t = content_latents[i].clone().to(device)
            for s in reversed(range(1, s_rev)):
                t = torch.full((x_t.size(0),), s, device=device, dtype=torch.long)

                #DDIM backward step
                eps_pred = model_finetune(x_t, t)
                x_t_prev = (
                    torch.sqrt(alphas_cumprod[s - 1]) * model_finetune.decode(x_t, t)
                    + torch.sqrt(1 - alphas_cumprod[s - 1]) * eps_pred
                )

                #style disentanglement loss evaluation
                I_ci = content_latents[i].clone().to(device)
                I_cs = model_finetune.decode(x_t, t)
                loss_sd = style_disentanglement_loss(I_ci, I_cs, I_ss, I_s)
                optimizer.zero_grad()
                loss_sd.backward()
                optimizer.step()

                x_t = x_t_prev.detach()

if __name__ == "__main__":
    pass