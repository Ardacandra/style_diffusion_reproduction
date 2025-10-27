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
import clip

from src.helper import *
from src.style_removal import ddim_deterministic

# CLIP imagenet-like normalization used by OpenAI CLIP (ViT-B/32)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def tensor_to_clip_input_tensor(img: torch.Tensor, size: int = 224, device: str = "cuda"):
    """
    Convert a torch tensor (latent or image) into a CLIP-friendly tensor **without** leaving torch.
    Expects img shape [B, C, H, W] or [C, H, W]. Values can be in [-1,1] or [0,1].
    Returns float tensor of shape [B, 3, size, size] on `device`.
    Differentiable (no cpu/numpy/PIL).
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)  # [1, C, H, W]

    img = img.to(dtype=torch.float32, device=device)

    # If in [-1,1], map to [0,1]
    if img.min() < 0.0:
        img = (img.clamp(-1, 1) + 1.0) / 2.0
    else:
        img = img.clamp(0.0, 1.0)

    # If single-channel, repeat to 3 channels
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    elif img.shape[1] == 4:
        # if RGBA, drop alpha
        img = img[:, :3, :, :]

    # Resize to CLIP input size using bilinear interpolation
    img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)

    # Normalize with CLIP mean/std
    mean = _CLIP_MEAN.to(device).view(1, 3, 1, 1)
    std = _CLIP_STD.to(device).view(1, 3, 1, 1)
    img = (img - mean) / std

    return img  # differentiable tensor on device

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

def style_disentanglement_loss(f_ci: torch.Tensor, f_cs: torch.Tensor, f_ss: torch.Tensor, f_s: torch.Tensor, lambda_l1: int, lambda_dir: int,) -> torch.Tensor:
    """
    Compute the style disentanglement loss. All tensors has been preprocessed with CLIP for semantic feature embedding.

    Args:
        f_ci (torch.Tensor): Original content image latent or embedding.
        f_cs (torch.Tensor): Style-modified content image (decoded from diffusion model).
        f_ss (torch.Tensor): Reconstructed style image from the style latent.
        f_s  (torch.Tensor): Original style reference image.
        lambda_l1 (int): Hyperparameter to weigh l1 loss.
        lambda_dir (int): Hyperparameter to weigh direction loss.

    Returns:
        loss (torch.Tensor): A scalar tensor representing the loss.
    """
    #L1 loss
    d_s = f_s - f_ss
    d_cs = f_cs - f_ci
    l1_loss = F.l1_loss(d_cs, d_s)

    #direction loss
    cosine_sim = F.cosine_similarity(d_cs, d_s, dim=-1).mean()
    dir_loss = 1 - cosine_sim

    #combined loss
    loss = lambda_l1 * l1_loss + lambda_dir * dir_loss

    return loss

def style_diffusion_fine_tuning(
    style_tensor: torch.Tensor,
    style_latent: torch.Tensor,
    content_latents: list,
    model: nn.Module,
    diffusion,
    clip_model,
    clip_preprocess,
    s_rev: int,
    k: int,
    k_s: int,
    lr: float,
    lr_multiplier: float,
    lambda_l1: int,
    lambda_dir: int,
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
        clip_model: CLIP model for pre-trained projected.
        clip_preprocess : CLIP preprocessing.
        s_rev (int): Number of reverse diffusion steps.
        k (int): Number of fine-tuning outer iterations.
        k_s (int): Number of inner steps for style reconstruction loss optimization.
        lr (float): Learning rate for fine-tuning.
        lr_multiplier (float): Linear learning rate multiplier for fine-tuning.
        lambda_l1 (int): style disentanglement loss hyperparameter to weigh l1 loss
        lambda_dir (int): style disentanglement loss hyperparameter to weigh direction loss
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
    #create linear scheduler
    lambda_lr = lambda epoch: lr_multiplier ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    #freeze CLIP model weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    #training loop
    for iter in range(k):
        if logger is not None:
            logger.info(f"Starting fine-tuning iteration {iter+1}...")

        #initialize style reference I_s
        I_s = style_tensor.detach().to(device)

        #optimize the style reconstruction loss
        for i in range(k_s):
            if logger is not None:
                logger.info(f"Starting style reconstruction iteration {i+1}...")

            x_t = style_latent.clone().to(device)

            ddim_timesteps_backward = np.linspace(0, diffusion.num_timesteps-1, s_rev, dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]

            for step in range(len(ddim_timesteps_backward)-1):
                
                # Use DDIM deterministic reverse diffusion
                if logger is not None:
                    logger.info(f"Style reconstruction DDIM step: {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device,
                    requires_grad=True,
                )

                #style reconstruction loss evaluation
                I_ss = x_t_prev
                loss_sr = style_reconstruction_loss(I_ss, I_s)

                optimizer.zero_grad()
                loss_sr.backward()
                optimizer.step()

                x_t = x_t_prev.detach()
        
        #initialize style reconstruction reference I_ss
        x_t_style = style_latent.clone().to(device)

        ddim_timesteps_backward = np.linspace(0, diffusion.num_timesteps-1, s_rev, dtype=int)
        ddim_timesteps_backward = ddim_timesteps_backward[::-1]

        with torch.no_grad():
            for step in range(len(ddim_timesteps_backward)-1):
                if logger is not None:
                    logger.info(f"Precomputing I_ss: DDIM step {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_style = ddim_deterministic(
                    x_start=x_t_style,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device,
                    requires_grad=False,
                )
        I_ss = x_t_style.detach()
            
        #optimize the style disentanglement loss
        for i in range(len(content_latents)):
            if logger is not None:
                logger.info(f"Starting style disentanglement for sample number {i+1}...")

            x_t = content_latents[i].clone().to(device)

            ddim_timesteps_backward = np.linspace(0, diffusion.num_timesteps-1, s_rev, dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]

            for step in range(len(ddim_timesteps_backward)-1):

                # Use DDIM deterministic reverse diffusion
                if logger is not None:
                    logger.info(f"Style disentanglement DDIM step: {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device,
                    requires_grad=True,
                )
                
                #style disentanglement loss evaluation
                I_ci = content_latents[i].detach()
                I_cs = x_t_prev

                # Apply CLIP's preprocess
                if logger is not None:
                    logger.info(f"Applying CLIP preprocessing...")
                
                #log tensor shapes and stats for debugging
                tensors_before = {
                    "I_ci": I_ci,
                    "I_cs": I_cs,
                    "I_ss": I_ss,
                    "I_s":  I_s,
                }
                for name, t in tensors_before.items():
                    summarize_tensor(name, t, logger)
                
                #detach tensors that does not flow gradients to the finetuned model
                f_ci = tensor_to_clip_input_tensor(I_ci, size=224, device=device).detach()
                f_cs = tensor_to_clip_input_tensor(I_cs, size=224, device=device)
                f_ss = tensor_to_clip_input_tensor(I_ss, size=224, device=device).detach()
                f_s  = tensor_to_clip_input_tensor(I_s, size=224, device=device).detach()

                f_ci = clip_model.encode_image(f_ci)
                f_cs = clip_model.encode_image(f_cs)
                f_ss  = clip_model.encode_image(f_ss)
                f_s = clip_model.encode_image(f_s)

                #log tensor shapes and stats for debugging
                tensors_after = {
                    "f_ci": f_ci,
                    "f_cs": f_cs,
                    "f_ss": f_ss,
                    "f_s":  f_s,
                }
                for name, t in tensors_before.items():
                    summarize_tensor(name, t, logger)

                if logger is not None:
                    logger.info(f"CLIP preprocessing done.")

                #calculate style disentanglement loss
                loss_sd = style_disentanglement_loss(f_ci, f_cs, f_ss, f_s, lambda_l1, lambda_dir)
                optimizer.zero_grad()
                loss_sd.backward()
                optimizer.step()

                x_t = x_t_prev.detach()

        scheduler.step()

    if logger is not None:
        logger.info("Style transfer fine-tuning completed.")
    return model_finetuned

if __name__ == "__main__":
    #Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    IMAGE_SIZE = 256
    S_FOR = 40
    S_REV = 6
    # S_REV = 20

    K = 5
    K_S = 50
    LR = 0.000004
    LR_MULTIPLIER = 1.2
    LAMBDA_L1 = 10
    LAMBDA_DIR = 1

    N_CONTENT_SAMPLE = 50

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

    #load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

    #apply style diffusion fine-tuning
    model_finetuned = style_diffusion_fine_tuning(
        original_style_tensor,
        style_latent,
        content_latents,
        model,
        diffusion,
        clip_model,
        clip_preprocess,
        S_REV,
        K,
        K_S,
        LR,
        LR_MULTIPLIER,
        LAMBDA_L1,
        LAMBDA_DIR,
        DEVICE,
        logger=logger,
    )
    torch.save(model_finetuned.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}finetuned_style_model.pt"))

    #generate sample stylized image
    x_t = content_latents[0].clone().to(DEVICE)
    ddim_timesteps_backward = np.linspace(0, S_FOR-1, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    x0_est = ddim_deterministic(x_t, model_finetuned, diffusion, ddim_timesteps_backward, device=DEVICE)

    stylized_image = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stylized_image = ((stylized_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    stylized_image = (stylized_image * 255).astype(np.uint8)
    # plt.imshow(stylized_image)
    # plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "sample_stylized_image.png"), bbox_inches='tight', dpi=300)
    Image.fromarray(stylized_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "sample_stylized_image.jpg"))
