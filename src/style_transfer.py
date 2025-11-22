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
import clip
from models.improved_ddpm.script_util import i_DDPM

from src.helper import rgb_to_luma_601, prepare_image_as_tensor
from src.diffusion import ddim_deterministic, get_linear_alphas_cumprod

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

def style_disentanglement_loss(I_content_color: torch.Tensor, I_content_gray: torch.Tensor, I_style_color: torch.Tensor, I_style_gray: torch.Tensor, lambda_l1: float, lambda_dir: float,) -> torch.Tensor:
    """
    Compute the style disentanglement loss. All tensors has been preprocessed with CLIP for semantic feature embedding.

    Args:
        I_content_color (torch.Tensor): CLIP embedding of the stylized (color) content image.
        I_content_gray (torch.Tensor): CLIP embedding of the grayscale (structure-only) content image.
        I_style_color (torch.Tensor): CLIP embedding of the reference style image in color.
        I_style_gray (torch.Tensor): CLIP embedding of the grayscale version of the style image.
        lambda_l1 (float): Hyperparameter to weigh l1 loss.
        lambda_dir (float): Hyperparameter to weigh direction loss.

    Returns:
        loss (torch.Tensor): A scalar tensor representing the loss.
    """
    #L1 loss
    d_s = I_style_color - I_style_gray
    d_cs = I_content_color - I_content_gray
    l1_loss = F.l1_loss(d_cs, d_s)

    #direction loss
    cosine_sim = F.cosine_similarity(d_cs, d_s)
    dir_loss = 1 - cosine_sim

    #combined loss
    loss = lambda_l1 * l1_loss + lambda_dir * dir_loss

    return loss

def style_diffusion_fine_tuning(
    style_color: torch.Tensor,
    style_gray: torch.Tensor,
    style_latent: torch.Tensor,
    content_gray: list,
    content_latent: list,
    model: nn.Module,
    alphas_cumprod: list,
    clip_model,
    clip_preprocess,
    t_trans: int,
    s_rev: int,
    k: int,
    k_s: int,
    lr: float,
    lr_multiplier: float,
    lambda_style: float,
    lambda_l1: float,
    lambda_dir: float,
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
        style_color (torch.Tensor): The reference style image tensor in color space.
        style_gray (torch.Tensor): Grayscale version of the style image, used for disentanglement.
        style_latent (torch.Tensor): Latent representation of the style image.
        content_gray (list[torch.Tensor]): List of grayscale content images.
        content_latent (list[torch.Tensor]): List of latent representations of the content images.
        model (nn.Module): The diffusion model to fine-tune.
        alphas_cumprod (list[torch.Tensor]): Precomputed alpha cumprod tensor
        clip_model: CLIP model for pre-trained projected.
        clip_preprocess : CLIP preprocessing.
        t_trans (int): Diffusion timestep for style transfer.
        s_rev (int): Number of reverse diffusion steps.
        k (int): Number of fine-tuning outer iterations.
        k_s (int): Number of inner steps for style reconstruction loss optimization.
        lr (float): Learning rate for fine-tuning.
        lr_multiplier (float): Linear learning rate multiplier for fine-tuning.
        lambda_style (float): style reconstruction loss hyperparameter to weigh style loss
        lambda_l1 (float): style disentanglement loss hyperparameter to weigh l1 loss
        lambda_dir (float): style disentanglement loss hyperparameter to weigh direction loss
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
        I_s = style_color.detach().to(device)

        #optimize the style reconstruction loss
        for i in range(k_s):
            if logger is not None:
                logger.info(f"Starting style reconstruction iteration {i+1}...")

            x_t = style_latent.clone().to(device)

            ddim_timesteps_backward = np.linspace(0, t_trans, s_rev, dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]

            for step in range(len(ddim_timesteps_backward)-1):
                
                # Use DDIM deterministic reverse diffusion
                if logger is not None:
                    logger.info(f"Style reconstruction DDIM step: {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    alphas_cumprod=alphas_cumprod,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device,
                    requires_grad=True,
                )

                #style reconstruction loss evaluation
                I_ss = x_t_prev
                loss_sr = style_reconstruction_loss(I_ss, I_s) * lambda_style

                optimizer.zero_grad()
                loss_sr.backward()
                optimizer.step()

                x_t = x_t_prev.detach()
            
        #optimize the style disentanglement loss
        for i in range(len(content_latent)):
            if logger is not None:
                logger.info(f"Starting style disentanglement for sample number {i+1}...")

            x_t = content_latent[i].clone().to(device)

            ddim_timesteps_backward = np.linspace(0, t_trans, s_rev, dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]

            for step in range(len(ddim_timesteps_backward)-1):

                # Use DDIM deterministic reverse diffusion
                if logger is not None:
                    logger.info(f"Style disentanglement DDIM step: {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    alphas_cumprod=alphas_cumprod,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device,
                    requires_grad=True,
                )
                
                #style disentanglement loss evaluation

                #detach tensors that does not flow gradients to the finetuned model
                preprocess = T.Compose([T.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] to [0, 1].
                                              clip_preprocess.transforms[:2] +   # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])    # + skip convert PIL to tensor

                I_content_color = clip_model.encode_image(preprocess(x_t_prev))
                I_content_gray = clip_model.encode_image(preprocess(content_gray[i])).detach()
                I_style_color  = clip_model.encode_image(preprocess(style_color)).detach()
                I_style_gray = clip_model.encode_image(preprocess(style_gray)).detach()

                #calculate style disentanglement loss
                loss_sd = style_disentanglement_loss(
                    I_content_color, 
                    I_content_gray, 
                    I_style_color, 
                    I_style_gray, 
                    lambda_l1, 
                    lambda_dir
                )

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
    # CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    CHECKPOINT_PATH = "models/checkpoints/512x512_diffusion.pt"
    IMAGE_SIZE = 512

    DIFFUSION_NUM_TIMESTEPS = 1000
    DIFFUSION_BETA_START = 0.0001
    DIFFUSION_BETA_END = 0.02

    T_TRANS = 601
    S_FOR = 40
    S_REV = 6

    K = 5
    K_S = 50
    LR = 0.000004
    LR_MULTIPLIER = 1.2
    LAMBDA_STYLE = 1
    LAMBDA_L1 = 10
    LAMBDA_DIR = 6

    N_CONTENT_SAMPLE = 50

    CONTENT_GRAY_PATH = "output/test_run/content_processed/"
    CONTENT_LATENT_PATH = "output/test_run/content_latents/"
    STYLE_COLOR_PATH = "data/style/van_gogh/000.jpg"
    STYLE_GRAY_PATH = "output/test_run/style_processed/style.pt"
    STYLE_LATENT_PATH = "output/test_run/style_latents/style.pt"

    SAMPLE_CONTENT_ID = 10

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

    #load model
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

    #get sample style color, gray, and latent
    style_color = Image.open(STYLE_COLOR_PATH)
    style_color = prepare_image_as_tensor(style_color, image_size=IMAGE_SIZE, device=DEVICE)
    style_gray = torch.load(STYLE_GRAY_PATH, map_location=DEVICE, weights_only=True)
    style_latent = torch.load(STYLE_LATENT_PATH, map_location=DEVICE, weights_only=True)
    
    #get sample content color, gray, and latent
    content_filenames = [f for f in os.listdir(CONTENT_LATENT_PATH) if f.lower().endswith('.pt')]
    content_filenames.sort()
    sample_content_filenames = content_filenames[:N_CONTENT_SAMPLE]
    logger.info(f"Sample content image filename: {sample_content_filenames[SAMPLE_CONTENT_ID]}")

    content_gray = []
    for file in sample_content_filenames:
        content_gray.append(
            torch.load(os.path.join(CONTENT_GRAY_PATH, file), map_location=DEVICE, weights_only=True)            
        )

    content_latent = []
    for file in sample_content_filenames:
        content_latent.append(
            torch.load(os.path.join(CONTENT_LATENT_PATH, file), map_location=DEVICE, weights_only=True)
        )
    
    logger.info(f"Style color tensor shape: {style_color.shape}")
    logger.info(f"Style gray tensor shape: {style_gray.shape}")
    logger.info(f"Style latent tensor shape: {style_latent.shape}")
    logger.info(f"Content gray tensors sample count: {len(content_gray)}")
    logger.info(f"Content gray tensor shape: {content_gray[SAMPLE_CONTENT_ID].shape}")
    logger.info(f"Content latent tensors sample count: {len(content_latent)}")
    logger.info(f"Content latent tensor shape: {content_latent[SAMPLE_CONTENT_ID].shape}")

    #test reverse diffusion to reconstruct style
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    style_x0_est = ddim_deterministic(style_latent, model, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    style_x0_est_image = style_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style_x0_est_image = ((style_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style_x0_est_image = (style_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(style_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_after_reverse_diffusion.jpg"))

    #test reverse diffusion to reconstruct sample content
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    content_x0_est = ddim_deterministic(content_latent[SAMPLE_CONTENT_ID], model, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    content_x0_est_image = content_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x0_est_image = ((content_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x0_est_image = (content_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(content_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_reverse_diffusion.jpg"))

    #load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

    #apply style diffusion fine-tuning
    model_finetuned = style_diffusion_fine_tuning(
        style_color,
        style_gray,
        style_latent,
        content_gray,
        content_latent,
        model,
        alphas_cumprod,
        clip_model,
        clip_preprocess,
        T_TRANS,
        S_REV,
        K,
        K_S,
        LR,
        LR_MULTIPLIER,
        LAMBDA_STYLE,
        LAMBDA_L1,
        LAMBDA_DIR,
        DEVICE,
        logger=logger,
    )
    torch.save(model_finetuned.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}finetuned_style_model.pt"))
    # model_finetuned = i_DDPM("IMAGENET", IMAGE_SIZE)
    # finetuned_ckpt = torch.load(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}finetuned_style_model.pt"), weights_only=True)
    # model_finetuned.load_state_dict(finetuned_ckpt)
    # model_finetuned.eval().to(DEVICE)

    #test reverse diffusion to reconstruct style with finetuned model
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    # ddim_timesteps_backward = np.linspace(0, T_TRANS, S_FOR, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    style_x0_est = ddim_deterministic(style_latent, model_finetuned, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    style_x0_est_image = style_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style_x0_est_image = ((style_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style_x0_est_image = (style_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(style_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_after_reverse_diffusion_with_finetuned_model.jpg"))

    #test reverse diffusion to reconstruct sample_content with finetuned model
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    # ddim_timesteps_backward = np.linspace(0, T_TRANS, S_FOR, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    content_x0_est = ddim_deterministic(content_latent[SAMPLE_CONTENT_ID], model_finetuned, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    content_x0_est_image = content_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x0_est_image = ((content_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x0_est_image = (content_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(content_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_reverse_diffusion_with_finetuned_model.jpg"))