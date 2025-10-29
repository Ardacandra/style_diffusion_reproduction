import os
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
from src.style_removal import *
from src.style_transfer import *

if __name__ == "__main__":
    #temporary logic for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    IMAGE_SIZE = 256
    S_FOR = 40
    # S_REV = 40
    # S_REV = 20
    # S_REV = 6

    CONTENT_IMAGE_PATH = "data/content/0045.jpg"
    CONTENT_LATENT_PATH = "output/test_run/content_latents/0045.pt"
    STYLE_IMAGE_PATH = "data/style/van_gogh/000.jpg"
    STYLE_LATENT_PATH = "output/test_run/style_latents/style.pt"

    OUTPUT_DIR = "output/"
    OUTPUT_PREFIX = "debug__"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR, f"debug.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info(f"Debugging...")

    #load pre-trained model
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

    #checking sample content image
    content_image = Image.open(CONTENT_IMAGE_PATH).convert('RGB')
    content_image.save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image.jpg"))

    #checking tensor of sample content image
    content_tensor = prepare_image_as_tensor(content_image, image_size=IMAGE_SIZE, device=DEVICE)
    summarize_tensor("content_tensor", content_tensor, logger)

    #checking image recreated from content tensor
    content_image_rec = content_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_image_rec = ((content_image_rec + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_image_rec = (content_image_rec * 255).astype(np.uint8)
    Image.fromarray(content_image_rec).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_from_tensor.jpg"))

    #apply color removal
    content_image_luma = rgb_to_luma_601(content_image)
    Image.fromarray(content_image_luma).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_color_removal.jpg"))

    #forward diffusion to obtain latents
    content_x0 = prepare_image_as_tensor(Image.fromarray(content_image_luma), image_size=IMAGE_SIZE, device=DEVICE)
    summarize_tensor("content_x0", content_tensor, logger)
    ddim_timesteps_forward = np.linspace(0, diffusion.num_timesteps - 1, S_FOR, dtype=int)
    content_x_t = ddim_deterministic(content_x0, model, diffusion, ddim_timesteps_forward, DEVICE, logger=logger)
    summarize_tensor("content_x_t", content_x_t, logger)
    content_x_t_image = content_x_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x_t_image = ((content_x_t_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x_t_image = (content_x_t_image * 255).astype(np.uint8)
    Image.fromarray(content_x_t_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_forward_diffusion.jpg"))

    #reverse diffusion with 40 steps
    s_rev = 40
    ddim_timesteps_backward = np.linspace(0, S_FOR - 1, s_rev, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    content_x0_est_s40 = ddim_deterministic(content_x_t, model, diffusion, ddim_timesteps_backward, DEVICE, logger=logger)
    summarize_tensor("content_x0_est_s40", content_x0_est_s40, logger)
    content_x0_est_s40_image = content_x0_est_s40.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x0_est_s40_image = ((content_x0_est_s40_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x0_est_s40_image = (content_x0_est_s40_image * 255).astype(np.uint8)
    Image.fromarray(content_x0_est_s40_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_reverse_diffusion_40_steps.jpg"))

    #reverse diffusion with 6 steps
    s_rev = 6
    ddim_timesteps_backward = np.linspace(0, S_FOR - 1, s_rev, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    content_x0_est_s6 = ddim_deterministic(content_x_t, model, diffusion, ddim_timesteps_backward, DEVICE, logger=logger)
    summarize_tensor("content_x0_est_s6", content_x0_est_s6, logger)
    content_x0_est_s6_image = content_x0_est_s6.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x0_est_s6_image = ((content_x0_est_s6_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x0_est_s6_image = (content_x0_est_s6_image * 255).astype(np.uint8)
    Image.fromarray(content_x0_est_s6_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_reverse_diffusion_6_steps.jpg"))

    #checking sample style image
    style_image = Image.open(STYLE_IMAGE_PATH).convert('RGB')
    style_image.save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image.jpg"))

    #checking tensor of sample style image
    style_tensor = prepare_image_as_tensor(style_image, image_size=IMAGE_SIZE, device=DEVICE)
    summarize_tensor("style_tensor", style_tensor, logger)

    #checking image recreated from style tensor
    style_image_rec = style_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style_image_rec = ((style_image_rec + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style_image_rec = (style_image_rec * 255).astype(np.uint8)
    Image.fromarray(style_image_rec).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_from_tensor.jpg"))

    #apply color removal
    style_image_luma = rgb_to_luma_601(style_image)
    Image.fromarray(style_image_luma).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_after_color_removal.jpg"))

    #forward diffusion to obtain latents
    style_x0 = prepare_image_as_tensor(Image.fromarray(style_image_luma), image_size=IMAGE_SIZE, device=DEVICE)
    summarize_tensor("style_x0", style_tensor, logger)
    ddim_timesteps_forward = np.linspace(0, diffusion.num_timesteps - 1, S_FOR, dtype=int)
    style_x_t = ddim_deterministic(style_x0, model, diffusion, ddim_timesteps_forward, DEVICE, logger=logger)
    summarize_tensor("style_x_t", style_x_t, logger)
    style_x_t_image = style_x_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style_x_t_image = ((style_x_t_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style_x_t_image = (style_x_t_image * 255).astype(np.uint8)
    Image.fromarray(style_x_t_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_after_forward_diffusion.jpg"))

    #reverse diffusion with 40 steps
    s_rev = 40
    ddim_timesteps_backward = np.linspace(0, S_FOR - 1, s_rev, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    style_x0_est_s40 = ddim_deterministic(style_x_t, model, diffusion, ddim_timesteps_backward, DEVICE, logger=logger)
    summarize_tensor("style_x0_est_s40", style_x0_est_s40, logger)
    style_x0_est_s40_image = style_x0_est_s40.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style_x0_est_s40_image = ((style_x0_est_s40_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style_x0_est_s40_image = (style_x0_est_s40_image * 255).astype(np.uint8)
    Image.fromarray(style_x0_est_s40_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_after_reverse_diffusion_40_steps.jpg"))

    #reverse diffusion with 6 steps
    s_rev = 6
    ddim_timesteps_backward = np.linspace(0, S_FOR - 1, s_rev, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    style_x0_est_s6 = ddim_deterministic(style_x_t, model, diffusion, ddim_timesteps_backward, DEVICE, logger=logger)
    summarize_tensor("style_x0_est_s6", style_x0_est_s6, logger)
    style_x0_est_s6_image = style_x0_est_s6.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style_x0_est_s6_image = ((style_x0_est_s6_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style_x0_est_s6_image = (style_x0_est_s6_image * 255).astype(np.uint8)
    Image.fromarray(style_x0_est_s6_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style_image_after_reverse_diffusion_6_steps.jpg"))

    #============================================================================================
    #DEBUGGING STYLE TRANSFER LOGIC
    #============================================================================================
    S_REV = 20
    K = 5
    # K_S = 50
    K_S = 10
    LR = 0.00004
    LR_MULTIPLIER = 1.2
    LAMBDA_L1 = 10
    LAMBDA_DIR = 1

    #load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    #freeze CLIP model weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    #initialize fine-tuned model
    model_finetuned, diffusion = create_model_and_diffusion(**options)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model_finetuned.load_state_dict(state_dict)
    model_finetuned.to(DEVICE)
    optimizer = torch.optim.Adam(model_finetuned.parameters(), lr=LR)
    lambda_lr = lambda epoch: LR_MULTIPLIER ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    #define variables needed for the style transfer
    original_style_tensor = style_tensor.clone().to(DEVICE)
    style_latent = style_x0_est_s40.clone().to(DEVICE)
    content_latent = content_x_t.clone().to(DEVICE)

    #training loop
    for iter in range(K):
        logger.info(f"Starting fine-tuning iteration {iter+1}...")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        #initialize style reference I_s
        I_s = style_tensor.detach().to(DEVICE)

        #optimize the style reconstruction loss
        for i in range(K_S):
            logger.info(f"Starting style reconstruction iteration {i+1}...")

            x_t = style_latent.clone().to(DEVICE)

            ddim_timesteps_backward = np.linspace(0, diffusion.num_timesteps-1, S_REV, dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]

            for step in range(len(ddim_timesteps_backward)-1):
                
                # Use DDIM deterministic reverse diffusion
                logger.info(f"Style reconstruction DDIM step: {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_prev = ddim_deterministic(
                    x_start=x_t,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=DEVICE,
                    requires_grad=True,
                )

                #style reconstruction loss evaluation
                I_ss = x_t_prev
                summarize_tensor("I_ss", I_ss, logger)
                summarize_tensor("I_s", I_s, logger)
                loss_sr = style_reconstruction_loss(I_ss, I_s)
                logger.info(f"Style reconstruction loss: {loss_sr:.5f}")

                optimizer.zero_grad()
                loss_sr.backward()
                optimizer.step()

                x_t = x_t_prev.detach()

        #initialize style reconstruction reference I_ss
        x_t_style = style_latent.clone().to(DEVICE)

        ddim_timesteps_backward = np.linspace(0, diffusion.num_timesteps-1, S_REV, dtype=int)
        ddim_timesteps_backward = ddim_timesteps_backward[::-1]

        with torch.no_grad():
            for step in range(len(ddim_timesteps_backward)-1):
                logger.info(f"Precomputing I_ss: DDIM step {ddim_timesteps_backward[step]} -> {ddim_timesteps_backward[step+1]}")

                x_t_style = ddim_deterministic(
                    x_start=x_t_style,
                    model=model_finetuned,
                    diffusion=diffusion,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=DEVICE,
                    requires_grad=False,
                )
        I_ss = x_t_style.detach()
        summarize_tensor("I_ss", I_ss, logger)

        #optimize the style disentanglement loss
        # for i in range(len(content_latents)):
        for i in range(1):
            logger.info(f"Starting style disentanglement for sample number {i+1}...")

            # x_t = content_latents[i].clone().to(device)
            x_t = content_latent.clone().to(DEVICE)

            ddim_timesteps_backward = np.linspace(0, diffusion.num_timesteps-1, S_REV, dtype=int)
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
                    device=DEVICE,
                    requires_grad=True,
                )
                
                #style disentanglement loss evaluation
                # I_ci = content_latents[i].detach()
                I_ci = content_latent.detach()
                I_cs = x_t_prev

                # Apply CLIP's preprocess
                logger.info(f"Applying CLIP preprocessing...")

                # #log tensor shapes and stats for debugging
                # tensors_before = {
                #     "I_ci": I_ci,
                #     "I_cs": I_cs,
                #     "I_ss": I_ss,
                #     "I_s":  I_s,
                # }
                # for name, t in tensors_before.items():
                #     logger.info("Tensor stats before CLIP preprocessing:")
                #     summarize_tensor(name, t, logger)
                
                #detach tensors that does not flow gradients to the finetuned model
                f_ci = tensor_to_clip_input_tensor(I_ci, size=224, device=DEVICE).detach()
                f_cs = tensor_to_clip_input_tensor(I_cs, size=224, device=DEVICE)
                f_ss = tensor_to_clip_input_tensor(I_ss, size=224, device=DEVICE).detach()
                f_s  = tensor_to_clip_input_tensor(I_s, size=224, device=DEVICE).detach()

                # #log tensor shapes and stats for debugging
                # tensors_mid = {
                #     "f_ci": f_ci,
                #     "f_cs": f_cs,
                #     "f_ss": f_ss,
                #     "f_s":  f_s,
                # }
                # for name, t in tensors_mid.items():
                #     logger.info("Tensor stats after CLIP preprocessing:")
                #     summarize_tensor(name, t, logger)

                f_ci = clip_model.encode_image(f_ci)
                f_cs = clip_model.encode_image(f_cs)
                f_ss  = clip_model.encode_image(f_ss)
                f_s = clip_model.encode_image(f_s)

                # #log tensor shapes and stats for debugging
                # tensors_after = {
                #     "f_ci": f_ci,
                #     "f_cs": f_cs,
                #     "f_ss": f_ss,
                #     "f_s":  f_s,
                # }
                # for name, t in tensors_after.items():
                #     logger.info("Tensor stats after CLIP encoding:")
                #     summarize_tensor(name, t, logger)

                logger.info(f"CLIP preprocessing done.")

                #calculate style disentanglement loss
                loss_sd = style_disentanglement_loss(f_ci, f_cs, f_ss, f_s, LAMBDA_L1, LAMBDA_DIR)
                logger.info(f"Style disentanglement loss: {loss_sd:.5f}")
                optimizer.zero_grad()
                loss_sd.backward()
                optimizer.step()

                x_t = x_t_prev.detach()

        scheduler.step()

    #test reverse diffusion of fine-tuned model
    ddim_timesteps_backward = np.linspace(0, S_FOR - 1, S_REV, dtype=int)
    # ddim_timesteps_backward = np.linspace(0, S_FOR - 1, S_FOR, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    model_finetuned_output = ddim_deterministic(content_latent, model_finetuned, diffusion, ddim_timesteps_backward, DEVICE, logger=logger)
    summarize_tensor("model_finetuned_output", model_finetuned_output, logger)
    model_finetuned_output_image = model_finetuned_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    model_finetuned_output_image = ((model_finetuned_output_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    model_finetuned_output_image = (model_finetuned_output_image * 255).astype(np.uint8)
    Image.fromarray(model_finetuned_output_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "model_finetuned_output.jpg"))  