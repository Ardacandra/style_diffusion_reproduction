import argparse
import yaml
import logging
import os
from PIL import Image
from datasets import load_dataset   # huggingface 'datasets' library
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import clip

from models.improved_ddpm.script_util import i_DDPM
from src.helper import rgb_to_luma_601, prepare_image_as_tensor
from src.diffusion import ddim_deterministic, get_linear_alphas_cumprod
from src.style_transfer import *

def main(config_path):
    # loading configurations
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_path'], exist_ok=True)    
    run_output_path = os.path.join(cfg['output_path'], cfg['run_id'])
    os.makedirs(run_output_path, exist_ok=True)    

    logging.basicConfig(
        filename=os.path.join(run_output_path, f"{cfg['run_id']}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info(f"Run parameters: {cfg}")

    model = i_DDPM("IMAGENET", cfg['image_size'])
    init_ckpt = torch.load(cfg['pretrained_model_path'], weights_only=True)
    model.load_state_dict(init_ckpt)
    model.to(cfg['device'])

    #get alphas_cumprod
    alphas_cumprod = get_linear_alphas_cumprod(
        timesteps=cfg['diffusion_num_timesteps'],
        beta_start=cfg['diffusion_beta_start'],
        beta_end=cfg['diffusion_beta_end']
    )

    if cfg['run_mode'] == 'style_removal':
        logger.info("Starting style removal...")

        ### apply color removal and diffusion-based style removal for style images
        logger.info(f"Processing style image: {cfg['style_path']}")
        style_output_path = os.path.join(run_output_path, "style_processed")
        os.makedirs(style_output_path, exist_ok=True)
        style_image = Image.open(cfg['style_path']).convert('RGB')
        style_image_luma = rgb_to_luma_601(style_image)       
        #convert luma image into tensor
        x0 = prepare_image_as_tensor(Image.fromarray(style_image_luma), image_size=cfg['image_size'], device=cfg['device'])
        #forward diffusion
        ddim_timesteps_forward = np.linspace(0, cfg['t_remov'], cfg['style_removal_s_for'], dtype=int)
        x_t = ddim_deterministic(x0, model, alphas_cumprod, ddim_timesteps_forward, cfg['device'], logger=logger)
        #reverse diffusion (DDIM)
        ddim_timesteps_backward = np.linspace(0, cfg['t_remov'], cfg['style_removal_s_rev'], dtype=int)
        ddim_timesteps_backward = ddim_timesteps_backward[::-1]
        assert ddim_timesteps_backward[-1]==0
        x0_est = ddim_deterministic(x_t, model, alphas_cumprod, ddim_timesteps_backward, device=cfg['device'], logger=logger)
        #save output tensor and image
        torch.save(x0_est, os.path.join(style_output_path, 'style.pt'))
        logger.info(f"Style latent saved to {style_output_path}")
        image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_recon = ((image_recon + 1) / 2).clip(0, 1)
        image_recon = (image_recon * 255).astype(np.uint8)
        Image.fromarray(image_recon).save(os.path.join(style_output_path, "style.jpg"))
        logger.info(f"Style image processed and saved to {style_output_path}")

        ###apply color removal and diffusion-based style removal for content images
        content_output_path = os.path.join(run_output_path, "content_processed")
        os.makedirs(content_output_path, exist_ok=True)
        for image_file in os.listdir(cfg['content_path']):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                logger.info(f"Processing content image: {image_file}")
                image = Image.open(os.path.join(cfg['content_path'], image_file)).convert('RGB')
                image_luma = rgb_to_luma_601(image)
                #convert luma image into tensor
                x0 = prepare_image_as_tensor(Image.fromarray(image_luma), image_size=cfg['image_size'], device=cfg['device'])
                #forward diffusion
                ddim_timesteps_forward = np.linspace(0, cfg['t_remov'], cfg['style_removal_s_for'], dtype=int)
                x_t = ddim_deterministic(x0, model, alphas_cumprod, ddim_timesteps_forward, cfg['device'], logger=logger)
                #reverse diffusion (DDIM)
                ddim_timesteps_backward = np.linspace(0, cfg['t_remov'], cfg['style_removal_s_rev'], dtype=int)
                ddim_timesteps_backward = ddim_timesteps_backward[::-1]
                assert ddim_timesteps_backward[-1]==0
                x0_est = ddim_deterministic(x_t, model, alphas_cumprod, ddim_timesteps_backward, device=cfg['device'], logger=logger)
                #save output tensor and image
                torch.save(x0_est, os.path.join(content_output_path, f"{image_file.split('.')[0]}.pt"))
                logger.info(f"Content latent saved to {content_output_path}")
                image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
                image_recon = ((image_recon + 1) / 2).clip(0, 1)
                image_recon = (image_recon * 255).astype(np.uint8)
                Image.fromarray(image_recon).save(os.path.join(content_output_path, image_file))
                logger.info(f"Style image processed and saved to {content_output_path}")
    
    if cfg['run_mode'] == 'style_transfer':
        logger.info("Starting style transfer...")
        
        if cfg['precompute_latents']:
            logger.info("Precomputing latents...")

            ddim_timesteps_forward = np.linspace(0, cfg['t_trans'], cfg['style_transfer_s_for'], dtype=int)

            #precompute content latents
            content_latent_output_path = os.path.join(run_output_path, "content_latents")
            os.makedirs(content_latent_output_path, exist_ok=True)
            
            for content_file in os.listdir(cfg['content_gray_path']):
                if content_file.lower().endswith(('.pt')):
                    logger.info(f"Generating latent for content file: {content_file}")
                    content_tensor = torch.load(os.path.join(cfg['content_gray_path'], content_file), map_location=cfg['device'], weights_only=True)
                    content_latent = ddim_deterministic(content_tensor, model, alphas_cumprod, ddim_timesteps_forward, cfg['device'], logger=logger)
                    torch.save(content_latent, os.path.join(content_latent_output_path, f"{content_file.lower().split('.')[0]}.pt"))
                    logger.info(f"{content_file} processed and saved to {content_latent_output_path}")
            
            #precompute style latent
            style_latent_output_path = os.path.join(run_output_path, "style_latents")
            os.makedirs(style_latent_output_path, exist_ok=True) 

            logger.info(f"Generating latent for style : {cfg['style_gray_path']}")
            style_tensor = torch.load(cfg['style_gray_path'], map_location=cfg['device'], weights_only=True)
            style_latent = ddim_deterministic(style_tensor, model, alphas_cumprod, ddim_timesteps_forward, cfg['device'], logger=logger)
            torch.save(style_latent, os.path.join(style_latent_output_path, "style.pt"))
            logger.info(f"Style image processed and saved to {style_latent_output_path}")        

        logger.info("Starting style transfer fine-tuning...")

        #get style color, gray, and latent
        style_color = Image.open(cfg['style_color_path'])
        style_color = prepare_image_as_tensor(style_color, image_size=cfg['image_size'], device=cfg['device'])
        style_gray = torch.load(cfg['style_gray_path'], map_location=cfg['device'], weights_only=True)
        style_latent = torch.load(cfg['style_latent_path'], map_location=cfg['device'], weights_only=True)

        #get content color, gray, and latent
        content_filenames = [f for f in os.listdir(cfg['content_latent_path']) if f.lower().endswith('.pt')]
        content_filenames.sort()

        content_gray = []
        for file in content_filenames:
            content_gray.append(
                torch.load(os.path.join(cfg['content_gray_path'], file), map_location=cfg['device'], weights_only=True)            
            )

        content_latent = []
        for file in content_filenames:
            content_latent.append(
                torch.load(os.path.join(cfg['content_latent_path'], file), map_location=cfg['device'], weights_only=True)
            )

        #load clip model
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=cfg['device'])

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
            cfg['t_trans'],
            cfg['style_transfer_s_rev'],
            cfg['k'],
            cfg['k_s'],
            cfg['lr'],
            cfg['lr_multiplier'],
            cfg['lambda_style'],
            cfg['lambda_l1'],
            cfg['lambda_dir'],
            cfg['device'],
            logger=logger,
        )
        torch.save(model_finetuned.state_dict(), os.path.join(run_output_path, f"finetuned_style_model.pt"))

        #generate stylized image
        logger.info("Generating stylized images using fine-tuned model...")

        stylized_output_path = os.path.join(run_output_path, "content_stylized")
        os.makedirs(stylized_output_path, exist_ok=True)
        for i in range(len(content_latent)):
            x_t = content_latent[i].clone().to(cfg['device'])
            ddim_timesteps_backward = np.linspace(0, cfg['t_trans'], cfg['style_transfer_s_rev'], dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]
            x0_est = ddim_deterministic(x_t, model_finetuned, alphas_cumprod, ddim_timesteps_backward, device=cfg['device'])

            stylized_image = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
            stylized_image = ((stylized_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
            stylized_image = (stylized_image * 255).astype(np.uint8)
            Image.fromarray(stylized_image).save(os.path.join(stylized_output_path, f"{content_filenames[i].split('.')[0]}.jpg"))
        
        logger.info("Stylized images generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)