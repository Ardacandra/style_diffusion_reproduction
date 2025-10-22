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

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from src.helper import *
from src.style_removal import *

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

    if cfg['run_mode'] == 'style_removal':
        pass

        # logger.info("Starting style removal...")

        # #for content images, apply color removal only
        # content_output_path = os.path.join(run_output_path, "content_processed")
        # os.makedirs(content_output_path, exist_ok=True)
        # for image_file in os.listdir(cfg['content_path']):
        #     if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        #         logger.info(f"Processing content image: {image_file}")
        #         image = Image.open(os.path.join(cfg['content_path'], image_file)).convert('RGB')
        #         image_luma = rgb_to_luma_601(image)
        #         Image.fromarray(image_luma).save(os.path.join(content_output_path, image_file))
        #         logger.info(f"{image_file} processed and saved to {content_output_path}")  

        # #for style images, apply color removal and diffusion-based style removal
        # logger.info(f"Processing style image: {cfg['style_path']}")
        # style_output_path = os.path.join(run_output_path, "style_processed")
        # os.makedirs(style_output_path, exist_ok=True)
        # style_image = Image.open(cfg['style_path']).convert('RGB')
        # style_image_luma = rgb_to_luma_601(style_image)       

        # #forward diffusion ODE/inversion to obtain latents
        # options = model_and_diffusion_defaults()
        # options.update({
        #     'attention_resolutions': '32,16,8',
        #     'class_cond': False,
        #     'diffusion_steps': cfg['t_diffusion'],
        #     'image_size': cfg['image_size'],
        #     'learn_sigma': True,
        #     'noise_schedule': 'linear',
        #     'num_channels': 256,
        #     'num_head_channels': 64,
        #     'num_res_blocks': 2,
        #     'resblock_updown': True,
        #     'use_fp16': False,
        #     'use_scale_shift_norm': True,
        # })

        # model, diffusion = create_model_and_diffusion(**options)
        # state_dict = torch.load(cfg['pretrained_model_path'], map_location=cfg['device'], weights_only=True)
        # model.load_state_dict(state_dict)
        # model.eval().to(cfg['device'])

        # #convert luma image into tensor
        # x0 = prepare_image_as_tensor(Image.fromarray(style_image_luma), image_size=cfg['image_size'], device=cfg['device'])

        # #forward diffusion
        # t = torch.tensor([diffusion.num_timesteps - 1]).to(cfg['device'])
        # x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))
        
        # #reverse diffusion with fewer steps (DDIM)
        # ddim_timesteps = make_ddim_timesteps(cfg['t_diffusion'], cfg['t_remov'])
        # x0_est = ddim_reverse_deterministic(x_t, model, diffusion, ddim_timesteps, device=cfg['device'], logger=logger)

        # image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # image_recon = ((image_recon + 1) / 2).clip(0, 1)  # scale back to [0,1]
        # image_recon = (image_recon * 255).astype(np.uint8)  # [0, 255], uint8
        # Image.fromarray(image_recon).save(os.path.join(style_output_path, 'style.jpg'))
        # logger.info(f"Style image processed and saved to {style_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)