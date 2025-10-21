import argparse
import yaml
import logging
import os

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
        logger.info("Starting style removal...")

        #for content images, apply color removal only
        content_output_path = os.path.join(run_output_path, "content_processed")
        os.makedirs(content_output_path, exist_ok=True)
        for image_file in os.listdir(cfg['content_path']):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                logger.info(f"Processing content image: {image_file}")
                image = Image.open(os.path.join(cfg['content_path'], image_file)).convert('RGB')
                image_luma = rgb_to_luma_601(image)
                Image.fromarray(image_luma).save(os.path.join(content_output_path, image_file))
                logger.info(f"{image_file} processed and saved to {content_output_path}")  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)