import argparse
import yaml
import logging
import os



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
    logger.info(f"run parameters: {cfg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)