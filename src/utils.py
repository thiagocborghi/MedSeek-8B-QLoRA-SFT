import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("configs/training.yaml", "r") as f:
    config = yaml.safe_load(f)

def save_model(model, tokenizer):
    save_path = config["saving"]["save_dir"]
    os.makedirs(save_path, exist_ok=True)

    if config["saving"]["hf_format"]:
        logger.info(f"Saving model in Hugging Face format: {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    if config["saving"]["merged_format"]:
        merged_path = os.path.join(save_path, "merged_16bit")
        logger.info(f"Saving merged 16-bit model: {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    logger.info("Model saved successfully.")
