import os
import torch
import yaml
from unsloth import FastLanguageModel
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")

def load_config(config_type="training"):
    config_path = f"configs/{config_type}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

training_config = load_config("training")
inference_config = load_config("inference")

if HF_TOKEN:
    login(HF_TOKEN)

def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        training_config["model"]["base_model"],
        max_seq_length=training_config["model"]["max_seq_length"],
        dtype=training_config["model"]["dtype"],
        load_in_4bit=training_config["model"]["load_in_4bit"],
        token=HF_TOKEN
    )
    return model, tokenizer

def load_trained_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        training_config["model"]["checkpoint"],
        max_seq_length=training_config["model"]["max_seq_length"],
        dtype=training_config["model"]["dtype"],
        load_in_4bit=training_config["model"]["load_in_4bit"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer

def load_model_for_inference():
    model, tokenizer = FastLanguageModel.from_pretrained(
        inference_config["model"]["checkpoint"],
        max_seq_length=inference_config["inference"]["max_length"],
        load_in_4bit=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=HF_TOKEN
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer
