import yaml
from model import load_model_for_inference


with open("configs/inference.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("configs/prompts.yaml", "r") as f:
    prompts_config = yaml.safe_load(f)
    

inference_prompt_template = prompts_config["inference_prompt_template"]

model, tokenizer = load_model_for_inference()

def generate_text(prompts):
    device = config["inference"]["device"]
    
    if isinstance(prompts, str):
        prompts = [prompts]

    formatted_prompts = [inference_prompt_template.format(prompt) for prompt in prompts]

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=config["inference"]["max_length"],
        temperature=config["inference"]["temperature"],
        top_p=config["inference"]["top_p"],
        top_k=config["inference"]["top_k"],
        use_cache=True
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
