import yaml
from datasets import load_dataset


with open("configs/training.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("configs/prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)
train_prompt_style = prompts["train_prompt"]


def load_medical_dataset():        
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_subset"],
        split=config["data"]["dataset_split"]
    )

    def format_data(examples):
        formatted_texts = []
        for q, cot, ans in zip(
            examples[config["data"]["dataset_columns"]["question"]],
            examples[config["data"]["dataset_columns"]["reasoning"]],
            examples[config["data"]["dataset_columns"]["answer"]]
        ):
            text = train_prompt_style.format(q, cot, ans) 
            formatted_texts.append(text)

        return {"text": formatted_texts}

    dataset = dataset.map(format_data, batched=True)
    return dataset
