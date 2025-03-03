import yaml
from trl import SFTTrainer
from model import get_model
from utils import save_model  
from transformers import TrainingArguments
from data_loader import load_medical_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported

with open("configs/training.yaml", "r") as f:
    config = yaml.safe_load(f)

model, tokenizer = get_model(config)

model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
    random_state=config["lora"]["random_state"],
    use_rslora=config["lora"]["use_rslora"],
    loftq_config=config["lora"]["loftq_config"]
)

dataset = load_medical_dataset()

training_args = TrainingArguments(
    per_device_train_batch_size=config["training"]["batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    warmup_steps=config["training"]["warmup_steps"],
    max_steps=config["training"]["max_steps"],
    learning_rate=config["training"]["learning_rate"],
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=config["training"]["logging_steps"],
    optim=config["training"]["optimizer"],
    weight_decay=config["training"]["weight_decay"],
    lr_scheduler_type=config["training"]["lr_scheduler"],
    seed=config["training"]["seed"],
    output_dir=config["training"]["output_dir"],
    save_steps=config["training"]["checkpoint_steps"],
    report_to=config["training"]["report_to"],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config["model"]["max_seq_length"],
    dataset_num_proc=2,
    args=training_args
)

trainer_stats = trainer.train()

save_model(model, tokenizer)
