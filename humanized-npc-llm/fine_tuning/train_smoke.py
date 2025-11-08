import os
import sys

import torch
# VERY IMPORTANT: import unsloth first so it can patch PyTorch/transformers
import unsloth
import weave
import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# If data_loader.py is in the same fine_tuning/ folder, this is optional but harmless
sys.path.append('../fine_tuning')
from data_loader import create_chat_messages


def load_config(config_path: str):
    """Loads the training config file."""
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_dataset(example, tokenizer):
    """
    Build chat messages for one example and apply the chat template.
    """
    messages = create_chat_messages(example)
    if messages is None:
        return {"text": None}  # will be filtered out later

    try:
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # full conversation + assistant reply already present
        )
        return {"text": formatted_text + tokenizer.eos_token}
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return {"text": None}


def main():
    # --- 1. Load Configuration ---
    config_path = "../data_engineering/config/training.yaml"
    config = load_config(config_path)

    # --- 2. W&B (optional; WANDB_API_KEY must be set in env if you use it) ---
    os.environ["WANDB_PROJECT"] = config["wandb_project"]

    # --- 3. Load Model & Tokenizer via Unsloth ---
    print(f"Loading base model: {config['base_model_id']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model_id"],
        max_seq_length=config["max_seq_length"],
        dtype=None,          # auto-select bf16/fp16
        load_in_4bit=True,   # QLoRA
    )

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Configure PEFT (QLoRA) ---
    print("Configuring PEFT (QLoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing=config["gradient_checkpointing"],
        random_state=42,
        max_seq_length=config["max_seq_length"],
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    # --- 5. Load and Process Datasets ---
    print("Loading and formatting datasets...")

    # Use paths from the config
    train_data_file = config.get("train_data_file", "../data_engineering/data/processed/classwork/train.jsonl")
    val_data_file = config.get("val_data_file", "../data_engineering/data/processed/classwork/val.jsonl")

    def map_func(ex):
        return format_dataset(ex, tokenizer)

    # Train dataset
    train_dataset = load_dataset("json", data_files=train_data_file, split="train")
    train_dataset = train_dataset.map(
        map_func,
        num_proc=os.cpu_count(),
    )
    train_dataset = train_dataset.filter(lambda ex: ex["text"] is not None)
    n = min(100, len(train_dataset))
    train_dataset = train_dataset.select(range(n))
    print("First formatted sample:", train_dataset[0]["text"][:500])

    # Validation dataset
    val_dataset = load_dataset("json", data_files=val_data_file, split="train")
    val_dataset = val_dataset.map(
        map_func,
        num_proc=os.cpu_count(),
    )
    val_dataset = val_dataset.filter(lambda ex: ex["text"] is not None)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- 6. Configure Trainer ---
    print("Setting up TrainingArguments and SFTTrainer...")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],

        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],

        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],

        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],

        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,

        report_to="wandb",
        run_name=config["wandb_run_name"],

        # Use your optimizer / scheduler / regularization settings
        optim=config.get("optimizer", "paged_adamw_8bit"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),

        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
        packing=False,  # try True later if many short samples
    )

    # --- 7. Train ---
    print("--- Starting Model Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # --- 8. Save Final Model ---
    final_model_path = os.path.join(config["output_dir"], "final_model")
    print(f"Saving final model adapters to {final_model_path}")
    os.makedirs(final_model_path, exist_ok=True)

    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save the training config for reference
    with open(os.path.join(final_model_path, "training_config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    print("--- Task 2: Fine-Tuning Complete ---")


if __name__ == "__main__":
    main()