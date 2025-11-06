#!/usr/bin/env python3
# """
# Main training script for fine-tuning with QLoRA.
# """

# def main():
#     print("Training script - TODO: Implement")
#     print("1. Load data from ../1_data_engineering/outputs/")
#     print("2. Convert to instruction format")
#     print("3. Setup QLoRA")
#     print("4. Train model")
#     print("5. Save adapters")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# """
# Main training script for fine-tuning with QLoRA.
# """

# def main():
#     print("Training script - TODO: Implement")
#     print("1. Load data from ../1_data_engineering/outputs/")
#     print("2. Convert to instruction format")
#     print("3. Setup QLoRA")
#     print("4. Train model")
#     print("5. Save adapters")

# if __name__ == "__main__":
#     main()
import torch
import yaml
import os
import json
import sys

# Add the directory containing data_loader.py to the Python path
sys.path.append('../fine_tuning')


from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Import our custom data loading logic
from data_loader import create_chat_messages

def load_config(config_path):
    """Loads the training config file."""
    print(f"Loading configuration from {config_path}...")
    # Load the configuration directly from the provided Google Drive path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_tokenizer(model_name):
    """Loads tokenizer and sets padding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def format_dataset(example, tokenizer):
    """
    Applies the chat message creation and tokenization
    for a single example.
    """
    # 1. Create the list of messages
    messages = create_chat_messages(example)
    if messages is None:
        return {"text": None} # This will be filtered out

    # 2. Apply the chat template
    # We set add_generation_prompt=False as we provide the full conversation
    # including the assistant's response.
    try:
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Add EOS token to signal end of turn
        return {"text": formatted_text + tokenizer.eos_token}
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return {"text": None}


def main():
    # --- 1. Load Configuration ---
    # Use the provided path for the configuration file
    config = load_config("../data_engineering/config/training.yaml")


    # Set up W&B (Weights & Biases) for logging
    os.environ["WANDB_PROJECT"] = config['wandb_project']

    # --- 2. Load Model & Tokenizer (with Unsloth) ---
    print(f"Loading base model: {config['base_model_id']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['base_model_id'],
        max_seq_length = config['max_seq_length'],
        dtype = None,           # Auto-detect (bf16 or fp16)
        load_in_4bit = True,    # Enable QLoRA
    )

    # --- 3. Configure PEFT (QLoRA) ---
    print("Configuring PEFT (QLoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora_r'],
        lora_alpha = config['lora_alpha'],
        lora_dropout = config['lora_dropout'],
        target_modules = config['lora_target_modules'],
        bias = "none",
        use_gradient_checkpointing = config['gradient_checkpointing'],
        random_state = 42,
        max_seq_length = config['max_seq_length'],
    )

    # --- 4. Load and Process Datasets ---
    print(f"Loading and formatting datasets...")
    tokenizer = get_tokenizer(config['base_model_id'])

    # Create a partial function for mapping
    format_func = lambda ex: format_dataset(ex, tokenizer)

    # Adjust the data file paths to load from Google Drive
    train_data_file = "../data_engineering/outputs/train.jsonl"
    val_data_file = "../data_engineering/outputs/val.jsonl"
    # Load and process train dataset
    train_dataset = load_dataset("json", data_files=train_data_file, split="train")
    train_dataset = train_dataset.map(format_func, num_proc=os.cpu_count())
    train_dataset = train_dataset.filter(lambda ex: ex['text'] is not None)

    # Load and process validation dataset
    val_dataset = load_dataset("json", data_files=val_data_file, split="train") # HF loads jsonl as 'train' split
    val_dataset = val_dataset.map(format_func, num_proc=os.cpu_count())
    val_dataset = val_dataset.filter(lambda ex: ex['text'] is not None)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- 5. Configure Trainer ---
    print("Setting up TrainingArguments and SFTTrainer...")

    training_args = TrainingArguments(
        output_dir = config['output_dir'], # This path might also need adjustment if saving to Drive
        per_device_train_batch_size = config['batch_size'],
        per_device_eval_batch_size = config['batch_size'],
        gradient_accumulation_steps = config['gradient_accumulation_steps'],

        num_train_epochs = config['num_train_epochs'],
        learning_rate = config['learning_rate'],

        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),

        logging_steps = config['logging_steps'],
        eval_strategy = "steps",
        eval_steps = config['eval_steps'],
        save_strategy = "steps",
        save_steps = config['save_steps'],

        save_total_limit = config['save_total_limit'],
        load_best_model_at_end = True,

        report_to = "wandb",
        run_name = config['wandb_run_name'],

        # Removed optimizer and related args as they are not direct args in this version
        # optimizer = config['optimizer'],
        # lr_scheduler_type = config['lr_scheduler_type'],
        # warmup_ratio = config['warmup_ratio'],
        # weight_decay = config['weight_decay'],
        # max_grad_norm = config['max_grad_norm'],

        seed = 42,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text", # Use the "text" column we created
        max_seq_length = config['max_seq_length'],
        args = training_args,
        packing = False, # Set to True for speedup if you have many short samples
    )

    # --- 6. Start Training ---
    print("--- Starting Model Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # --- 7. Save Final Model ---
    final_model_path = f"{config['output_dir'].replace('../', '', 1)}/final_model"
    print(f"Saving final model adapters to {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save the training config for reference
    with open(os.path.join(final_model_path, "training_config_snapshot.yaml"), 'w') as f:
        yaml.dump(config, f)

    print("--- Task 2: Fine-Tuning Complete ---")

if __name__ == "__main__":
    main()