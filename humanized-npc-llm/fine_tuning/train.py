import os
import sys
import yaml
import torch
import json
from typing import Dict, List, Any
import warnings

# ============================================================================
# Torch / Dynamo configuration (fix for recompile_limit / NaNs)
# ============================================================================
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.suppress_errors = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# VERY IMPORTANT: import unsloth after Dynamo config
import unsloth
from unsloth import FastLanguageModel

from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import wandb

# If data_loader.py is in the same fine_tuning/ folder
sys.path.append('../fine_tuning')
from data_loader import create_chat_messages


# ============================================================================
# NaN Detection Callback (hard stop on NaN/Inf)
# ============================================================================

class NaNDetectionCallback(TrainerCallback):
    def __init__(self):
        self.nan_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not state.log_history:
            return control

        latest_log = state.log_history[-1]
        loss = latest_log.get("loss", None)
        if loss is None:
            return control

        if not torch.isfinite(torch.tensor(loss)):
            self.nan_count += 1

            print("\n" + "=" * 80)
            print(f"⚠️  NaN/Inf loss detected at step {state.global_step}")
            print(f"    loss = {loss}")
            print("=" * 80 + "\n")

            if wandb.run:
                wandb.log(
                    {
                        "errors/nan_loss_detected": 1,
                        "errors/nan_count": self.nan_count,
                    },
                    step=state.global_step,
                )

            # Stop immediately on first NaN/Inf
            control.should_training_stop = True
            return control

        return control


# ============================================================================
# Gradient Monitoring Callback (uses HF-logged grad_norm)
# ============================================================================

class GradientMonitorCallback(TrainerCallback):
    def __init__(self, alert_threshold: float = 50.0):
        self.alert_threshold = alert_threshold
        self.high_gradient_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not state.log_history:
            return control

        latest_log = state.log_history[-1]
        grad_norm = latest_log.get("grad_norm", None)
        if grad_norm is None:
            return control

        if grad_norm > self.alert_threshold:
            self.high_gradient_count += 1

            print("\n" + "=" * 80)
            print(f"⚠️  HIGH GRADIENT ALERT at step {state.global_step}")
            print(f"    grad_norm = {grad_norm:.2f} (threshold: {self.alert_threshold})")
            print(f"    count = {self.high_gradient_count}")
            print("=" * 80 + "\n")

            if wandb.run:
                wandb.log(
                    {
                        "warnings/high_gradient_norm": grad_norm,
                        "warnings/high_gradient_count": self.high_gradient_count,
                    },
                    step=state.global_step,
                )

        return control


# ============================================================================
# DialogueMetricsCallback (your original, with safety tweaks)
# ============================================================================

class DialogueMetricsCallback(TrainerCallback):
    """
    Custom callback for logging dialogue-specific metrics during training.
    """

    def __init__(self, tokenizer, test_prompts: List[Dict] = None):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts or self._get_default_test_prompts()
        self.generation_count = 0
        self.last_generation_step = 0  # to avoid double runs

    def _get_default_test_prompts(self) -> List[Dict]:
        return [
            {
                "name": "grumpy_blacksmith",
                "persona": "I am a grumpy blacksmith. I dislike chit-chat and prefer working with metal.",
                "world_facts": "location=Forge; items=[iron_sword,steel_hammer]",
                "query": "Good morning! How are you today?"
            },
            {
                "name": "curious_mage",
                "persona": "I am a curious mage who loves teaching magic. I speak formally and use archaic language.",
                "world_facts": "location=Tower_Library; items=[spell_tome,magic_staff]",
                "query": "Can you teach me a spell?"
            },
            {
                "name": "friendly_merchant",
                "persona": "I am a friendly merchant. I love trading and always offer fair deals.",
                "world_facts": "location=Market_Square; items=[health_potion,rope,bread]",
                "query": "What do you have for sale?"
            }
        ]

    def on_evaluate(self, args, state, control, **kwargs):
        # Run every eval and avoid duplicate calls at same step
        if state.global_step == self.last_generation_step:
            return

        self.last_generation_step = state.global_step
        model = kwargs.get('model')
        if model is None:
            return

        try:
            self._log_gradient_norms(model, state.global_step)
            self._log_gpu_memory(state.global_step)
            if self.test_prompts and state.global_step > 0:
                self._log_sample_generations(model, state.global_step)
        except Exception as e:
            print(f"⚠️  DialogueMetricsCallback error at step {state.global_step}: {e}")

    def _log_gradient_norms(self, model, step: int):
        total_norm = 0.0
        max_norm = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                param_norm = param.grad.data.norm(2).item()
                if not torch.isfinite(torch.tensor(param_norm)):
                    continue
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** 0.5
            mean_norm = total_norm / param_count

            if wandb.run:
                wandb.log(
                    {
                        "gradients/total_norm": total_norm,
                        "gradients/mean_norm": mean_norm,
                        "gradients/max_norm": max_norm,
                        "gradients/param_count": param_count,
                    },
                    step=step,
                )

    def _log_gpu_memory(self, step: int):
        if torch.cuda.is_available():
            try:
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                max_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                utilization = (allocated_gb / total_memory) * 100

                if wandb.run:
                    wandb.log(
                        {
                            "system/gpu_memory_allocated_gb": allocated_gb,
                            "system/gpu_memory_reserved_gb": reserved_gb,
                            "system/gpu_memory_max_allocated_gb": max_allocated_gb,
                            "system/gpu_memory_utilization_percent": utilization,
                        },
                        step=step,
                    )
            except Exception as e:
                print(f"⚠️  GPU memory logging error: {e}")

    def _log_sample_generations(self, model, step: int):
        model.eval()
        generation_samples = []

        with torch.no_grad():
            for prompt_data in self.test_prompts:
                try:
                    prompt_text = f"""<|im_start|>system
You are a humanized video game NPC. You must speak naturally, stay in character, and respond in valid JSON format.

<Persona>
{prompt_data['persona']}
</Persona>

<WorldFacts>
{prompt_data.get('world_facts', 'No specific context.')}
</WorldFacts>

<Rules>
- Respond with a valid JSON object: {{"utterance": "...", "mood": "..."}}
- Your "utterance" must be conversational and in-character.
- Keep responses short and natural.
</Rules>
<|im_end|>
<|im_start|>user
Player: "{prompt_data['query']}"
<|im_end|>
<|im_start|>assistant
"""

                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(model.device)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )

                    try:
                        json.loads(generated_text)
                        is_valid_json = True
                    except Exception:
                        is_valid_json = False

                    generation_samples.append(
                        {
                            "test_name": prompt_data["name"],
                            "persona": prompt_data["persona"][:60] + "...",
                            "query": prompt_data["query"],
                            "response": generated_text[:200],
                            "valid_json": is_valid_json,
                            "step": step,
                        }
                    )

                    del inputs, outputs

                except Exception as e:
                    print(f"Generation error for {prompt_data['name']}: {e}")

        if generation_samples and wandb.run:
            try:
                table = wandb.Table(
                    columns=[
                        "test_name",
                        "persona",
                        "query",
                        "response",
                        "valid_json",
                        "step",
                    ],
                    data=[
                        [
                            s["test_name"],
                            s["persona"],
                            s["query"],
                            s["response"],
                            s["valid_json"],
                            s["step"],
                        ]
                        for s in generation_samples
                    ],
                )
                wandb.log(
                    {f"generations/samples_step_{step}": table},
                    step=step,
                )

                valid_count = sum(1 for s in generation_samples if s["valid_json"])
                wandb.log(
                    {
                        "quality/json_validity_rate": valid_count
                        / len(generation_samples)
                    },
                    step=step,
                )
            except Exception as e:
                print(f"⚠️  W&B logging error: {e}")

        model.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Config / dataset helpers
# ============================================================================

def load_config(config_path: str):
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_dataset(example, tokenizer):
    """
    Build chat messages and apply the chat template.
    Includes length validation to skip extreme outliers.
    """
    try:
        messages = create_chat_messages(example)
        if messages is None:
            return {"text": None}

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if len(formatted_text) > 8192:
            print(f"⚠️  Skipping sample with length {len(formatted_text)} (too long)")
            return {"text": None}

        return {"text": formatted_text + tokenizer.eos_token}

    except Exception as e:
        print(f"⚠️  Error formatting sample: {e}")
        return {"text": None}


# ============================================================================
# MAIN
# ============================================================================

def main():
    # --- 1. Load Configuration ---
    # Restored to your original path
    config_path = "./training.yaml"
    config = load_config(config_path)

    # --- 2. W&B Setup ---
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"

    # --- 3. Load Model & Tokenizer ---
    print(f"Loading base model: {config['base_model_id']}")
    print("=" * 60)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model_id"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Configure PEFT (QLoRA) ---
    print("Configuring PEFT (QLoRA)...")
    print("=" * 60)

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    print("=" * 60 + "\n")

    # --- 5. Load and Process Datasets ---
    print("Loading and formatting datasets...")

    # Restored your v2 defaults
    train_data_file = config.get(
        "train_data_file",
        "../data_engineering/data/processed/v2/train.jsonl",
    )
    val_data_file = config.get(
        "val_data_file",
        "../data_engineering/data/processed/v2/val.jsonl",
    )

    def map_func(ex):
        return format_dataset(ex, tokenizer)

    print(f"Loading training data from: {train_data_file}")
    train_dataset = load_dataset("json", data_files=train_data_file, split="train")
    print(f"Raw training samples loaded: {len(train_dataset)}")

    train_dataset = train_dataset.map(
        map_func,
        num_proc=os.cpu_count(),
    )
    train_dataset = train_dataset.filter(lambda ex: ex["text"] is not None)

    print(f"✓ Training samples after filtering: {len(train_dataset):,}")
    print("First formatted sample preview:")
    print(train_dataset[0]["text"][:500])
    print("...\n")

    print(f"Loading validation data from: {val_data_file}")
    val_dataset = load_dataset("json", data_files=val_data_file, split="train")
    print(f"Raw validation samples loaded: {len(val_dataset)}")

    val_dataset = val_dataset.map(
        map_func,
        num_proc=os.cpu_count(),
    )
    val_dataset = val_dataset.filter(lambda ex: ex["text"] is not None)

    print(f"✓ Validation samples after filtering: {len(val_dataset):,}\n")

    # --- 6. Training steps summary ---
    effective_batch_size = (
        config["batch_size"] * config["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(1, len(train_dataset) // effective_batch_size)
    total_steps = steps_per_epoch * config["num_train_epochs"]

    print("=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Training samples:           {len(train_dataset):,}")
    print(f"Validation samples:         {len(val_dataset):,}")
    print(f"Batch size per device:      {config['batch_size']}")
    print(f"Gradient accumulation:      {config['gradient_accumulation_steps']}")
    print(f"Effective batch size:       {effective_batch_size}")
    print(f"Number of epochs:           {config['num_train_epochs']}")
    print(f"Steps per epoch:            {steps_per_epoch:,}")
    print(f"Total training steps:       {total_steps:,}")
    print(f"Eval frequency:             Every {config['eval_steps']} steps")
    print(f"Save frequency:             Every {config['save_steps']} steps")
    print("=" * 60 + "\n")

    # --- 7. TrainingArguments & Trainer ---
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
        metric_for_best_model="eval_loss",

        report_to="wandb",
        run_name=config["wandb_run_name"],

        optim=config.get("optimizer", "paged_adamw_8bit"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),

        seed=42,
        logging_first_step=True,
        disable_tqdm=False,
        include_inputs_for_metrics=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
    )

    nan_detector = NaNDetectionCallback()
    gradient_monitor = GradientMonitorCallback(alert_threshold=50.0)
    dialogue_callback = DialogueMetricsCallback(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
        packing=True,  # efficiency boost
        callbacks=[nan_detector, gradient_monitor, dialogue_callback],
    )

    # --- 8. Extra W&B metadata ---
    if wandb.run:
        wandb.config.update(
            {
                "model_architecture": "Qwen2.5-3B-Instruct",
                "training_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "effective_batch_size": effective_batch_size,
                "total_steps": total_steps,
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU",
                "peft_method": "QLoRA",
                "target_task": "persona-consistent NPC dialogue",
            }
        )

    # --- 9. Train with emergency checkpoint on error ---
    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print("=" * 60 + "\n")

    try:
        trainer.train()
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETE")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user. Saving current state...")
    except Exception as e:
        print(f"\n🛑 TRAINING ERROR: {e}")
        print("\nAttempting to save emergency checkpoint...")

        try:
            emergency_path = os.path.join(config["output_dir"], "emergency_checkpoint")
            os.makedirs(emergency_path, exist_ok=True)
            model.save_pretrained(emergency_path)
            tokenizer.save_pretrained(emergency_path)
            print(f"✅ Emergency checkpoint saved to: {emergency_path}")
        except Exception as save_error:
            print(f"❌ Could not save emergency checkpoint: {save_error}")
        raise

    # --- 10. Save final model ---
    final_model_path = os.path.join(config["output_dir"], "final_model")
    print(f"\nSaving final model adapters to {final_model_path}")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    config_snapshot_path = os.path.join(final_model_path, "training_config_snapshot.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f)
    print(f"✓ Saved training config to {config_snapshot_path}")

    print("\n" + "=" * 60)
    print("FINE-TUNING SUMMARY")
    print("=" * 60)
    print(f"Model:                  {config['base_model_id']}")
    print(f"Training samples:       {len(train_dataset):,}")
    print(f"Epochs completed:       {config['num_train_epochs']}")
    print(f"Total steps:            {total_steps:,}")
    print(f"Final model saved to:   {final_model_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()