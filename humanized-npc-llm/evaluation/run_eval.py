import os
import yaml
import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer
from tqdm import tqdm
import logging
import time
import sys
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

sys.path.append('../evaluation')

# Import our custom metric calculators
from metrics import (
    PersonaMetrics,
    EvaluationMetrics,
    aggregate_metrics,
    compare_models,
    get_best_and_worst_examples
)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# --- Helper Functions ---

def load_config(config_path: str) -> dict:
    """Loads the evaluation config file."""
    log.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        log.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        raise


def setup_tokenizer(tokenizer: Any) -> Any:
    """Sets up tokenizer with proper padding token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    return tokenizer


def load_models(config: dict) -> Tuple:
    """Loads the baseline and fine-tuned models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    max_seq_len = config['baseline_model']['max_seq_length']
    base_model_id = config['baseline_model']['id']
    tuned_model_path = config['data']['tuned_model_path']

    # 1. Load Baseline Model
    log.info(f"Loading baseline model: {base_model_id}")
    try:
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        base_tokenizer = setup_tokenizer(base_tokenizer)
        log.info("✓ Baseline model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load baseline model: {e}")
        raise

    # 2. Load Fine-Tuned Model
    log.info(f"Loading fine-tuned model from: {tuned_model_path}")
    try:
        tuned_model, tuned_tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,  # Start from base
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )

        # Apply LoRA adapters
        tuned_model.load_adapter(tuned_model_path)
        tuned_model.enable_adapters()
        tuned_tokenizer = setup_tokenizer(tuned_tokenizer)

        log.info("✓ Fine-tuned model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load fine-tuned model: {e}")
        raise

    return (base_model, base_tokenizer), (tuned_model, tuned_tokenizer), device


def load_test_data(config: dict) -> Any:
    """Loads and prepares the test dataset."""
    test_file = config['data']['test_file']
    log.info(f"Loading test data from: {test_file}")

    try:
        dataset = load_dataset("json", data_files=test_file, split="train")

        limit = config['evaluation'].get('limit_samples', 0)
        if limit > 0:
            log.warning(f"⚠️  Limiting evaluation to {limit} samples for testing.")
            dataset = dataset.select(range(min(limit, len(dataset))))

        log.info(f"✓ Loaded {len(dataset)} test samples.")
        return dataset

    except Exception as e:
        log.error(f"Failed to load test data: {e}")
        raise


def create_prompt_from_record(record: dict, tokenizer: Any) -> Optional[str]:
    """
    Converts a data record into a formatted prompt string for inference.
    Must match the format used in data_loader.py from Task 2.
    """
    try:
        messages = []

        # 1. System Prompt
        persona = "\n".join(record.get("persona", ["I am a standard NPC."]))
        world_facts = "\n".join(record.get("world_facts", ["No specific context."]))

        system_prompt = f"""You are a humanized video game NPC. You must speak naturally, stay in character, and respond in valid JSON format.

<Persona>
{persona}
</Persona>

<WorldFacts>
{world_facts}
</WorldFacts>

<Rules>
- Respond with a valid JSON object: {{"utterance": "...", "mood": "..."}}
- Your "utterance" must be conversational and in-character.
- Keep responses short and natural.
</Rules>
"""
        messages.append({"role": "system", "content": system_prompt})

        # 2. Player Query (first dialog turn)
        if record.get("dialog") and len(record["dialog"]) > 0:
            player_query = record["dialog"][0].get("text", "Hello.")
        else:
            player_query = "Hello."

        # Do NOT add "Player: " prefix - the template handles this
        messages.append({"role": "user", "content": player_query})

        # 3. Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    except Exception as e:
        log.error(f"Failed to create prompt for record {record.get('id')}: {e}")
        return None


@torch.no_grad()
def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_config: dict
) -> str:
    """
    Generates a single response from a model.
    """
    try:
        # Calculate safe max length
        max_input_length = model.config.max_position_embeddings - gen_config['max_new_tokens']

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
            do_sample=gen_config['do_sample'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only newly generated tokens
        response_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Clean up artifacts
        response_text = response_text.strip().replace("<|im_end|>", "").strip()

        return response_text

    except Exception as e:
        log.error(f"Generation failed: {e}")
        return ""


def save_detailed_results(
    results: List[Dict[str, Any]],
    output_file: str
) -> None:
    """Saves detailed generation results to JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        log.info(f"✓ Detailed results saved to: {output_file}")
    except Exception as e:
        log.error(f"Failed to save detailed results: {e}")


def save_report(
    report: Dict[str, Any],
    output_file: str
) -> None:
    """Saves final aggregated report to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        log.info(f"✓ Final report saved to: {output_file}")
    except Exception as e:
        log.error(f"Failed to save report: {e}")


def print_summary_table(baseline_report: dict, tuned_report: dict, comparison: dict) -> None:
    """Prints a formatted summary table to console."""

    print("\n" + "="*60)
    print("           EVALUATION REPORT SUMMARY")
    print("="*60)
    print(f"{'Metric':<30} {'Baseline':<15} {'Fine-Tuned':<15}")
    print("-"*60)

    def fmt_pct(val):
        if val is None:
            return "N/A"
        return f"{val*100:.2f}%"

    def fmt_float(val):
        if val is None:
            return "N/A"
        return f"{val:.4f}"

    # Schema & Constraints
    print(f"{'Schema Validity':<30} {fmt_pct(baseline_report.get('avg_is_valid_schema')):<15} {fmt_pct(tuned_report.get('avg_is_valid_schema')):<15}")
    print(f"{'Brief (<60 tokens)':<30} {fmt_pct(baseline_report.get('avg_is_brief')):<15} {fmt_pct(tuned_report.get('avg_is_brief')):<15}")
    print(f"{'Clean (no banlist)':<30} {fmt_pct(baseline_report.get('avg_is_clean')):<15} {fmt_pct(tuned_report.get('avg_is_clean')):<15}")

    print("-"*60)

    # Persona Faithfulness
    print(f"{'Persona Contradiction':<30} {fmt_pct(baseline_report.get('avg_persona_contradiction')):<15} {fmt_pct(tuned_report.get('avg_persona_contradiction')):<15}")
    print(f"{'Persona Similarity (max)':<30} {fmt_float(baseline_report.get('avg_persona_similarity_max')):<15} {fmt_float(tuned_report.get('avg_persona_similarity_max')):<15}")

    print("-"*60)

    # Hallucination
    print(f"{'UCR (Hallucination)':<30} {fmt_pct(baseline_report.get('avg_ucr')):<15} {fmt_pct(tuned_report.get('avg_ucr')):<15}")
    print(f"{'NEP (Grounding Precision)':<30} {fmt_pct(baseline_report.get('avg_nep')):<15} {fmt_pct(tuned_report.get('avg_nep')):<15}")

    print("-"*60)

    # Diversity
    print(f"{'Distinct-1 (Diversity)':<30} {fmt_float(baseline_report.get('diversity_distinct_1')):<15} {fmt_float(tuned_report.get('diversity_distinct_1')):<15}")
    print(f"{'Distinct-2 (Diversity)':<30} {fmt_float(baseline_report.get('diversity_distinct_2')):<15} {fmt_float(tuned_report.get('diversity_distinct_2')):<15}")
    print(f"{'Entropy':<30} {fmt_float(baseline_report.get('diversity_entropy')):<15} {fmt_float(tuned_report.get('diversity_entropy')):<15}")

    print("="*60)
    print()


def generate_examples_report(
    baseline_results: List[Dict],
    tuned_results: List[Dict],
    output_file: str,
    n: int = 3
) -> None:
    """Generates a report with best and worst examples."""

    baseline_best = get_best_and_worst_examples(
        baseline_results,
        metric_key="persona_similarity_max",
        n=n
    )

    tuned_best = get_best_and_worst_examples(
        tuned_results,
        metric_key="persona_similarity_max",
        n=n
    )

    examples_report = {
        "baseline_best_examples": baseline_best["best"],
        "baseline_worst_examples": baseline_best["worst"],
        "tuned_best_examples": tuned_best["best"],
        "tuned_worst_examples": tuned_best["worst"]
    }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(examples_report, f, indent=4)
        log.info(f"✓ Examples report saved to: {output_file}")
    except Exception as e:
        log.error(f"Failed to save examples report: {e}")


def run_evaluation(config: dict) -> None:
    """Main evaluation pipeline."""

    start_time_total = time.time()

    # --- 1. Load Models and Data ---
    log.info("="*60)
    log.info("STEP 1: Loading Models and Data")
    log.info("="*60)

    (base_model, base_tokenizer), (tuned_model, tuned_tokenizer), device = load_models(config)
    test_dataset = load_test_data(config)

    gen_config = config['evaluation']['generation_config']

    # --- 2. Initialize Metric Computers ---
    log.info("="*60)
    log.info("STEP 2: Initializing Metric Models")
    log.info("="*60)

    persona_computer = PersonaMetrics(
        nli_model_name=config['evaluation']['metrics']['nli_model'],
        embedding_model_name=config['evaluation']['metrics']['embedding_model'],
        device=device
    )

    metrics_computer = EvaluationMetrics(
        persona_metrics_computer=persona_computer,
        sim_threshold=config['evaluation']['metrics']['persona_similarity_threshold'],
        store_detailed_results=config['evaluation'].get('store_detailed_results', True)
    )

    # --- 3. Run Generation Loop ---
    log.info("="*60)
    log.info("STEP 3: Running Generation and Evaluation")
    log.info("="*60)

    baseline_results = []
    tuned_results = []
    generation_records = []

    failed_prompts = []

    start_time_gen = time.time()

    for idx, item in enumerate(tqdm(test_dataset, desc="Evaluating Test Set")):

        # Create prompts
        base_prompt = create_prompt_from_record(item, base_tokenizer)
        tuned_prompt = create_prompt_from_record(item, tuned_tokenizer)

        if not base_prompt or not tuned_prompt:
            failed_prompts.append({
                "index": idx,
                "record_id": item.get("id"),
                "reason": "Prompt creation failed"
            })
            log.warning(f"⚠️  Skipping record {item.get('id')} - prompt creation failed")
            continue

        # Generate responses
        base_response = generate_response(base_model, base_tokenizer, base_prompt, gen_config)
        tuned_response = generate_response(tuned_model, tuned_tokenizer, tuned_prompt, gen_config)

        # Compute metrics
        base_metrics = metrics_computer.compute_all(item, base_response)
        tuned_metrics = metrics_computer.compute_all(item, tuned_response)

        baseline_results.append(base_metrics)
        tuned_results.append(tuned_metrics)

        # Store combined record
        generation_record = {
            "record_id": item.get("id"),
            "source_dataset": item.get("source"),
            "persona": item.get("persona"),
            "player_query": item.get("dialog")[0].get("text") if item.get("dialog") else "",
            "baseline": {
                "response": base_response,
                "metrics": base_metrics
            },
            "tuned": {
                "response": tuned_response,
                "metrics": tuned_metrics
            }
        }
        generation_records.append(generation_record)

    end_time_gen = time.time()
    gen_time = end_time_gen - start_time_gen
    samples_per_sec = len(generation_records) / gen_time if gen_time > 0 else 0

    log.info(f"✓ Generation complete: {len(generation_records)} samples in {gen_time:.2f}s ({samples_per_sec:.2f} samples/sec)")

    if failed_prompts:
        log.warning(f"⚠️  {len(failed_prompts)} prompts failed to generate")

    # --- 4. Aggregate Metrics ---
    log.info("="*60)
    log.info("STEP 4: Aggregating Metrics")
    log.info("="*60)

    baseline_report = aggregate_metrics(baseline_results)
    tuned_report = aggregate_metrics(tuned_results)

    # --- 5. Statistical Comparison ---
    log.info("="*60)
    log.info("STEP 5: Statistical Comparison")
    log.info("="*60)

    comparison = compare_models(baseline_results, tuned_results)

    # --- 6. Create Final Report ---
    end_time_total = time.time()
    total_time = end_time_total - start_time_total

    final_report = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_samples": len(generation_records),
            "failed_prompts": len(failed_prompts),
            "total_time_seconds": total_time,
            "generation_time_seconds": gen_time,
            "samples_per_second": samples_per_sec,
            "tuned_model_path": config['data']['tuned_model_path'],
            "baseline_model_id": config['baseline_model']['id'],
            "config": config
        },
        "baseline_model_metrics": baseline_report,
        "tuned_model_metrics": tuned_report,
        "statistical_comparison": comparison,
        "failed_records": failed_prompts
    }

    # --- 7. Save Results ---
    log.info("="*60)
    log.info("STEP 6: Saving Results")
    log.info("="*60)

    # Save detailed generations
    output_gen_file = config['outputs']['generation_file']
    save_detailed_results(generation_records, output_gen_file)

    # Save final report
    output_report_file = config['outputs']['report_file']
    save_report(final_report, output_report_file)

    # Save examples report
    examples_file = config['outputs'].get('examples_file',
        output_report_file.replace('.json', '_examples.json'))
    generate_examples_report(baseline_results, tuned_results, examples_file)

    # --- 8. Print Summary ---
    log.info("="*60)
    log.info("EVALUATION COMPLETE")
    log.info("="*60)

    print_summary_table(baseline_report, tuned_report, comparison)

    log.info(f"📁 Files saved:")
    log.info(f"   - Detailed generations: {output_gen_file}")
    log.info(f"   - Final report: {output_report_file}")
    log.info(f"   - Examples: {examples_file}")

    log.info(f"✓✓✓ All done! Total time: {total_time:.2f}s ✓✓✓")

if __name__ == "__main__":
    # Load config and run evaluation
    config_path = "eval.yaml"

    try:
        config = load_config(config_path)
        run_evaluation(config)
    except Exception as e:
        log.error(f"❌ Evaluation failed: {e}")
        raise