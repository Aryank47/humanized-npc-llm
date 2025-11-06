#!/usr/bin/env python3
# """
# Inference script for generating NPC responses.
# """

# def main():
#     print("Inference script - TODO: Implement")
#     print("Usage: python inference.py --prompt 'Hell, blacksmith!'")

# if __name__ == "__main__":
#     main()

import torch
import json
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# --- Configuration ---
# Point this to the final_model directory saved by train.py
MODEL_PATH = "./outputs/model/" 
MAX_NEW_TOKENS = 70 # Needs to be > 60 tokens from plan

def load_model_for_inference(model_path):
    """Loads the fine-tuned adapters for inference."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048, # Must match training
        dtype = None,
        load_in_4bit = True,
    )
    
    # Ensure padding token is set for batch generation (if needed)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def build_inference_prompt(persona: list, world_facts: list, history: list):
    """
    Builds the system prompt and history for an inference request.
    
    - persona: List of persona strings
    - world_facts: List of world fact strings
    - history: List of {"role": "user/assistant", "content": "..."} dicts
    """
    
    persona_str = "\n".join(f"- {fact}" for fact in persona if fact)
    world_str = "\n".join(f"- {fact}"for fact in world_facts if fact)
    
    system_prompt = f"""You are a humanized video game NPC. You must speak naturally, stay in character, and respond in valid JSON format.

<Persona>
{persona_str if persona_str else "I am a simple character."}
</Persona>

<WorldFacts>
{world_str if world_str else "I know only what I see."}
</WorldFacts>

<Rules>
- Respond with a valid JSON object: {{"utterance": "...", "mood": "..."}}
- Your "utterance" must be conversational and in-character.
- Your "mood" should reflect the tone of your utterance (e.g., "curious", "grumpy", "helpful").
- Keep responses short and natural (under 60 tokens).
</Rules>
"""
    
    messages = [{"role": "system", "content": system_prompt.strip()}]
    
    # Add the provided history
    messages.extend(history)
    
    return messages
    
def generate_response(model, tokenizer, messages):
    """
    Generates a single JSON response from the model.
    """
    # Apply the chat template for inference
    # CRITICAL: add_generation_prompt=True tells the model it's
    # its turn to speak.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True # IMPORTANT
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    # Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens = MAX_NEW_TOKENS,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.9,
    )
    
    # Decode only the *newly generated* tokens
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response_text

def parse_json_output(text: str) -> dict:
    """Safely parses the model's JSON output."""
    try:
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
            
        json_str = text[start:end+1]
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON Parse Error: {e}\nRaw output: {text}")
        return {"utterance": text, "mood": "parse_error"}

def main():
    """Main function to run an inference example."""
    
    model, tokenizer = load_model_for_inference(MODEL_PATH)
    FastLanguageModel.for_inference(model) # Optimize for inference
    
    # --- Example 1: Blacksmith (from CharacterCodex) ---
    print("\n--- Example 1: Blacksmith ---")
    persona_1 = [
        "I am Thorgrim Ironforge",
        "My role is blacksmith",
        "I am stubborn",
        "I am skilled in metalwork",
        "I distrust elves"
    ]
    world_facts_1 = ["location: Riverside Forge", "era: Medieval fantasy"]
    history_1 = [
        {"role": "user", "content": "Can you repair my sword?"}
    ]

    messages_1 = build_inference_prompt(persona_1, world_facts_1, history_1)
    
    raw_output_1 = generate_response(model, tokenizer, messages_1)
    parsed_output_1 = parse_json_output(raw_output_1)
    
    print(f"Player: Can you repair my sword?")
    print(f"NPC (JSON): {json.dumps(parsed_output_1, indent=2)}")

    # --- Example 2: Grumpy Guard (adding to history) ---
    print("\n--- Example 2: Grumpy Guard ---")
    persona_2 = [
        "I am a guard in Rivertown",
        "I am grumpy",
        "I am suspicious of outsiders"
    ]
    world_facts_2 = ["location: Rivertown Gate", "time_of_day: Night"]
    history_2 = [
        {"role": "user", "content": "Excuse me, sir. Is the gate open?"}
    ]

    messages_2 = build_inference_prompt(persona_2, world_facts_2, history_2)
    
    raw_output_2 = generate_response(model, tokenizer, messages_2)
    parsed_output_2 = parse_json_output(raw_output_2)
    
    print(f"Player: Excuse me, sir. Is the gate open?")
    print(f"NPC (JSON): {json.dumps(parsed_output_2, indent=2)}")
    
    # --- Example 3: Follow-up question ---
    print("\n--- Example 3: Follow-up ---")
    
    # Add the previous turns to the history
    history_2.append({"role": "assistant", "content": json.dumps(parsed_output_2)})
    history_2.append({"role": "user", "content": "Why not? I'm not a threat."})
    
    messages_3 = build_inference_prompt(persona_2, world_facts_2, history_2)
    
    raw_output_3 = generate_response(model, tokenizer, messages_3)
    parsed_output_3 = parse_json_output(raw_output_3)

    print(f"Player: Why not? I'm not a threat.")
    print(f"NPC (JSON): {json.dumps(parsed_output_3, indent=2)}")


if __name__ == "__main__":
    main()

