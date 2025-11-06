# """
# Data loader for training.
# Loads JSONL from Person 1 and convert to instruction format.
# """

# def load_jsonl(path):
#     """Load JSONL file."""
#     pass

# def to_instruction_format(record):
#     """Convert NPC dialogue record to instruction format."""
#     pass
import json

def create_chat_messages(record: dict) -> list | None:
    """
    Converts a single JSONL record from Task 1 into a list of
    chat messages (system, user, assistant) for the model.
    
    This is the core logic that translates your schema into a
    trainable format.
    """
    
    # --- 1. Build System Prompt ---
    persona_facts = record.get("persona", [])
    world_facts = record.get("world_facts", [])
    
    persona_str = "\n".join(f"- {fact}" for fact in persona_facts if fact)
    world_str = "\n".join(f"- {fact}" for fact in world_facts if fact)
    
    # This prompt explicitly tells the model its role, persona, and
    # the JSON output format.
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
    
    # --- 2. Process Dialogue History ---
    dialog_history = record.get("dialog", [])
    if not dialog_history:
        return None # Skip records with no dialogue

    # The last turn *must* be an NPC response, which is our target
    if dialog_history[-1]["role"] != "npc":
        return None # Skip if the last turn isn't the NPC

    # Add all preceding turns to history
    for turn in dialog_history[:-1]:
        if turn.get("role") and turn.get("text"):
            # Map "player" to "user" and "npc" to "assistant"
            role = "user" if turn["role"] == "player" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
            
    # --- 3. Create Final User/Assistant Pair ---
    
    # The last *player* turn (if it exists)
    # dialog_history[-2] is the player, dialog_history[-1] is the NPC
    if len(dialog_history) >= 2 and dialog_history[-2]["role"] == "player":
        messages.append({
            "role": "user",
            "content": dialog_history[-2]["text"]
        })
    else:
        # This handles cases (like CharacterCodex) where there is only
        # an NPC response without a preceding player turn.
        # We invent a simple "user" prompt to kick off the conversation.
        messages.append({"role": "user", "content": "Hello."})


    # --- 4. Create the Target JSON (Assistant's Response) ---
    target_turn = dialog_history[-1]
    
    # Synthesize the mood from the data
    mood = "neutral"
    if record.get("control", {}).get("mood"):
        mood = record["control"]["mood"].lower()
    elif record.get("intent", "generic") != "generic":
        mood = record.get("intent", "neutral").lower()

    # Create the target JSON string the model must learn to generate
    target_json_str = json.dumps({
        "utterance": target_turn["text"],
        "mood": mood
    })
    
    messages.append({"role": "assistant", "content": target_json_str})
    
    return messages
