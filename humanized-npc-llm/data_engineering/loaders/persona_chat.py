# data_engineering/loaders/persona_chat.py
from datasets import load_dataset
from slugify import slugify


def load_persona_chat(split="train"):
    # Use the cleaned, true-cased split with gold replies
    ds = load_dataset("bavard/personachat_truecased", split=split)
    # Each row: persona (list[str]), history (list[str]), candidates (str)
    for i, row in enumerate(ds):
        persona = row.get("persona") or []
        history = row.get("history") or []
        gold = row.get("candidates") or None
        if not (history and gold):
            continue
        # Pair the last user turn (history[-1]) with the gold reply
        dialog = [
            {"role": "player", "text": history[-1]},
            {"role": "npc", "text": gold},
        ]
        yield {
            "id": f"pc_{slugify(f'{split}-{i}')}",
            "source": "personachat",
            "split": split,
            "persona": persona,
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {},
            "dialog": dialog,
            "meta": {
                "license": "unknown",
                "dataset_ref": "https://huggingface.co/datasets/bavard/personachat_truecased",
            },
        }