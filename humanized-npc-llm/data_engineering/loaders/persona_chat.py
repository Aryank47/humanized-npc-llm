from datasets import load_dataset
from slugify import slugify


def load_persona_chat(split="train"):
    # Mirror that exposes personality + utterances[history, utterance]
    ds = load_dataset("awsaf49/persona-chat", split=split)
    for i, row in enumerate(ds):
        persona = row.get("personality") or []
        utts = []
        for u in row.get("utterances", []):
            hist = u.get("history") or []
            gold = u.get("utterance") or None
            if hist and gold:
                # Pair last user turn with the gold reply
                utts.append({"role": "player", "text": hist[-1]})
                utts.append({"role": "npc", "text": gold})
        if not utts:
            continue
        yield {
            "id": f"pc_{slugify(str(i))}",
            "source": "personachat",
            "split": split,
            "persona": persona,
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {},
            "dialog": utts,
            "meta": {
                "license": "unknown",
                "dataset_ref": "https://huggingface.co/datasets/awsaf49/persona-chat",
            },
        }
