from datasets import load_dataset
from slugify import slugify
from collections import defaultdict


def load_ed(split="train"):
    ds = load_dataset("facebook/empathetic_dialogues", split=split)
    convos = defaultdict(list)
    for row in ds:
        convos[row["conv_id"]].append(row)

    for conv_id, turns in convos.items():
        turns = sorted(turns, key=lambda x: x["utterance_idx"])
        dialog = []
        for t in turns:
            role = "player" if int(t["speaker_idx"]) == 0 else "npc"
            dialog.append({"role": role, "text": t["utterance"]})
        if not dialog:
            continue
        mood = turns[0].get("context", "Neutral") or "Neutral"
        yield {
            "id": f"ed_{slugify(conv_id)}",
            "source": "ed",
            "split": split,
            "persona": [],
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {"mood": mood},
            "dialog": dialog,
            "meta": {
                "license": "cc-by-nc-4.0",
                "dataset_ref": "https://huggingface.co/datasets/facebook/empathetic_dialogues",
            },
        }
