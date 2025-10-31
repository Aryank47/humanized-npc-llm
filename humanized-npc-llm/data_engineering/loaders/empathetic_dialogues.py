from datasets import load_dataset
from slugify import slugify
from collections import defaultdict

def _norm_ed(s: str) -> str:
    if not isinstance(s, str): return ""
    return (s.replace("_comma_", ",")
             .replace("_period_", ".")
             .replace("_exclamation_", "!")
             .replace("_question_", "?")
             .replace(" ,", ",")
             .replace(" .", ".")
             .strip())

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
            dialog.append({"role": role, "text": _norm_ed(t.get("utterance", ""))})
        if not dialog:
            continue
        if len({d["role"] for d in dialog}) == 1:
            dialog = [
                {"role": ("player" if i % 2 == 0 else "npc"), "text": d["text"]}
                for i, d in enumerate(dialog)
            ]
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
