from datasets import load_dataset
from slugify import slugify


def load_spc(split="train"):
    ds = load_dataset("google/Synthetic-Persona-Chat", split=split)  # HF ID
    for i, row in enumerate(ds):
        persona = row.get("persona", [])
        dialog = []
        for t in row.get("dialog", []):
            role = "player" if t.get("speaker", "user") in ["user", "0", 0] else "npc"
            dialog.append({"role": role, "text": t["text"]})
        if not dialog:
            continue
        yield {
            "id": f"spc_{slugify(str(i))}",
            "source": "spc",
            "split": split,
            "persona": persona,
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {},
            "dialog": dialog,
            "meta": {
                "license": "unknown",
                "dataset_ref": "https://huggingface.co/datasets/google/Synthetic-Persona-Chat",
            },
        }
