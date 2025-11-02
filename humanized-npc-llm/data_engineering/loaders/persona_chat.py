# data_engineering/loaders/persona_chat.py
from datasets import load_dataset
from slugify import slugify


def load_persona_chat(split="train"):
    """
    Load bavard/personachat_truecased from the Parquet conversion branch to avoid
    the custom builder that looks for missing JSON files on `main`.
    """
    # Try parquet conversion first (no custom code)
    ds = load_dataset(
        "bavard/personachat_truecased",
        split=split,                     # "train" or "validation"
        revision="refs/convert/parquet"  # <- key change
    )

    for i, row in enumerate(ds):
        # Be tolerant to minor column drift
        persona = row.get("persona") or row.get("personas") or []
        history = row.get("history") or row.get("dialog_history") or []
        gold = (row.get("candidates")
                or row.get("response")
                or row.get("utterance")
                or row.get("label"))

        if isinstance(gold, list):
            gold = gold[0] if gold else None

        if not (history and isinstance(gold, str) and gold.strip()):
            continue

        yield {
            "id": f"pc_{slugify(f'{split}-{i}')}",
            "source": "personachat",
            "split": split,
            "persona": persona,
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {},
            "dialog": [
                {"role": "player", "text": history[-1]},
                {"role": "npc", "text": gold.strip()},
            ],
            "meta": {
                "license": "unknown",
                "dataset_ref": "https://huggingface.co/datasets/bavard/personachat_truecased",
            },
        }