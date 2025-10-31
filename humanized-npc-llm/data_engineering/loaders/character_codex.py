# data_engineering/loaders/character_codex.py
from datasets import load_dataset
from slugify import slugify


def _first_sentence(text, max_len=400):
    if not text:
        return ""
    txt = str(text).strip()
    # Prefer greeting if it's already “spoken”, otherwise trim scenario
    for ender in [". ", "? ", "! "]:
        if ender in txt:
            txt = txt.split(ender)[0] + ender.strip()
            break
    return txt[:max_len]

def load_character_codex(split="train", max_scenario_length=400):
    # CC only has a 'train' split on HF
    if split != "train":
        split = "train"

    ds = load_dataset("NousResearch/CharacterCodex", split=split)
    for i, row in enumerate(ds):
        # Some rows have top-level fields; others nest under 'character'
        card = (row.get("character") or {})
        name = card.get("name") or row.get("name") or None
        occupation = card.get("occupation") or row.get("occupation") or "generic"
        traits = card.get("traits") or row.get("traits") or []
        backstory = card.get("backstory") or row.get("backstory") or ""

        persona = []
        if name:
            persona.append(f"I am {name}")
        if occupation:
            persona.append(f"My role is {occupation}")
        if isinstance(traits, list):
            persona.extend([f"I am {t}" for t in traits[:5]])
        if backstory:
            persona.append(str(backstory)[:200])

        # Prefer a “greeting” (already phrased like dialogue). Otherwise try scenario/prompt.
        greeting = row.get("greeting") or card.get("greeting")
        scenario = row.get("scenario") or row.get("prompt") or card.get("scenario") or card.get("prompt")
        opening = (greeting or _first_sentence(scenario, max_len=max_scenario_length) or "").strip()

        if not persona or not opening:
            continue

        dialog = [
            {"role": "player", "text": "Hello."},
            {"role": "npc", "text": opening},
        ]
        style = []
        if row.get("genre"):
            style.append(str(row["genre"]).lower())

        yield {
            "id": f"cc_{slugify(str(i))}",
            "source": "charcodex",
            "split": split,
            "persona": persona[:8],
            "world_facts": [],
            "context": {"npc_role": str(occupation)},
            "intent": "greeting",
            "control": {"style": style or ["fantasy"]},
            "dialog": dialog,
            "meta": {
                "license": "apache-2.0",
                "dataset_ref": "https://huggingface.co/datasets/NousResearch/CharacterCodex",
            },
        }