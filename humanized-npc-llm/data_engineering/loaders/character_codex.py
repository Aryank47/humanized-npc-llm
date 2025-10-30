from datasets import load_dataset
from slugify import slugify


def _first_sentence(text, max_len=400):
    if not text:
        return ""
    txt = str(text).strip()
    # Split on sentence enders to avoid mid-sentence truncation
    for ender in [". ", "? ", "! "]:
        if ender in txt:
            txt = txt.split(ender)[0] + ender.strip()
            break
    return txt[:max_len]


def load_character_codex(split="train", max_scenario_length=400):
    ds = load_dataset("NousResearch/CharacterCodex", split=split)
    for i, row in enumerate(ds):
        card = row.get("character", {}) or {}
        persona = []
        if card.get("name"):
            persona.append(f"I am {card['name']}")
        if card.get("occupation"):
            persona.append(f"My role is {card['occupation']}")
        traits = card.get("traits") or []
        if isinstance(traits, list):
            persona.extend([f"I am {t}" for t in traits[:5]])
        if card.get("backstory"):
            persona.append(card["backstory"][:200])

        scen = row.get("scenario") or row.get("greeting") or row.get("prompt") or ""
        scen_sent = _first_sentence(scen, max_len=max_scenario_length)

        if not persona or not scen_sent:
            continue

        dialog = [
            {"role": "player", "text": "Hello."},
            {"role": "npc", "text": scen_sent},
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
            "context": {"npc_role": card.get("occupation", "generic")},
            "intent": "greeting",
            "control": {"style": style or ["fantasy"]},
            "dialog": dialog,
            "meta": {
                "license": "apache-2.0",
                "dataset_ref": "https://huggingface.co/datasets/NousResearch/CharacterCodex",
            },
        }
