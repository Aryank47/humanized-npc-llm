# data_engineering/loaders/npc_dialogue.py
import re

from datasets import load_dataset
from slugify import slugify


def _norm(s):
    return str(s).strip() if s is not None else ""


def _get_ci(row, *candidates, default=""):
    # case-insensitive getter over HF row dict
    for k in candidates:
        if k in row:
            return row[k]
    low = {rk.lower(): rk for rk in row.keys()}
    for k in candidates:
        rk = low.get(k.lower())
        if rk is not None:
            return row[rk]
    return default


def load_npc_dialogue(split="train", max_examples=None):
    """
    Hugging Face dataset: amaydle/npc-dialogue
    Actual columns (capitalized): Name, Biography, Query, Response, Emotion
    https://huggingface.co/datasets/amaydle/npc-dialogue
    """
    try:
        ds = load_dataset("amaydle/npc-dialogue", split=split)
    except Exception:
        ds = load_dataset("amaydle/npc-dialogue", split="train")

    n = 0
    for i, row in enumerate(ds):
        name = _norm(_get_ci(row, "Name"))
        bio = _norm(_get_ci(row, "Biography", "bio"))
        query = _norm(_get_ci(row, "Query", "context"))
        response = _norm(_get_ci(row, "Response", "response"))
        emotion = _norm(_get_ci(row, "Emotion", "emotion")) or "neutral"

        if not response:
            continue

        dialog = [
            {"role": "player", "text": (query or "Hello.")},
            {"role": "npc", "text": response},
        ]

        persona_list = []
        if name:
            persona_list.append(name)
        if bio:
            # lightly chunk biography into short persona traits (up to 3)
            for piece in [p.strip() for p in re.split(r"[.;|\n]+", bio) if p.strip()][
                :3
            ]:
                persona_list.append(piece)

        yield {
            "id": f"npcd_{slugify(str(i))}",
            "source": "npcd",
            "split": split,
            "persona": persona_list,
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {"mood": emotion.lower()},
            "dialog": dialog,
            "meta": {
                "license": "unknown",
                "dataset_ref": "https://huggingface.co/datasets/amaydle/npc-dialogue",
            },
        }
        n += 1
        if max_examples and n >= max_examples:
            break
