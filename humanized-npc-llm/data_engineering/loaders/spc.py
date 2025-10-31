from datasets import load_dataset
from slugify import slugify
import re

_U1_PAT = re.compile(r"^\s*User\s*1\s*:\s*(.+)$", re.IGNORECASE)
_U2_PAT = re.compile(r"^\s*User\s*2\s*:\s*(.+)$", re.IGNORECASE)


def _to_lines(text):
    for raw in str(text).splitlines():
        t = raw.strip()
        if t:
            yield t


def _parse_conversation_block(text):
    """
    Parse 'Best Generated Conversation' text:
    Lines like 'User 1: ...' / 'User 2: ...'
    Map User 1 -> player, User 2 -> npc.
    """
    dialog = []
    for line in _to_lines(text):
        m1 = _U1_PAT.match(line)
        m2 = _U2_PAT.match(line)
        if m1:
            dialog.append({"role": "player", "text": m1.group(1).strip()})
        elif m2:
            dialog.append({"role": "npc", "text": m2.group(1).strip()})
        # else: ignore narrations that don't match
    return dialog


def _as_list(val):
    # Personas can be a single string with bullets/newlines; normalize to list[str]
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    # split on newlines/semicolons
    parts = re.split(r"[\n;]+", str(val))
    return [p.strip() for p in parts if p.strip()]


def load_spc(split="train"):
    """
    Drop-in replacement using google/Synthetic-Persona-Chat.
    Will also work with split='evaluation' (as provided on the card).
    """
    ds = load_dataset("google/Synthetic-Persona-Chat", split=split)

    # The card shows 'user 1 personas', 'user 2 personas', and
    # 'Best Generated Conversation' text we must parse.
    # Column names on HF are lowercased with spaces; handle variants robustly.
    for i, row in enumerate(ds):
        # Try multiple key variants for safety
        u1p = (
            row.get("user 1 personas")
            or row.get("User 1 Personas")
            or row.get("user_1_personas")
        )
        u2p = (
            row.get("user 2 personas")
            or row.get("User 2 Personas")
            or row.get("user_2_personas")
        )
        convo_text = (
            row.get("best generated conversation")
            or row.get("Best Generated Conversation")
            or row.get("conversation")
            or ""
        )

        dialog = _parse_conversation_block(convo_text)
        if len(dialog) < 2:
            continue  # skip very short/noisy entries

        persona = _as_list(u1p) + _as_list(u2p)

        yield {
            "id": f"spc_google_{slugify(str(i))}",
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
