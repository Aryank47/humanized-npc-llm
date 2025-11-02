# # data_engineering/loaders/skyrim_il.py
# import re

# import requests
# from selectolax.parser import HTMLParser
# from slugify import slugify

# SPEAKER_RE = re.compile(r"^\s*(.+?)\s*[:：]\s*(.+)$", re.UNICODE)
# STAGE_RE   = re.compile(r"\[[^\]]+\]")  # [stage directions]

# PLAYER_ALIASES = {"player", "dragonborn", "dovahkiin"}

# def _valid_speaker(name: str) -> bool:
#     name = (name or "").strip()
#     return bool(name) and len(name) < 50 and any(c.isalpha() for c in name.lower())

# def _role_for(speaker: str) -> str:
#     return "player" if (speaker or "").strip().lower() in PLAYER_ALIASES else "npc"

# def _clean(text: str) -> str:
#     if not isinstance(text, str): return ""
#     text = STAGE_RE.sub("", text).strip()
#     return re.sub(r"\s+", " ", text)

# def fetch_text(url, timeout=30):
#     r = requests.get(url, timeout=timeout, headers={"User-Agent": "npc-pipeline/1.0"})
#     r.raise_for_status()
#     ctype = r.headers.get("Content-Type", "")
#     return r.text, ctype.lower()

# def _parse_plain_text_dump(txt: str, max_turns=20):
#     """Parse lines like 'Speaker: utterance' from a .txt dump."""
#     turns = []
#     for raw in txt.splitlines():
#         line = raw.strip()
#         if not line or line.startswith("#"):  # allow commented dumps
#             continue
#         m = SPEAKER_RE.match(line)
#         if not m:
#             continue
#         speaker, spoken = m.group(1), _clean(m.group(2))
#         if not (_valid_speaker(speaker) and spoken):
#             continue
#         turns.append({"role": _role_for(speaker), "text": spoken})
#         if len(turns) >= max_turns:
#             break
#     if len(turns) >= 2:
#         yield {
#             "id": f"sky_dump_{slugify(turns[0]['text'][:40])}",
#             "source": "skyrim",
#             "split": "ablation_skyrim",
#             "persona": [],
#             "world_facts": [],
#             "context": {},
#             "intent": "generic",
#             "control": {"style": ["fantasy"]},
#             "dialog": turns[:max_turns],
#             "meta": {"license": "proprietary-transcript", "dataset_ref": "textdump"},
#         }

# def _parse_html_transcript(html: str, max_turns=20):
#     """
#     Handle Imperial Library-style pages that use headings (e.g., <h6>Speaker)
#     followed by paragraphs, as well as inline 'Speaker: line' occurrences.
#     """
#     h = HTMLParser(html)
#     turns = []
#     current_speaker = None

#     for node in h.css("h6, p, li"):
#         tag = getattr(node, "tag", "").lower()

#         if tag == "h6":
#             sp = _clean(node.text(strip=True))
#             current_speaker = sp if _valid_speaker(sp) else None
#             continue

#         # Paragraph/list item
#         text = _clean(node.text(separator=" ", strip=True))
#         if not text:
#             continue

#         # Prefer explicit 'Speaker: line' if present
#         m = SPEAKER_RE.match(text)
#         if m and _valid_speaker(m.group(1)):
#             current_speaker = m.group(1).strip()
#             spoken = _clean(m.group(2))
#         else:
#             # If we have a heading-selected speaker, use it
#             if current_speaker:
#                 spoken = text
#             else:
#                 # No heading and no explicit 'Speaker:' → skip
#                 continue

#         if not spoken:
#             continue

#         turns.append({"role": _role_for(current_speaker), "text": spoken})
#         if len(turns) >= max_turns:
#             break

#     if len(turns) >= 2:
#         yield {
#             "id": f"sky_{slugify(turns[0]['text'][:40])}",
#             "source": "skyrim",
#             "split": "ablation_skyrim",
#             "persona": [],
#             "world_facts": [],
#             "context": {},
#             "intent": "generic",
#             "control": {"style": ["fantasy"]},
#             "dialog": turns[:max_turns],
#             "meta": {"license": "proprietary-transcript", "dataset_ref": "html"},
#         }

# def parse_skyrim_page(url, max_turns=20):
#     txt, ctype = fetch_text(url)
#     if any(ext in url.lower() for ext in (".txt", ".tsv", ".csv")) or "text/plain" in ctype:
#         yield from _parse_plain_text_dump(txt, max_turns=max_turns)
#     else:
#         yield from _parse_html_transcript(txt, max_turns=max_turns)


import csv
import pathlib
import re

from slugify import slugify

_WHITESPACE = re.compile(r"\s+")
_BRACKETED = re.compile(r"^\s*\[.+\]\s*$")  # skip pure stage directions like [Angry]


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # convert literal \n to spaces, collapse whitespace
    s = s.replace("\\n", " ").replace("\r", " ").strip()
    return _WHITESPACE.sub(" ", s)


def load_imperial_skyrim_txt(path="external/skyrim.txt", max_examples=None):
    """
    Parse Imperial Library Skyrim dump lines like:
    FormID: 0401AAEF\t...\tCUST\t0\tI feel nothing but sympathy...
    We treat each line as a single NPC utterance.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return

    n = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or not raw.startswith("FormID:"):
                continue
            parts = raw.split("\t")
            # text is usually the last non-empty column
            text = ""
            for t in reversed(parts):
                if t and not t.startswith("FormID:"):
                    text = t
                    break
            text = _norm(text)
            if not text or _BRACKETED.match(text):
                continue

            # ID built from FormID and trailing index if present
            formid = parts[0].split(":", 1)[-1].strip()
            idx = None
            # try to pick plausible integer column near the end for stability
            for cand in reversed(parts[:-1]):
                if cand.isdigit():
                    idx = cand
                    break
            rec_id = f"skyil_{slugify(formid)}_{idx or 'x'}_{n}"

            yield {
                "id": rec_id,
                "source": "skyrim_il",
                "split": "train",
                "persona": [],
                "world_facts": [],
                "context": {"npc_role": "Skyrim NPC"},
                "intent": "generic",
                "control": {"style": ["fantasy"]},
                "dialog": [{"role": "npc", "text": text}],
                "meta": {
                    "license": "imperial-library-export",
                    "dataset_ref": "https://www.imperial-library.info/out-of-game/game-data",
                    "file": str(p),
                },
            }
            n += 1
            if max_examples and n >= max_examples:
                break


def load_imperial_eso_csv(path="external/es.csv", max_examples=None):
    """
    Parse ESO_Strings CSV (Imperial Library) with a 'Text' column.
    Generates single-turn NPC utterances.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return

    # Robust delimiter sniff
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        n = 0
        for row in reader:
            text = _norm(row.get("Text") or row.get("text") or "")
            if not text or _BRACKETED.match(text):
                continue
            rec_id = f"eso_{slugify(row.get('ID') or n)}"
            yield {
                "id": rec_id,
                "source": "eso",
                "split": "train",
                "persona": [],
                "world_facts": [],
                "context": {"npc_role": "ESO NPC"},
                "intent": "generic",
                "control": {"style": ["fantasy"]},
                "dialog": [{"role": "npc", "text": text}],
                "meta": {
                    "license": "imperial-library-export",
                    "dataset_ref": "https://www.imperial-library.info/out-of-game/game-data",
                    "file": str(p),
                },
            }
            n += 1
            if max_examples and n >= max_examples:
                break
