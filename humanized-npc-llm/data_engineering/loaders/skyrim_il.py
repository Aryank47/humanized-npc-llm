import requests, re
from selectolax.parser import HTMLParser
from slugify import slugify

SPEAKER_RE = re.compile(r"^\s*(.+?)\s*[:：]\s*(.+)$", re.UNICODE)


def _valid_speaker(name: str) -> bool:
    name = (name or "").strip()
    return bool(name) and len(name) < 50 and any(c.isalpha() for c in name.lower())


def fetch_text(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "npc-pipeline/1.0"})
    r.raise_for_status()
    return r.text


def parse_skyrim_page(url, max_turns=20):
    html = HTMLParser(fetch_text(url))
    nodes = [
        n.text(separator="\n").strip() for n in html.css("p,li") if n.text(strip=True)
    ]
    turns = []
    for line in nodes:
        m = SPEAKER_RE.match(line)
        if not m:
            continue
        speaker, spoken = m.group(1), m.group(2)
        if not _valid_speaker(speaker):
            continue
        role = (
            "player"
            if speaker.lower() in {"player", "dragonborn", "dovahkiin"}
            else "npc"
        )
        turns.append({"role": role, "text": spoken})

    if len(turns) >= 2:
        yield {
            "id": f"sky_{slugify(url)}",
            "source": "skyrim",
            "split": "ablation_skyrim",
            "persona": [],
            "world_facts": [],
            "context": {},
            "intent": "generic",
            "control": {"style": ["fantasy"]},
            "dialog": turns[:max_turns],
            "meta": {"license": "proprietary-transcript", "dataset_ref": url},
        }
