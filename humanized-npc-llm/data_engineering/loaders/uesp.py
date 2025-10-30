import requests
from selectolax.parser import HTMLParser


def fetch_uesp_facts(url, max_facts=3):
    r = requests.get(url, timeout=30, headers={"User-Agent": "npc-pipeline/1.0"})
    r.raise_for_status()
    h = HTMLParser(r.text)
    paras = [
        p.text(strip=True)
        for p in h.css("div.mw-parser-output > p")
        if p.text(strip=True)
    ]
    facts = []
    for p in paras:
        if len(p) > 40:
            facts.append(p[:200])
        if len(facts) >= max_facts:
            break
    return facts
