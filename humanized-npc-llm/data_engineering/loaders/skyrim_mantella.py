import json
import pathlib

from slugify import slugify


def load_skyrim_mantella(json_path, max_examples=None):
    """
    Consume Mantella's Alpaca-style Skyrim dataset JSON:
      [{ "instruction": "...", "input": "...", "output": "..." }, ...]
    We map (player,input)->npc(output).
    """
    p = pathlib.Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    n = 0
    for i, ex in enumerate(data):
      instr = (ex.get("instruction") or "").strip()
      inpt  = (ex.get("input") or "").strip()
      out   = (ex.get("output") or "").strip()
      if not out:
          continue

      # Prefer 'input' as the player's utterance; otherwise fall back to instruction
      player_text = inpt or instr or "Hello."
      dialog = [
          {"role": "player", "text": player_text},
          {"role": "npc",    "text": out},
      ]
      yield {
          "id": f"sky_mantella_{slugify(str(i))}",
          "source": "skyrim",
          "split": "ablation_skyrim",
          "persona": [],
          "world_facts": [],
          "context": {"npc_role": "generic", "location": "skyrim"},
          "intent": "generic",
          "control": {"style": ["fantasy"]},
          "dialog": dialog,
          "meta": {
              "license": "unknown",
              "dataset_ref": "https://github.com/art-from-the-machine/Mantella-LLM-Fine-Tuning",
          },
      }
      n += 1
      if max_examples and n >= max_examples:
          break