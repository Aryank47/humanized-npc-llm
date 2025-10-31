import yaml, pathlib, logging
from tqdm import tqdm
from data_engineering.schema_tools import load_schema, validate_record
from data_engineering.io_utils import write_jsonl, sha256_file
from data_engineering.mix_tools import (
    allocate_counts,
    reservoir_sample,
    split_tvts,
    deduplicate_by_dialog,
)

from data_engineering.loaders.persona_chat import load_persona_chat
from data_engineering.loaders.spc import load_spc
from data_engineering.loaders.character_codex import load_character_codex
from data_engineering.loaders.empathetic_dialogues import load_ed

try:
    from data_engineering.loaders.light_parlai import load_light_say_only

    HAS_PARLAI = True
except Exception:
    HAS_PARLAI = False

from data_engineering.loaders.skyrim_il import parse_skyrim_page
from data_engineering.loaders.uesp import fetch_uesp_facts
from collections import Counter


main_records = []
src_counter = Counter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def attach_world_facts(records, uesp_urls, max_fact_len=200):
    facts_pool = []
    for url in uesp_urls:
        try:
            facts = fetch_uesp_facts(url, max_facts=3)
            facts_pool.append((url, facts))
        except Exception as e:
            logger.warning(f"UESP fetch failed for {url}: {e}")
    if not facts_pool:
        return records
    out = []
    i = 0
    for r in records:
        if r["source"] in {"personachat", "spc", "charcodex", "light"}:
            url, facts = facts_pool[i % len(facts_pool)]
            facts = [f[:max_fact_len] for f in facts]
            rr = dict(r)
            rr["world_facts"] = (r.get("world_facts") or []) + facts
            meta = r.get("meta") or {}
            rr["meta"] = {**meta, "uesp_attrib": url}
            out.append(rr)
            i += 1
        else:
            out.append(r)
    return out


def _src_loader(name, split, cfg):
    if name == "personachat":
        return load_persona_chat(split=split)
    if name == "spc":
        return load_spc(split=split)
    if name == "charcodex":
        return load_character_codex(
            split=split,
            max_scenario_length=cfg["options"].get("max_scenario_length", 400),
        )
    if name == "light":
        if not HAS_PARLAI:
            return []
        return load_light_say_only(max_pairs=cfg["options"].get("light_max_pairs"))
    if name == "ed":
        return load_ed(split=split)
    return []



def _sanitize_record(rec):
    r = dict(rec)
    # clean dialog
    dlg = []
    for t in r.get("dialog", []):
        role = str(t.get("role", "")).strip()
        text = t.get("text")
        if not role or text is None:
            continue
        dlg.append({"role": role, "text": str(text)})
    r["dialog"] = dlg
    # clean persona/world_facts (strings only)
    r["persona"] = [str(x) for x in (r.get("persona") or []) if isinstance(x, (str, int, float))]
    r["world_facts"] = [str(x) for x in (r.get("world_facts") or []) if isinstance(x, (str, int, float))]
    # clean context (no None)
    ctx = {}
    for k in ("location", "npc_role", "time_of_day"):
        v = (r.get("context") or {}).get(k)
        if isinstance(v, str) and v.strip():
            ctx[k] = v.strip()
    r["context"] = ctx
    return r

def _collect_ablation_skyrim(k_skyrim):
    urls_file = pathlib.Path("config/skyrim_urls.txt")
    urls = urls_file.read_text().splitlines() if urls_file.exists() else []
    out = []
    for u in urls:
        try:
            out.extend(list(parse_skyrim_page(u, max_turns=20)))
            if len(out) >= k_skyrim:
                break
        except Exception as e:
            logger.warning(f"Skyrim parse failed for {u}: {e}")
    return out[:k_skyrim]

def _set_split(records, name):
    out = []
    for r in records:
        rr = dict(r)
        rr["split"] = name
        out.append(rr)
    return out

def run(cfg_path, out_dir):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    schema = load_schema("schema/npc_schema.json")

    total = cfg["targets"]["total_examples"]
    weights = cfg["weights"]
    abw = cfg.get("ablation_weights", {})

    # Combine base + ablation to a single allocation pass (no double counting)
    all_weights = {**weights, **abw}
    counts = allocate_counts(total, all_weights)

    # ---- MAIN COLLECTION (Option A: ignore upstream splits, sample from 'train') ----
    main_records = []
    for src, k in counts.items():
        if src == "skyrim":  # handled separately
            continue
        gen = _src_loader(src, "train", cfg)
        sampled = reservoir_sample(gen, k, seed=cfg["seed"])
        for r in sampled:
            main_records.append(r)
            src_counter[r.get("source","?")] += 1

    logger.info(f"[collect] sampled by source: {dict(src_counter)}")

    # Optional: attach UESP facts (attribution stored in meta)
    if cfg["options"].get("attach_uesp_world_facts", False):
        uesp_f = pathlib.Path("config/uesp_urls.txt")
        if uesp_f.exists():
            urls = [u for u in uesp_f.read_text().splitlines() if u.strip()]
            main_records = attach_world_facts(
                main_records,
                urls,
                max_fact_len=cfg["options"].get("max_fact_length", 200),
            )

    # Dedup → Validate → Single final split
    main_records = deduplicate_by_dialog(main_records)

    valid = []
    valid_counter = Counter()
    for r in tqdm(main_records, desc="validate(main)"):
        rr = _sanitize_record(r)  # if you added sanitizer; otherwise use r
        if validate_record(schema, rr):
            valid.append(rr)
            valid_counter[rr.get("source","?")] += 1

    logger.info(f"[validate] kept by source: {dict(valid_counter)}")

    train, val, test = split_tvts(
        valid,
        cfg["splits"]["train_ratio"],
        cfg["splits"]["val_ratio"],
        seed=cfg["seed"],
    )
    
    train = _set_split(train, "train")
    val   = _set_split(val,   "val")
    test  = _set_split(test,  "test")

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    # ---- Classwork-only: Skyrim ablation shard (kept separate) ----
    if cfg.get("include", {}).get("skyrim"):
        k_sky = counts.get("skyrim", 0)
        ab = _collect_ablation_skyrim(k_sky)  # unchanged helper
        ab = deduplicate_by_dialog(ab)
        ab_valid = []
        for r in tqdm(ab, desc="validate(ablation_skyrim)"):
            if validate_record(schema, r):
                ab_valid.append(r)
        write_jsonl(out / "ablation_skyrim.jsonl", ab_valid)

    # Efficient line counts (generator, not list())
    manifest = {
        "config": cfg_path,
        "counts": {
            "train": sum(1 for _ in open(out / "train.jsonl", "rb")),
            "val": sum(1 for _ in open(out / "val.jsonl", "rb")),
            "test": sum(1 for _ in open(out / "test.jsonl", "rb")),
        },
        "sha256": {
            "train": sha256_file(out / "train.jsonl"),
            "val": sha256_file(out / "val.jsonl"),
            "test": sha256_file(out / "test.jsonl"),
        },
    }
    if (out / "ablation_skyrim.jsonl").exists():
        manifest["counts"]["ablation_skyrim"] = sum(
            1 for _ in open(out / "ablation_skyrim.jsonl", "rb")
        )
        manifest["sha256"]["ablation_skyrim"] = sha256_file(
            out / "ablation_skyrim.jsonl"
        )

    (out / "MANIFEST.json").write_text(__import__("json").dumps(manifest, indent=2))
    logger.info("Done. Shards at: %s", str(out))
