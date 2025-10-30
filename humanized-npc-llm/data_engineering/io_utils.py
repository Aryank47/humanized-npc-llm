import json, hashlib, pathlib, orjson


def write_jsonl(path, records):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
