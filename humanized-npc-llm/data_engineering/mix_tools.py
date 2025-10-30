import math, random
from collections import defaultdict
import hashlib

def reservoir_sample(iterable, k, seed=42):
    random.seed(seed)
    sample = []
    for t, item in enumerate(iterable, 1):
        if t <= k:
            sample.append(item)
        else:
            m = random.randint(1, t)
            if m <= k:
                sample[m - 1] = item
    return sample


def allocate_counts(total, weights):
    s = sum(weights.values())
    counts = {k: int(round(total * (v / s))) for k, v in weights.items()}
    # fix rounding drift
    diff = total - sum(counts.values())
    keys = list(weights.keys())
    i = 0
    while diff != 0:
        counts[keys[i % len(keys)]] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1
    return counts


def split_tvts(records, train_ratio, val_ratio, seed=42):
    import random

    r = list(records)
    random.Random(seed).shuffle(r)
    n = len(r)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = r[:n_train]
    val = r[n_train : n_train + n_val]
    test = r[n_train + n_val :]
    return train, val, test


def deduplicate_by_dialog(records):
    seen=set(); out=[]
    for r in records:
        txt = " ".join(t["text"] for t in r.get("dialog", []))
        h = hashlib.md5(txt.encode("utf-8")).hexdigest()
        if h in seen: 
            continue
        seen.add(h); out.append(r)
    return out