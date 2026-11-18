import re
from difflib import SequenceMatcher
import pandas as pd
from .util import norm_name


def squash(x):
    """Normalise + remove ALL spaces: fixes OCR splits like 'AL TON' -> 'alton'."""
    return norm_name(x).replace(" ", "")


def build_supplier_keys(names, threshold=0.90):
    """names: iterable of raw names -> dict raw_name -> supplier_key.
    1) squash (case/legal-suffix/space-insensitive)  2) merge near-identical squashed keys (typos)."""
    raw = pd.Series(list(names)).dropna().unique()
    sq = {r: squash(r) for r in raw}
    keys = sorted({k for k in sq.values() if k})
    parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(keys):            # small lists (hundreds), O(n^2) is fine
        for b in keys[i + 1:]:
            if a[:3] == b[:3] and abs(len(a) - len(b)) <= 3 and SequenceMatcher(None, a, b).ratio() >= threshold:
                parent[find(b)] = find(a)
    return {r: ("S:" + find(k)) if k else "S:unknown" for r, k in sq.items()}
