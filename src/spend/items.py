"""Item de-duplication: same code => same item; near-identical descriptions => same item."""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BLANKISH = {"", "product", "products", "parts", "part", "service", "services", "item", "items", "misc", "other", "na", "n/a", "-"}


def clean_text(x):
    """Lower-case, fix common OCR confusions (0 inside words -> o), drop punctuation and unit noise."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).lower()
    s = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", s)          # B0OM -> boom
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class _UF:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def build_item_keys(codes, texts, threshold=0.90, block=2000, max_codes_per_text=3):
    """codes, texts: aligned Series. Returns Series of item_key.
    Step 1: lines sharing a code are one item.  Step 2: unique cleaned texts with cosine >= threshold are merged
    (char n-gram TF-IDF).  Step 3: texts that are blank/uninformative get a key of their own per code (or 'unknown')."""
    ct = texts.map(clean_text)
    codes = pd.Series([None if (c is None or pd.isna(c) or str(c).strip() == "") else str(c).strip() for c in codes],
                      index=texts.index, dtype=object)
    uniq = pd.Series(ct.unique())
    informative = uniq[~uniq.isin(BLANKISH) & (uniq.str.len() >= 3)].reset_index(drop=True)
    node = {t: i for i, t in enumerate(informative)}
    uf = _UF(len(informative))
    if len(informative) > 1:
        X = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True).fit_transform(informative))
        for s in range(0, X.shape[0], block):
            sim = (X[s:s + block] @ X.T).tocoo()
            m = (sim.data >= threshold) & (sim.row + s < sim.col)
            for r, c in zip(sim.row[m] + s, sim.col[m]):
                uf.union(r, c)
    # A code defines an item.  Codes are merged only through a *specific* shared text (one used by few codes),
    # so generic texts like "No Object" cannot chain unrelated codes together.
    text_root = {t: uf.find(i) for t, i in node.items()}
    coded = pd.DataFrame({"c": codes.values, "t": ct.values})
    coded = coded[coded.c.notna() & coded.t.isin(node)]
    t2c = coded.groupby("t").c.agg(lambda x: sorted(set(x)))
    uc = list(pd.Series(codes.dropna().unique()))
    cid = {c: i for i, c in enumerate(uc)}
    ufc = _UF(len(uc))
    root_codes = {}                                   # text-cluster root -> codes seen with it
    for t, cs in t2c.items():
        root_codes.setdefault(text_root[t], set()).update(cs)
    for r, cs in root_codes.items():
        if 1 < len(cs) <= max_codes_per_text:
            cs = sorted(cs)
            for c2 in cs[1:]:
                ufc.union(cid[cs[0]], cid[c2])
    # codeless lines: take the code of their text cluster when it is unambiguous
    unique_code = {r: next(iter(cs)) for r, cs in root_codes.items() if len(cs) == 1}
    keys = []
    for c, t in zip(codes, ct):
        if c is not None:
            keys.append("I:c" + str(ufc.find(cid[c])))
        elif t in node:
            r = text_root[t]
            keys.append("I:c" + str(ufc.find(cid[unique_code[r]])) if r in unique_code else "I:t" + str(r))
        else:
            keys.append("I:blank")
    return pd.Series(keys, index=texts.index)
