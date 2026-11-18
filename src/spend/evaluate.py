"""Score a run against labels (A: golden labels for ERP lines; B: noisy seed labels).  Dev tool only."""
import numpy as np
import pandas as pd


def _pairs(s): return (s * (s - 1) / 2).sum()


def _pr(df, a, b):
    both = _pairs(df.groupby([a, b]).size())
    return both / _pairs(df.groupby(a).size()), both / _pairs(df.groupby(b).size())


def evaluate(df, labels, cfg):
    if labels is None:
        print("[eval] no labels"); return
    d = df.merge(labels, left_on="record_id", right_index=True, how="inner", suffixes=("", "_g"))
    print(f"[eval] {len(d)} labelled rows")
    if labels["include"].notna().any():                       # exclusions (B seed)
        g_ex = d["include_g"] == 0; o_ex = d.include == 0
        print(f"[eval] exclusion: precision {(g_ex & o_ex).sum() / max(o_ex.sum(), 1):.3f}  recall {(g_ex & o_ex).sum() / max(g_ex.sum(), 1):.3f}")
    inc = d[d.include == 1]
    inc = inc[inc.include_g.fillna(1) == 1] if "include_g" in inc else inc
    inc = inc[inc.leaf.map(lambda t: isinstance(t, tuple) and len(t) > 0)]
    for lvl in (1, 2, 3):
        pred = inc[[f"l{i}" for i in range(1, lvl + 1)]].apply(tuple, axis=1)
        gold = inc.leaf.map(lambda t: tuple(t[:lvl]) if len(t) >= lvl else tuple(t + (t[-1],) * (lvl - len(t))))
        hit = (pred == gold) | (pred.map(lambda t: t) == gold)
        w = inc.spend.abs()
        print(f"[eval] L{lvl} accuracy: lines {hit.mean():.3f}   spend-weighted {(hit * w).sum() / max(w.sum(), 1):.3f}")
    if labels["supplier_key"].notna().any():
        s = d[d.supplier_key_g.notna()]
        big = s.supplier_key_g.value_counts().index[0]
        s2 = s[s.supplier_key_g != big] if s.supplier_key_g.value_counts().iloc[0] > 0.3 * len(s) else s
        p, r = _pr(s2, "supplier_key", "supplier_key_g"); print(f"[eval] supplier dedup (clean part): precision {p:.3f} recall {r:.3f}")
    if labels["item_key"].notna().any():
        p, r = _pr(d, "item_key", "item_key_g"); print(f"[eval] item dedup: precision {p:.3f} recall {r:.3f}")
