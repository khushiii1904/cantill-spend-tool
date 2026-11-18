import numpy as np
import pandas as pd
from .items import build_item_keys
from .classifier import load_taxonomy, classify

COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"]


def load_labels(cfg):
    """Optional labels -> DataFrame indexed by record_id with: leaf (tuple), include, exclusion_reason."""
    lc = cfg.get("labels")
    import os
    if not lc or not os.path.exists(lc["path"]):
        return None
    d = pd.read_csv(lc["path"], dtype=str)
    rid = lc.get("id_prefix", "") + d[lc["id_col"]]
    leaf = d[lc["level_cols"]].fillna("").apply(lambda r: tuple(x.strip() for x in r if x.strip()), axis=1)
    out = pd.DataFrame({"leaf": leaf.values,
                        "include": pd.to_numeric(d[lc["include_col"]], errors="coerce").values if lc.get("include_col") else np.nan,
                        "exclusion_reason": d[lc["reason_col"]].values if lc.get("reason_col") else None,
                        "supplier_key": d[lc["supplier_col"]].values if lc.get("supplier_col") else None,
                        "item_key": d[lc["item_col"]].values if lc.get("item_col") else None}, index=rid)
    return out[~out.index.duplicated()]


def finish(df, cfg, labels=None, log=print):
    df = df.reset_index(drop=True)
    thr = cfg.get("item_similarity", 0.90)
    df["item_key"] = build_item_keys(df.item_code, df.text, thr)
    leaves = load_taxonomy(cfg["taxonomy"]["path"], cfg["taxonomy"].get("level_cols"))
    log(f"[taxonomy] {len(leaves)} leaf categories")
    inc = df.include == 1
    lab = None
    if labels is not None:
        lab = df.loc[inc, "record_id"].map(labels["leaf"]).map(lambda x: x if isinstance(x, tuple) and len(x) else None)
        if "include" in labels and labels["include"].notna().any():          # seed labels flagged as excluded carry no category
            keep = df.loc[inc, "record_id"].map(labels["include"]).fillna(1) == 1
            lab = lab.where(keep, None)
    cl = classify(df.loc[inc, "text"], df.loc[inc, "supplier_key"], df.loc[inc, "item_key"], leaves, labels=lab,
                  embed_model=cfg.get("embedding_model"), log=log)
    for c in ["l1", "l2", "l3", "l4"]:
        df[c] = ""
        df.loc[inc, c] = cl[c]
    df["confidence"] = 0.95
    df.loc[inc, "confidence"] = cl["confidence"]
    df["exclusion_reason"] = df["reason"]
    df["spend"] = df["spend"].round(2)
    return df
