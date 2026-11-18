"""Exclusion rules = keyword rules from the config (+ defaults) + rules learned from labelled rows (if labels exist)."""
import re
import pandas as pd


def apply_keyword_rules(df, rules, cols_map):
    """rules: list of {column, regex, reason}; cols_map: name->Series source dataframe 'raw' columns."""
    reason = pd.Series("", index=df.index)
    for r in rules or []:
        col = r["column"]
        if col not in cols_map:
            continue
        hit = cols_map[col].fillna("").str.contains(r["regex"], case=False, regex=True, na=False)
        reason = reason.where(reason != "", pd.Series(r["reason"], index=df.index).where(hit, ""))
    return reason


def learn_rules(raw, labels_inc, labels_reason, columns, min_rows=5, min_rate=0.9):
    """Find values of categorical columns where >= min_rate of labelled rows were excluded -> exclusion rule.
    raw: raw dataframe (aligned to labels by index); labels_inc: Series 0/1; labels_reason: Series code."""
    learned = []
    lab = labels_inc.notna()
    for c in columns:
        if c not in raw:
            continue
        d = pd.DataFrame({"v": raw.loc[lab, c].fillna("").str.strip(), "ex": (labels_inc[lab] == 0).values, "r": labels_reason[lab].values})
        g = d.groupby("v").agg(n=("ex", "size"), rate=("ex", "mean"), reason=("r", lambda s: s.dropna().mode().iat[0] if s.notna().any() else "NON_PROCUREMENT"))
        for v, row in g[(g.n >= min_rows) & (g.rate >= min_rate)].iterrows():
            learned.append((c, v, row.reason))
    return learned


def apply_learned(raw, learned):
    reason = pd.Series("", index=raw.index)
    for c, v, rs in learned:
        m = raw[c].fillna("").str.strip() == v
        reason = reason.where(reason != "", pd.Series(rs, index=raw.index).where(m, ""))
    return reason
