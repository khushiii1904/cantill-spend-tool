"""Pipeline for a company with ONE invoice-line file (B, C): parse, convert currency, exclude, supplier keys."""
import numpy as np
import pandas as pd
import yaml, os
from .util import parse_number, parse_rate, parse_date
from .suppliers import build_supplier_keys
from .exclusions import apply_keyword_rules, learn_rules, apply_learned


def _fx_table(cfg):
    f = cfg.get("fx")
    if not f or not os.path.exists(f["path"]):
        return {}, {}
    d = pd.read_csv(f["path"], dtype=str)
    d["m"] = d[f["month_col"]].str.slice(0, 7); d["r"] = d[f["rate_col"]].map(parse_number)
    tbl = {(c, m): r for c, m, r in zip(d[f["currency_col"]], d.m, d.r)}
    cur_months = {c: sorted(g.m.unique()) for c, g in d.groupby(f["currency_col"])}
    return tbl, cur_months


def _nearest_month(months, m):
    y, mo = int(m[:4]), int(m[5:7]); t = y * 12 + mo
    return min(months, key=lambda x: abs(int(x[:4]) * 12 + int(x[5:7]) - t))


def run(cfg, labels=None, log=print):
    c = cfg["columns"]; raw = pd.read_csv(cfg["file"]["path"], dtype=str)
    out = pd.DataFrame({"record_id": cfg["file"].get("id_prefix", "") + raw[cfg["file"]["id_col"]], "system": "ERP"})
    # ---- amounts ----
    amt = raw[c["amount"]].map(parse_number)
    if cfg.get("amount_is_gross") and c.get("vat"):
        amt = amt / (1 + raw[c["vat"]].map(lambda x: parse_rate(x, cfg.get("default_vat", 0.0))))
    # ---- dates (flag, never silently trust) ----
    dates = {}
    for k in ("posting_date", "invoice_date"):
        if c.get(k):
            dates[k], _ = parse_date(raw[c[k]], cfg.get("date_formats"))
    out["date_flag"] = ""
    for k, v in dates.items():
        out.loc[v.isna(), "date_flag"] += f"{k}_unparsed;"
    # ---- currency -> reporting currency ----
    rep = cfg["reporting_currency"]; fx, cur_months = _fx_table(cfg)
    cur = raw[c["currency"]].fillna(rep).str.strip().str.upper() if c.get("currency") else pd.Series(rep, index=raw.index)
    rate = raw[c["rate"]].map(parse_number) if c.get("rate") else pd.Series(np.nan, index=raw.index)
    ref_dates = dates.get("invoice_date", pd.Series(pd.NaT, index=raw.index))
    alt_dates = dates.get("posting_date", pd.Series(pd.NaT, index=raw.index))
    med = {}                                                  # last resort: median line rate of that currency
    ok_r = rate.where(rate > 0)
    for cc, g in ok_r.groupby(cur):
        if g.notna().any(): med[cc] = g.median()
    usd = np.empty(len(raw)); how = []
    for i, (a, cu, r, d1, d2) in enumerate(zip(amt.values, cur.values, rate.values, ref_dates.values, alt_dates.values)):
        if cu == rep:
            usd[i] = a; how.append("same"); continue
        if r and r > 0:
            usd[i] = a / r; how.append("line_rate"); continue
        val = None
        for d in (d1, d2):
            if pd.isna(d): continue
            m = str(pd.Timestamp(d))[:7]
            if (cu, m) in fx: val = fx[(cu, m)]; break
        if val is None and cu in cur_months:
            ref = next((str(pd.Timestamp(d))[:7] for d in (d2, d1) if not pd.isna(d)), cur_months[cu][len(cur_months[cu]) // 2])
            val = fx[(cu, _nearest_month(cur_months[cu], ref))]; how.append("fx_nearest")
        elif val is not None: how.append("fx_month")
        if val is None and cu in med: val = med[cu]; how.append("median_rate")
        if val is None: how.append("no_rate")
        usd[i] = a / val if val else np.nan
    out["amount"] = usd
    out["fx_method"] = how
    # ---- text / supplier / item ----
    tc = [x for x in c["text_cols"] if x in raw]
    out["text"] = raw[tc].fillna("").astype(str).agg(" ".join, axis=1).str.strip() if tc else ""
    out["item_code"] = raw[c["item_code"]] if c.get("item_code") else None
    out["po"] = raw[c["po"]] if c.get("po") else None
    out["supplier_raw"] = raw[c["supplier"]]
    if cfg.get("supplier_kind", "name") == "id":
        out["supplier_key"] = "V:" + raw[c["supplier"]].fillna("unknown").str.strip()
    else:
        sk = build_supplier_keys(raw[c["supplier"]], cfg.get("supplier_similarity", 0.9)); out["supplier_key"] = raw[c["supplier"]].map(sk)
    out["status"] = ""
    # ---- exclusions, fixed priority: CANCELLED > NOT_APPROVED > NON_PROCUREMENT ----
    reason = pd.Series("", index=raw.index)
    st = cfg.get("status")
    if st and st["column"] in raw:
        s = raw[st["column"]].fillna("").str.strip().str.lower()
        canc = s.isin([x.lower() for x in st.get("cancelled_values", [])])
        notap = (~s.isin([x.lower() for x in st.get("approved_values", [])])) & ~canc & (s != "" if st.get("blank_is_approved", True) else True)
        reason = reason.mask(canc, "CANCELLED").mask(notap & (reason == ""), "NOT_APPROVED")
    ex = cfg.get("exclusion", {})
    rules = list(ex.get("rules", []))
    if ex.get("use_defaults"):
        rx = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "defaults.yaml")))["non_procurement_regex"]
        rules += [{"column": col, "regex": rx, "reason": "NON_PROCUREMENT"} for col in ex.get("default_columns", [])]
    kw = apply_keyword_rules(raw, rules, {k: raw[k] for k in raw.columns})
    reason = reason.where(reason != "", kw)
    learned = []
    if labels is not None and ex.get("learn_from_labels"):
        lab = labels.reindex(out.record_id)
        inc = pd.Series(lab["include"].values, index=raw.index); rs = pd.Series(lab["exclusion_reason"].values, index=raw.index)
        learned = learn_rules(raw, inc, rs, ex["learn_from_labels"], ex.get("min_rows", 5), ex.get("min_rate", 0.9))
        log(f"[exclusions] learned {len(learned)} value rules from labelled rows")
        reason = reason.where(reason != "", apply_learned(raw, learned))
    out["include"] = (reason == "").astype(int); out["reason"] = reason
    out["spend"] = np.where(out.include == 1, out.amount.fillna(0.0), 0.0)
    out["matched_to"] = ""; out["match_type"] = ""
    return out
