"""Pipeline for a company with two systems (A): clean -> exclude -> match/count once -> supplier/item keys."""
import numpy as np
import pandas as pd
from .util import parse_number, parse_rate, norm_po, norm_name
from .suppliers import build_supplier_keys, squash


def _text(df, cols):
    cols = [c for c in cols if c in df]
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def load_erp(cfg):
    c = cfg["erp"]; d = pd.read_csv(c["path"], dtype=str)
    out = pd.DataFrame({"record_id": c["id_prefix"] + d[c["id_col"]], "system": "ERP"})
    out["amount"] = d[c["amount_col"]].map(parse_number)               # already net
    out["po"] = d[c["po_col"]].map(norm_po)
    out["supplier_raw"] = d[c["supplier_col"]]
    out["item_code"] = d[c["item_code_col"]] if c.get("item_code_col") else None
    out["text"] = _text(d, c["text_cols"])
    out["status"] = "approved"
    return out


def load_tool(cfg):
    c = cfg["tool"]; d = pd.read_csv(c["path"], dtype=str)
    out = pd.DataFrame({"record_id": c["id_prefix"] + d[c["id_col"]], "system": "TOOL"})
    gross = d[c["amount_col"]].map(parse_number)
    vat = d[c["vat_col"]].map(lambda x: parse_rate(x, cfg["default_vat"]))
    out["amount"] = gross / (1 + vat)                                  # remove tax: net = gross / (1+rate)
    out["po"] = d[c["po_col"]].map(norm_po)
    out["supplier_raw"] = d[c["supplier_col"]]
    out["item_code"] = None
    out["text"] = _text(d, c["text_cols"])
    out["status"] = d[c["status_col"]].str.strip().str.lower()
    out["invoice"] = d[c["invoice_col"]]
    out["_dupkey"] = d[c["invoice_col"]].fillna("") + "|" + d[c["po_col"]].fillna("") + "|" + out["text"] + "|" + d[c["amount_col"]].fillna("")
    return out


def _close(a, b, tol):
    return abs(a - b) <= max(1.0, tol * abs(b))


def run(cfg):
    tol = cfg["match"]["amount_tolerance"]
    erp, tool = load_erp(cfg), load_tool(cfg)
    tc = cfg["tool"]
    approved, cancelled = set(tc["approved_values"]), set(tc["cancelled_values"])

    # ---- supplier keys (one pass over every spelling from both systems) ----
    skey = build_supplier_keys(pd.concat([erp.supplier_raw, tool.supplier_raw]), cfg["match"]["supplier_similarity"])
    erp["supplier_key"] = erp.supplier_raw.map(skey)
    tool["supplier_key"] = tool.supplier_raw.map(skey)

    # ---- exclusions on the tool side, in a fixed priority order ----
    tool["include"], tool["reason"], tool["matched_to"], tool["match_type"] = 1, "", "", ""
    m = tool.status.isin(cancelled);                         tool.loc[m, ["include", "reason"]] = [0, "CANCELLED"]
    m = ~tool.status.isin(approved | cancelled);             tool.loc[m, ["include", "reason"]] = [0, "NOT_APPROVED"]
    m = (tool.include == 1) & tool.duplicated("_dupkey");    tool.loc[m, ["include", "reason"]] = [0, "DUPLICATE"]; tool.loc[m, "match_type"] = "exported_twice"

    # ---- count once: match remaining tool lines to ERP lines ----
    by_po = erp.groupby("po").apply(lambda g: list(zip(g.record_id, g.amount)), include_groups=False).to_dict()
    by_amt = {}
    for rid, a, s in zip(erp.record_id, erp.amount, erp.supplier_key):
        by_amt.setdefault((round(a), s), []).append((rid, a))

    for i in tool.index[tool.include == 1]:
        a, po, s = tool.at[i, "amount"], tool.at[i, "po"], tool.at[i, "supplier_key"]
        hit = None
        cand = by_po.get(po, []) if po else []
        for rid, v in cand:                                           # 1) same PO + same amount (one line)
            if _close(a, v, tol): hit = (rid, "po+amount"); break
        if not hit and len(cand) > 1 and _close(a, sum(v for _, v in cand), tol):
            hit = (cand[0][0], "po+sum")                              # 2) invoice covers all receipts of the PO
        if not hit:                                                   # 3) no/unknown PO: same supplier + same amount
            for k in (round(a) - 1, round(a), round(a) + 1):
                for rid, v in by_amt.get((k, s), []):
                    if _close(a, v, tol): hit = (rid, "supplier+amount"); break
                if hit: break
        if hit:
            tool.loc[i, ["include", "reason", "matched_to", "match_type"]] = [0, "DUPLICATE", hit[0], hit[1]]

    # ---- tool lines that matched an ERP line inherit its item code (items are keyed later, in common.finish) ----
    ecode = erp.set_index("record_id").item_code
    mm = tool.matched_to != ""
    tool["item_code"] = None
    tool.loc[mm, "item_code"] = tool.loc[mm, "matched_to"].map(ecode)

    erp["include"], erp["reason"] = 1, ""
    allr = pd.concat([erp, tool], ignore_index=True)
    allr["spend"] = np.where(allr.include == 1, allr.amount, 0.0)
    return allr
