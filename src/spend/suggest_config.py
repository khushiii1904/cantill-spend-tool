"""Draft a config for a NEW company (e.g. Company C) by guessing column roles from headers and values.
usage: python -m src.spend.suggest_config data/company_C/<invoice file>.csv data/company_C/<taxonomy>.csv > configs/C.yaml
Always read the draft and fix it by hand: the guesses are printed with a comment."""
import re, sys
import pandas as pd
import yaml

HINTS = {
    "id": r"line.?id|record.?id|^id$|row.?id|line.?no|uid",
    "amount": r"net.*(amount|value)|(amount|value|total|spend).*(usd|net)|line.?amount|amount|value|total|cost|price",
    "vat": r"vat|tax.?rate|gst",
    "currency": r"currency|curr\b|ccy",
    "rate": r"rate|fx",
    "invoice_date": r"invoice.?date|inv.?date",
    "posting_date": r"posting|post.?date|gl.?date|date",
    "supplier": r"supplier.?name|vendor.?name|supplier|vendor|payee|creditor",
    "item_code": r"material|sku|item.?(code|no|number)|part.?(no|number)|product.?code",
    "text": r"desc|text|item.?name|material.?name|product|memo|narrat",
    "po": r"\bpo\b|purchase.?order|po.?(no|num|ref)",
    "status": r"status|state|approval",
    "gl_desc": r"gl.*desc|account.*desc|ledger.*desc|account.?name",
}


def guess(cols):
    res = {k: [] for k in HINTS}
    for c in cols:
        for k, rx in HINTS.items():
            if re.search(rx, c, re.I):
                res[k].append(c)
    return res


def main(data_path, tax_path):
    d = pd.read_csv(data_path, dtype=str, nrows=20000)
    g = guess(d.columns)
    numeric = lambda c: d[c].dropna().str.replace(r"[,\s]", "", regex=True).str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.9 if d[c].notna().any() else False
    amount = next((c for c in g["amount"] if numeric(c)), None)
    sup = g["supplier"][0] if g["supplier"] else None
    kind = "name"
    if sup and d[sup].dropna().str.contains(r"[A-Za-z]{3,}.*[A-Za-z]{3,}").mean() < 0.3:
        kind = "id"
    text_cols = [c for c in g["text"] if c not in (g["id"] + g["item_code"])] or []
    tax = pd.read_csv(tax_path, dtype=str, nrows=3)
    cfg = {
        "name": "C", "mode": "single_system", "reporting_currency": "USD", "output_prefix": "C", "supplier_kind": kind,
        "file": {"path": data_path, "id_col": (g["id"] or [d.columns[0]])[0]},
        "columns": {"amount": amount, "currency": (g["currency"] or [None])[0], "rate": None,
                    "posting_date": (g["posting_date"] or [None])[0], "invoice_date": (g["invoice_date"] or [None])[0],
                    "supplier": sup, "item_code": (g["item_code"] or [None])[0], "po": (g["po"] or [None])[0],
                    "text_cols": list(dict.fromkeys(text_cols + g["gl_desc"]))},
        "amount_is_gross": False, "default_vat": 0.0,
        "exclusion": {"use_defaults": True, "default_columns": list(dict.fromkeys(text_cols + g["gl_desc"] + ([sup] if kind == "name" and sup else [])))},
        "taxonomy": {"path": tax_path},
    }
    if g["status"]:
        cfg["status"] = {"column": g["status"][0], "approved_values": ["approved", "paid"], "cancelled_values": ["cancelled", "canceled", "void"]}
    print("# DRAFT - check every line. All candidate columns found:", {k: v for k, v in g.items() if v})
    print("# Status values (edit approved/cancelled lists):", d[g["status"][0]].value_counts().to_dict() if g["status"] else "none")
    print("# Taxonomy columns:", list(tax.columns))
    print("# If amounts are gross (tax included) set amount_is_gross: true and columns.vat; if there is a line fx rate set columns.rate")
    print(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
