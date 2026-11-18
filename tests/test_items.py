import pandas as pd
from src.spend.items import build_item_keys
from src.spend.suppliers import build_supplier_keys

def test_items():
    codes = pd.Series(["M1", None, "M2", None, None])
    texts = pd.Series(["VALVE;BALL,2IN", "VALVE;BALL,2IN", "No Object", "No Object", ""])
    k = build_item_keys(codes, texts)
    assert k[0] == k[1]              # codeless line joins the code that owns its text
    assert k[2] != k[0]
    assert k[4] == "I:blank"

def test_suppliers():
    m = build_supplier_keys(["AL TON MINING (PTY) LTD", "Alton Mining Pty Ltd", "BLYMORE HYDRAULICS"])
    assert m["AL TON MINING (PTY) LTD"] == m["Alton Mining Pty Ltd"] != m["BLYMORE HYDRAULICS"]
