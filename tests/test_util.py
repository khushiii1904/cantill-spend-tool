import numpy as np, pandas as pd
from src.spend.util import *

def test_numbers():
    assert parse_number("3,552.97") == 3552.97
    assert parse_number("1.234,56") == 1234.56
    assert parse_number("(12.5)") == -12.5
    assert np.isnan(parse_number("abc"))

def test_rate():
    assert parse_rate("15%", .15) == .15 and parse_rate("0.15", .15) == .15
    assert parse_rate(None, .15) == .15 and parse_rate("0", .15) == 0

def test_po_name():
    assert norm_po("PO#4510108243") == "4510108243" and norm_po("4510019411/20.0") == "4510019411"
    assert norm_name("OAKENT MINING SA (PTY) LTD") == "oakent mining"

def test_dates():
    d, ok = parse_date(pd.Series(["19/01/2024", "2022-01-29", None, "31/02/2023", "1900-01-01"]))
    assert list(ok) == [True, True, False, False, False]
