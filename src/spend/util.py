"""Small, well-tested cleaning helpers. Everything here is company-agnostic."""
import re
import pandas as pd
import numpy as np


def _isnull(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA


def parse_number(x):
    """'3,552.97' -> 3552.97 ; '1.234,56' -> 1234.56 ; '(12.5)' -> -12.5 ; junk -> NaN"""
    if _isnull(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        # whichever separator comes last is the decimal mark
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if len(parts[-1]) == 3 else s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_rate(x, default):
    """VAT typed as '15', '0.15', '15%' or blank -> 0.15. Blank -> default."""
    if _isnull(x) or str(x).strip() == "":
        return default
    v = parse_number(str(x).replace("%", ""))
    if np.isnan(v):
        return default
    return v / 100 if v > 1 else v


def parse_date(series, formats=None):
    """Try each format in turn on still-unparsed values. Returns (dates, ok_mask)."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in (formats or ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d.%m.%Y", "%Y%m%d"]):
        todo = out.isna() & s.notna() & (s != "")
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    ok = out.notna() & out.between("2000-01-01", "2035-12-31")  # impossible dates are flagged
    return out.where(ok), ok


def norm_po(x, pattern=r"(\d{10})"):
    """'PO#4510108243', 'PO 4510034567', '4510019411/20.0' -> 10-digit PO"""
    if _isnull(x):
        return None
    m = re.search(pattern, str(x))
    return m.group(1) if m else None


_LEGAL = r"\b(pty|ltd|limited|inc|llc|gmbh|sa|co|corp|corporation|company|plc|bv|ag|pvt|the|and)\b"


def norm_name(x):
    """'OAKENT MINING SA (PTY) LTD' -> 'oakent mining'"""
    if _isnull(x):
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", str(x).lower())
    s = re.sub(_LEGAL, " ", s)
    return re.sub(r"\s+", " ", s).strip()
