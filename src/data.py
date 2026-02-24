# src/data.py
from io import StringIO
import pandas as pd
import json
import chardet

STANDARD_COLUMNS = [
    "date","time","datetime","type","status","isin","description",
    "shares","price","amount","fee","tax","currency"
]

def detect_encoding(raw_bytes):
    try:
        res = chardet.detect(raw_bytes)
        return res.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"

def read_csv_bytes(raw_bytes, sep_candidates=(";", ",", "\t")):
    encoding = detect_encoding(raw_bytes)
    text = raw_bytes.decode(encoding, errors="replace")
    best_sep = None
    best_cols = 0
    for sep in sep_candidates:
        try:
            df_try = pd.read_csv(StringIO(text), sep=sep, nrows=5, dtype=str)
            if len(df_try.columns) > best_cols:
                best_cols = len(df_try.columns)
                best_sep = sep
        except Exception:
            continue
    sep = best_sep or ";"
    try:
        df = pd.read_csv(StringIO(text), sep=sep, dtype=str, quotechar='"', encoding=encoding)
    except Exception:
        df = pd.read_csv(StringIO(text), sep=";", dtype=str, quotechar='"', encoding="utf-8", errors="replace")
    meta = {"encoding": encoding, "sep": sep}
    return df, meta

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def parse_numeric_columns(df, cols=None):
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if any(k in c for k in ("share","qty","quantity","price","amount","fee","tax"))]
    for c in cols:
        if c in df.columns:
            s = df[c].fillna("").astype(str).str.replace(".", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s.replace("", "0"), errors="coerce").fillna(0.0)
    return df

def ensure_datetime(df, date_col_candidates=("date","trade_date","datum"), time_col_candidates=("time","trade_time")):
    df = df.copy()
    date_col = next((c for c in df.columns if c in date_col_candidates), None)
    time_col = next((c for c in df.columns if c in time_col_candidates), None)
    if date_col:
        date_series = df[date_col].astype(str).str.strip().replace({"": None})
        if time_col:
            time_series = df[time_col].astype(str).str.strip().replace({"": "00:00:00"})
            combined = date_series.fillna("") + " " + time_series.fillna("")
            df["datetime"] = pd.to_datetime(combined, dayfirst=True, errors="coerce")
        else:
            df["datetime"] = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    else:
        df["datetime"] = pd.NaT
    return df

def load_mapping_template(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_broker_by_columns(df, mapping_templates=None):
    cols = [c.lower().strip() for c in df.columns]
    if mapping_templates is None:
        mapping_templates = {}
    scores = {}
    for broker, mapping in mapping_templates.items():
        match_count = 0
        total = 0
        for target, candidates in mapping.items():
            total += 1
            for cand in candidates:
                if any(cand in c for c in cols):
                    match_count += 1
                    break
        scores[broker] = match_count / max(1, total)
    if not scores:
        return None, 0.0
    best = max(scores.items(), key=lambda x: x[1])
    return best  # (broker_name, confidence)

def apply_mapping(df, mapping):
    df = df.copy()
    out = {}
    for target in STANDARD_COLUMNS:
        src = mapping.get(target)
        if src and src in df.columns:
            out[target] = df[src]
        else:
            out[target] = pd.NA
    out_df = pd.DataFrame(out)
    out_df.attrs["original_columns"] = df.columns.tolist()
    return out_df

def load_and_clean(raw_bytes, mapping_templates=None):
    df_raw, meta = read_csv_bytes(raw_bytes)
    df_raw = normalize_columns(df_raw)
    df_raw = parse_numeric_columns(df_raw)
    df_raw = ensure_datetime(df_raw)
    meta["columns"] = df_raw.columns.tolist()
    broker, conf = detect_broker_by_columns(df_raw, mapping_templates)
    meta["detected_broker"] = broker
    meta["confidence"] = conf
    return df_raw, meta
