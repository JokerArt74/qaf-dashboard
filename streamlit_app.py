# streamlit_app.py
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Portfolio Uploader - Recovery", layout="wide")
st.title("Portfolio Uploader - Recovery")

st.markdown("Diese Minimalversion zeigt nur Upload und Preview. Später bauen wir die Logik wieder auf.")

# Upload + Auto-detect + Manual mapping
import glob
from src.data import load_and_clean, load_mapping_template, apply_mapping, STANDARD_COLUMNS

# lade mapping templates
mapping_files = glob.glob("mappings/*.json")
templates = {}
for mf in mapping_files:
    templates.update(load_mapping_template(mf))

uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])
status = st.empty()

if uploaded_file is not None:
    try:
        raw = uploaded_file.read()
        # Debug: detaillierte Matching-Info anzeigen
from collections import defaultdict

def compute_match_details(df_cols, template):
    df_cols_l = [c.lower() for c in df_cols]
    details = defaultdict(list)
    for target, candidates in template.items():
        for cand in candidates:
            for i, rc in enumerate(df_cols_l):
                if cand in rc:
                    details[target].append({"candidate": cand, "matched_column": df_cols[i]})
    return dict(details)

# nach df_raw, meta erzeugt
st.subheader("Debug Matching Details")
if broker and broker in templates:
    details = compute_match_details(df_raw.columns.tolist(), templates[broker])
    st.write(details)
else:
    # zeige mögliche Matches für alle Templates (hilft zu sehen, warum ein anderer Broker gewinnt)
    all_details = {}
    for b, t in templates.items():
        all_details[b] = compute_match_details(df_raw.columns.tolist(), t)
    st.write(all_details)

        st.subheader("Standardisierte Spalten")
        st.write(df_std.columns.tolist())
        st.session_state["df_std"] = df_std
        status.success("Standardisiertes DataFrame bereit.")
    except Exception as e:
        status.error(f"Fehler beim Einlesen/Verarbeiten: {e}")
else:
    st.info("Bitte zuerst eine CSV-Datei hochladen.")
