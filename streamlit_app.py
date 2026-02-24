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
        df_raw, meta = load_and_clean(raw, mapping_templates=templates)
        st.subheader("Parsing Meta")
        st.write(meta)
        st.subheader("Rohspalten")
        st.write(df_raw.columns.tolist())

        broker, conf = meta.get("detected_broker"), meta.get("confidence", 0.0)
        st.write("Auto Erkennung:", broker, f"confidence={conf:.2f}")

        if broker and conf >= 0.6 and broker in templates:
            # build mapping: choose first matching candidate
            raw_cols = [c.lower() for c in df_raw.columns]
            mapping = {}
            for target, candidates in templates[broker].items():
                chosen = None
                for cand in candidates:
                    for i, rc in enumerate(raw_cols):
                        if cand in rc:
                            chosen = df_raw.columns[i]
                            break
                    if chosen:
                        break
                if chosen:
                    mapping[target] = chosen
            df_std = apply_mapping(df_raw, mapping)
            st.success(f"Mapping für {broker} angewendet.")
        else:
            st.info("Automatische Erkennung unsicher. Bitte manuell zuordnen.")
            mapping = {}
            for target in STANDARD_COLUMNS:
                mapping[target] = st.selectbox(f"Quelle für {target}", options=[""] + list(df_raw.columns), key=f"map_{target}")
            df_std = apply_mapping(df_raw, mapping)

        st.subheader("Standardisierte Spalten")
        st.write(df_std.columns.tolist())
        st.session_state["df_std"] = df_std
        status.success("Standardisiertes DataFrame bereit.")
    except Exception as e:
        status.error(f"Fehler beim Einlesen/Verarbeiten: {e}")
else:
    st.info("Bitte zuerst eine CSV-Datei hochladen.")
