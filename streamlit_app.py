# streamlit_app.py
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Portfolio Uploader", layout="wide")
st.title("Portfolio Upload & Preview")

# File uploader (immer sichtbar)
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

# Platz für Statusmeldungen
status = st.empty()

def read_csv_from_uploaded(uploaded):
    raw = uploaded.read()
    text = None
    for enc in ("utf-8", "latin1", "cp1252", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        raise ValueError("Datei konnte nicht dekodiert werden.")
    df = pd.read_csv(StringIO(text), sep=";", decimal=",", quotechar='"', dtype=str)
    return df

# Wenn Datei vorhanden: immer Preview + Holdings anzeigen
if uploaded_file is not None:
    try:
        df = read_csv_from_uploaded(uploaded_file)

        # Grundlegende Aufbereitung
        df = df.rename(columns=lambda c: c.strip())
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].str.strip().fillna("") + " " + df["time"].str.strip().fillna(""),
                dayfirst=True, errors="coerce"
            )
        else:
            df["datetime"] = pd.NaT

        num_cols = ["shares", "price", "amount", "fee", "tax"]
        for c in num_cols:
            if c in df.columns:
                df[c] = df[c].fillna("0").astype(str).str.replace(",", ".")
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Preview
        st.subheader("Preview (erste 20 Zeilen)")
        st.dataframe(df.head(20), use_container_width=True)

        # Executed Trades
        if "status" in df.columns:
            executed = df[df["status"].str.lower() == "executed"].copy()
        else:
            executed = df.copy()
        executed = executed.sort_values("datetime", ascending=False)
        st.subheader("Executed Trades (neueste 50)")
        st.dataframe(executed.head(50), use_container_width=True)

        # signed_shares berechnen
        def signed_shares(row):
            t = str(row.get("type", "")).lower()
            s = row.get("shares", 0.0)
            try:
                s = float(s)
            except:
                s = 0.0
            if t == "buy":
                return s
            if t == "sell":
                return -s
            return 0.0

        executed["signed_shares"] = executed.apply(signed_shares, axis=1)

        # Holdings aggregieren
        group_cols = []
        if "isin" in executed.columns:
            group_cols.append("isin")
        else:
            group_cols.append("description")
        if "description" in executed.columns:
            group_cols.append("description")

        holdings = executed.groupby(group_cols, dropna=False).agg(
            total_shares=("signed_shares", "sum"),
            last_price=("price", "last"),
            invested_amount=("amount", "sum")
        ).reset_index()
        holdings = holdings[holdings["total_shares"].abs() > 1e-9]

        st.subheader("Holdings (aggregiert)")
        st.dataframe(holdings.sort_values("total_shares", ascending=False).head(200), use_container_width=True)

        status.success("Datei eingelesen und Preview/Holdings angezeigt.")

        # Separater Bereich: Optimierung starten
        st.markdown("---")
        st.subheader("Optimierung")
        if st.button("Optimierung starten"):
            # Platzhalter: hier kommt deine Optimierungslogik hin
            st.info("Optimierung läuft... (Platzhalter)")
            # Beispiel: Ergebnis anzeigen
            st.success("Optimierung abgeschlossen (Platzhalter).")

    except Exception as e:
        status.error(f"Fehler beim Einlesen/Verarbeiten: {e}")
else:
    st.info("Bitte zuerst eine CSV-Datei hochladen.")
