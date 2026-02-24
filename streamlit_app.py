# streamlit_app.py
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Portfolio Uploader", layout="wide")
st.title("Portfolio Upload & Preview")

# File uploader (immer sichtbar)
uploaded_file = st.file_uploader("CSV hochladen (Scalable Export)", type=["csv"])

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
