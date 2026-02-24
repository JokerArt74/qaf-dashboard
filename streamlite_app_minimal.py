# streamlit_app_minimal.py
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Upload Test", layout="wide")
st.title("Upload Test — Scalable CSV")

uploaded_file = st.file_uploader("CSV hochladen (Scalable Export)", type=["csv"])

if uploaded_file is not None:
    try:
        raw = uploaded_file.read()  # bytes

        # Versuche gängige Encodings
        text = None
        for enc in ("utf-8", "latin1", "cp1252", "utf-16"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                text = None

        if text is None:
            st.error("Datei konnte nicht dekodiert werden. Bitte als UTF-8/Latin1/CP1252/UTF-16 speichern.")
        else:
            # Einlesen: Semikolon als Separator, Komma als Dezimaltrennzeichen
            df = pd.read_csv(StringIO(text), sep=";", decimal=",", quotechar='"', dtype=str)

            st.subheader("Spalten")
            st.write(df.columns.tolist())

            st.subheader("Preview (erste 20 Zeilen)")
            st.dataframe(df.head(20), use_container_width=True)

            # Konvertierungen
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

            # Executed Trades
            executed = df[df["status"].str.lower() == "executed"].copy()
            executed = executed.sort_values("datetime", ascending=False)
            st.subheader("Executed Trades Anzahl")
            st.write(len(executed))

            # signed_shares
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

    except Exception as e:
        st.error(f"Fehler beim Einlesen/Verarbeiten: {e}")
