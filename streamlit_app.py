# streamlit_app.py
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Portfolio Uploader - Recovery", layout="wide")
st.title("Portfolio Uploader - Recovery")

st.markdown("Diese Minimalversion zeigt nur Upload und Preview. Später bauen wir die Logik wieder auf.")

uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])
if uploaded_file is None:
    st.info("Bitte zuerst eine CSV-Datei hochladen.")
else:
    try:
        raw = uploaded_file.read()
        # Versuche UTF-8, fallback auf latin1
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin1", errors="replace")
        # Versuche Semikolon als Trennzeichen, fallback Komma
        try:
            df = pd.read_csv(StringIO(text), sep=";", decimal=",", quotechar='"', dtype=str)
        except Exception:
            df = pd.read_csv(StringIO(text), sep=",", decimal=".", quotechar='"', dtype=str)
        df = df.rename(columns=lambda c: str(c).strip())
        st.subheader("Spalten")
        st.write(df.columns.tolist())
        st.subheader("Preview (erste 20 Zeilen)")
        st.dataframe(df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Fehler beim Einlesen der Datei: {e}")
