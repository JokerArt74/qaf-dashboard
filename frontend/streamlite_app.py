import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="QAF – Portfolio Manager", layout="wide")
st.title("QAF – Portfolio Manager (MVP Frontend)")

st.markdown("""
Willkommen bei QAF.

1. Lade einen CSV-Export deines Brokers hoch  
2. Sieh dir die Rohdaten an  
3. Später: Mapping, Holdings, Analytics, Optimizer
""")

uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])

if not uploaded_file:
    st.info("Bitte eine CSV-Datei hochladen.")
else:
    raw = uploaded_file.read()
    # einfache Encoding- und Separator-Erkennung
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin1", errors="replace")

    df = None
    detected_sep = None
    for sep in (";", ",", "\t"):
        try:
            df_try = pd.read_csv(StringIO(text), sep=sep, dtype=str)
            if df_try.shape[1] > 1:
                df = df_try
                detected_sep = sep
                break
        except Exception:
            continue

    if df is None:
        st.error("Datei konnte nicht eingelesen werden.")
    else:
        df.columns = [str(c).strip() for c in df.columns]
        st.subheader("Erkannter Separator")
        st.write(repr(detected_sep))

        st.subheader("Spalten")
        st.write(df.columns.tolist())

        st.subheader("Vorschau (erste 20 Zeilen)")
        st.dataframe(df.head(20), use_container_width=True)
