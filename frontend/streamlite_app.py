import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="QAF – Portfolio Manager", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("QAF Navigation")
page = st.sidebar.radio(
    "Menü",
    ["Upload", "Holdings", "Analytics", "Optimizer", "Settings"]
)

# --- Page: Upload ---
if page == "Upload":
    st.title("📥 CSV Upload")
    st.markdown("Lade hier deinen Broker-Export hoch.")

    uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])

    if not uploaded_file:
        st.info("Bitte eine CSV-Datei hochladen.")
    else:
        raw = uploaded_file.read()

        # Encoding erkennen
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
            st.success("Datei erfolgreich eingelesen.")
            st.write(f"Erkannter Separator: `{detected_sep}`")

            st.subheader("Vorschau")
            st.dataframe(df.head(20), use_container_width=True)

            # Speichern für andere Tabs
            st.session_state["uploaded_df"] = df


# --- Page: Holdings ---
elif page == "Holdings":
    st.title("📊 Holdings")

    if "uploaded_df" not in st.session_state:
        st.warning("Bitte zuerst eine CSV-Datei im Upload-Tab hochladen.")
    else:
        df = st.session_state["uploaded_df"]
        st.write("Rohdaten:")
        st.dataframe(df.head(), use_container_width=True)

        st.info("Hier werden später die aggregierten Holdings angezeigt.")


# --- Page: Analytics ---
elif page == "Analytics":
    st.title("📈 Analytics")
    st.info("Hier kommen später Charts, Performance, Risiko, Exposure usw.")


# --- Page: Optimizer ---
elif page == "Optimizer":
    st.title("🧠 Optimizer")
    st.info("Hier kommt später der Mean-Variance-Optimizer rein.")


# --- Page: Settings ---
elif page == "Settings":
    st.title("⚙️ Einstellungen")
    st.info("Hier kommen später User-Settings, API-Keys, Broker-Profile usw.")
