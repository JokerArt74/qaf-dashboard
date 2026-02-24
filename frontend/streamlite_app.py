import streamlit as st
import pandas as pd
from io import StringIO
import requests

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
        # Datei an Backend senden
        with st.spinner("Sende Datei an Backend..."):
            try:
                res = requests.post(
                    "http://localhost:8000/upload",
                    files={"file": uploaded_file}
                )
                data = res.json()
            except Exception as e:
                st.error(f"Fehler beim Senden an Backend: {e}")
                st.stop()

        if "error" in data:
            st.error(data["error"])
        else:
            st.success("Backend hat die Datei verarbeitet.")
            st.write("Spalten:", data["columns"])
            st.dataframe(data["rows"])

            # Speichern für andere Tabs
            st.session_state["uploaded_columns"] = data["columns"]
            st.session_state["uploaded_preview"] = data["rows"]


# --- Page: Holdings ---
elif page == "Holdings":
    st.title("📊 Holdings")

    if "uploaded_columns" not in st.session_state:
        st.warning("Bitte zuerst eine CSV-Datei im Upload-Tab hochladen.")
    else:
        st.write("Spalten:", st.session_state["uploaded_columns"])
        st.write("Vorschau:", st.session_state["uploaded_preview"])
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
