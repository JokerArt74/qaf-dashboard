import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="QAF Optimizer", layout="wide")

# ---------------------------------------------------------
# BRANDING HEADER
# ---------------------------------------------------------
col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.markdown("""
    <div style="font-size:48px; line-height:1; color:#3EA6FF;">◆</div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <h1 style="color:white; margin-bottom:0;">QAF – Quantitative Allocation Framework</h1>
    <p style="color:#AAAAAA; margin-top:0;">
        Pilot-Version • Portfolio Optimizer & Rebalancing Engine
    </p>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# OPTIMIZER FUNCTION
# ---------------------------------------------------------
def mean_variance_optimizer(returns_df, long_only=True, max_weight=0.3, min_weight=0.0):
    returns_df = returns_df.dropna(axis=1, thresh=len(returns_df) * 0.8)
    returns_df = returns_df.fillna(method="ffill").fillna(method="bfill")

    if returns_df.shape[1] == 0:
        return pd.Series([], dtype=float)

    cov = returns_df.cov().values
    n = cov.shape[0]
    cov += np.eye(n) * 1e-6

    inv_cov = np.linalg.inv(cov)
    ones = np.ones(n)
    weights = inv_cov @ ones
    weights = weights / weights.sum()

    if long_only:
        weights = np.clip(weights, 0, None)

    weights = np.clip(weights, min_weight, max_weight)

    if weights.sum() == 0:
        weights = np.ones(n) / n
    else:
        weights = weights / weights.sum()

    return pd.Series(weights.round(4), index=returns_df.columns)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔄 Rebalancing", "ℹ️ Über QAF"])

# ---------------------------------------------------------
# TAB 1 — DASHBOARD
# ---------------------------------------------------------
with tab1:

    # Onboarding
    with st.expander("ℹ️ Kurzanleitung für neue Nutzer"):
        st.write("""
        Willkommen im QAF Dashboard!

        **So funktioniert es:**
        1. Lade eine CSV-Datei mit historischen Renditen hoch  
        2. Stelle die Optimierungsparameter ein  
        3. Starte die Optimierung  
        4. Sieh dir Gewichte, Kennzahlen und Risiko/Rendite-Profil an  
        5. Lade den Report herunter oder gehe zum Rebalancing-Tab  

        Viel Erfolg beim Testen!
        """)

    # -------------------------------
    # Upload Bereich
    # -------------------------------
    st.markdown("<hr style='border:1px solid #222;'>", unsafe_allow_html=True)

    with st.container():
        st.markdown("### Upload Bereich")
        uploaded_file = st.file_uploader("Portfolio-Datei hochladen (CSV)", type=["csv"])

        if uploaded_file:
            st.success("Datei erfolgreich hochgeladen!")
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head())

    # -------------------------------
    # Parameter Bereich
    # -------------------------------
    if uploaded_file:

        st.markdown("<hr style='border:1px solid #222;'>", unsafe_allow_html=True)

        with st.container():
            st.markdown("### Parameter für die Optimierung")

            col1, col2 = st.columns(2)

            with col1:
                target_return = st.number_input(
                    "Zielrendite (z. B. 0.08 für 8%)",
                    value=0.08,
                    step=0.01
                )

                long_only = st.checkbox("Nur Long-Positionen erlauben", value=True)

            with col2:
                max_weight = st.number_input(
                    "Maximalgewicht pro Asset (z. B. 0.3 für 30%)",
                    value=0.3,
                    step=0.05
                )

                min_weight = st.number_input(
                    "Minimalgewicht pro Asset (z. B. 0.0 für 0%)",
                    value=0.0,
                    step=0.01
                )

        run_opt
