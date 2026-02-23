import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------------------------------------------------------
# PAGE CONFIG + HEADER
# ---------------------------------------------------------

st.set_page_config(page_title="QAF Optimizer", layout="wide")

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
    """
    Stabile Mean-Variance-Optimierung mit Constraints:
    - Long-only (optional)
    - Min/Max-Gewichte
    """

    # 1. Fehlende Werte bereinigen
    returns_df = returns_df.dropna(axis=1, thresh=len(returns_df) * 0.8)
    returns_df = returns_df.fillna(method="ffill").fillna(method="bfill")

    if returns_df.shape[1] == 0:
        return pd.Series([], dtype=float)

    # 2. Kovarianzmatrix
    cov = returns_df.cov().values
    n = cov.shape[0]

    # 3. Regularisierung
    cov += np.eye(n) * 1e-6

    # 4. Inverse
    inv_cov = np.linalg.inv(cov)

    # 5. Minimum-Variance
    ones = np.ones(n)
    weights = inv_cov @ ones
    weights = weights / weights.sum()

    # 6. Long-only
    if long_only:
        weights = np.clip(weights, 0, None)

    # 7. Min/Max-Gewichte
    weights = np.clip(weights, min_weight, max_weight)

    # 8. Normalisieren
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

    st.subheader("Upload Bereich")
    uploaded_file = st.file_uploader("Portfolio-Datei hochladen (CSV)", type=["csv"])

    if uploaded_file:
        st.success("Datei erfolgreich hochgeladen!")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head())

        st.subheader("Parameter für die Optimierung:")

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

        run_opt = st.button("Optimierung starten")

        if run_opt:

            st.subheader("Optimierungsergebnis")

            df = pd.read_csv(uploaded_file).dropna()

            weights = mean_variance_optimizer(
                df,
                long_only=long_only,
                max_weight=max_weight,
                min_weight=min_weight
            )

            if len(weights) == 0:
                st.error("Keine gültigen Assets nach Bereinigung. Bitte Daten prüfen.")
            else:
                mean_returns = df.mean() * 252
                cov = df.cov() * 252

                w = weights.values
                port_return = float(mean_returns @ w)
                port_vol = float(np.sqrt(w @ cov.values @ w))

                colA, colB = st.columns([1, 1])

                with colA:
                    st.write("Berechnete Gewichte:")
                    st.table(weights)

                with colB:
                    st.write("Kennzahlen:")
                    st.metric("Erwartete Rendite", f"{round(port_return,4)}")
                    st.metric("Volatilität", f"{round(port_vol,4)}")

                # Abstand vor dem Chart
st.markdown("### ")

st.subheader("Risiko/Rendite-Profil")

chart_data_rr = pd.DataFrame({
    "Name": ["Optimiertes Portfolio"],
    "Return": [port_return],
    "Volatility": [port_vol]
})

scatter = alt.Chart(chart_data_rr).mark_circle(size=120).encode(
    x=alt.X("Volatility", title="Volatilität"),
    y=alt.Y("Return", title="Erwartete Rendite"),
    tooltip=["Name", "Return", "Volatility"]
)

st.altair_chart(scatter, use_container_width=True)

# Linie nach dem Chart
st.markdown("---")

                with st.container():
                    st.markdown("### Kurzfassung für Entscheider")
                    st.info(
                        f"""
                        **Zielrendite:** {target_return}  
                        **Long-Only:** {long_only}  
                        **Optimiertes Portfolio:**  
                        Rendite **{round(port_return,4)}**, Volatilität **{round(port_vol,4)}**
                        """
                    )

    else:
        st.info("Bitte zuerst eine Datei hochladen, um die Optimierung zu aktivieren.")

# ---------------------------------------------------------
# TAB 2 — REBALANCING
# ---------------------------------------------------------

with tab2:

    st.subheader("Rebalancing – Kauf/Verkauf-Empfehlungen")

    st.write("Gib dein aktuelles Portfolio ein, um Trades zu berechnen.")

    current_portfolio_input = st.text_area(
        "Format: Ticker: Gewicht, z. B. AAPL: 0.2, MSFT: 0.3",
        value="",
        height=100
    )

    if uploaded_file and current_portfolio_input.strip():

        df = pd.read_csv(uploaded_file).dropna()

        weights = mean_variance_optimizer(df)

        try:
            current_dict = {}
            for line in current_portfolio_input.split(","):
                if ":" in line:
                    ticker, weight = line.split(":")
                    current_dict[ticker.strip()] = float(weight.strip())

            current_series = pd.Series(current_dict)
            aligned_current = current_series.reindex(weights.index).fillna(0)

            trades = weights - aligned_current

            st.write("Positive Werte = Kaufen, Negative Werte = Verkaufen")
            st.table(trades.rename("Trade"))

        except Exception as e:
            st.error(f"Fehler beim Einlesen des aktuellen Portfolios: {e}")

    else:
        st.info("Bitte Datei hochladen und aktuelles Portfolio eingeben.")

# ---------------------------------------------------------
# TAB 3 — ABOUT
# ---------------------------------------------------------

with tab3:
    st.subheader("Über QAF")
    st.write("""
    QAF – Quantitative Allocation Framework  
    Pilot-Version für professionelle Portfoliosteuerung.

    Funktionen:
    - Mean-Variance-Optimierung
    - Constraints (Min/Max/Long-Only)
    - Risiko/Rendite-Analyse
    - Rebalancing-Empfehlungen
    - Datenbereinigung & Stabilisierung

    Kontakt:
    michael@deinefirma.com
    """)
