import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="QAF Optimizer", layout="wide")

# =========================================================
# BRANDING HEADER
# =========================================================
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

# =========================================================
# OPTIMIZER FUNCTION
# =========================================================
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

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔄 Rebalancing", "ℹ️ Über QAF"])

# =========================================================
# =======================  TAB 1  =========================
# =========================================================
with tab1:

    # -----------------------------------------------------
    # ONBOARDING
    # -----------------------------------------------------
    with st.expander("ℹ️ Kurzanleitung für neue Nutzer"):
        st.write("""
        Willkommen im QAF Dashboard!

        **So funktioniert es:**
        1. Lade eine CSV-Datei mit historischen Renditen hoch  
        2. Stelle die Optimierungsparameter ein  
        3. Starte die Optimierung  
        4. Sieh dir Gewichte, Kennzahlen und Risiko/Rendite-Profil an  
        5. Lade den Report herunter oder gehe zum Rebalancing-Tab  
        """)

    # -----------------------------------------------------
    # UPLOAD BEREICH
    # -----------------------------------------------------
    st.markdown("<hr style='border:1px solid #222;'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Portfolio-Datei hochladen (CSV)", type=["csv"])

    if uploaded_file:
        st.success("Datei erfolgreich hochgeladen!")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head())

    # -----------------------------------------------------
    # PARAMETER BEREICH
    # -----------------------------------------------------
    if uploaded_file:

        st.markdown("<hr style='border:1px solid #222;'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            target_return = st.number_input("Zielrendite", value=0.08, step=0.01)
            long_only = st.checkbox("Nur Long-Positionen erlauben", value=True)

        with col2:
            max_weight = st.number_input("Maximalgewicht pro Asset", value=0.3, step=0.05)
            min_weight = st.number_input("Minimalgewicht pro Asset", value=0.0, step=0.01)

        run_opt = st.button("Optimierung starten")

        # =====================================================
        # ==========  BLOOMBERG-STYLE ERGEBNISBLOCK  ==========
        # =====================================================
        if run_opt:

            st.markdown("<hr style='border:1px solid #222;'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background-color:#0A0A0A; padding:15px; border-radius:6px; border:1px solid #333;">
            <h2 style="color:#3EA6FF; margin:0;">Portfolio Analyse – Bloomberg Style</h2>
            </div>
            """, unsafe_allow_html=True)

            df = pd.read_csv(uploaded_file).dropna()
            weights = mean_variance_optimizer(df, long_only, max_weight, min_weight)

            if len(weights) == 0:
                st.error("Keine gültigen Assets nach Bereinigung.")
            else:
                mean_returns = df.mean() * 252
                cov = df.cov() * 252

                w = weights.values
                port_return = float(mean_returns @ w)
                port_vol = float(np.sqrt(w @ cov.values @ w))

                col_left, col_right = st.columns([1, 2])

                # LEFT COLUMN
                with col_left:
                    st.markdown("""
                    <div style="background-color:#111; padding:20px; border-radius:10px; border:1px solid #222;">
                    <h3 style="color:white;">Key Metrics</h3>
                    """, unsafe_allow_html=True)

                    st.metric("Erwartete Rendite", f"{round(port_return,4)}")
                    st.metric("Volatilität", f"{round(port_vol,4)}")

                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("### ")
                    st.markdown("""
                    <div style="background-color:#111; padding:20px; border-radius:10px; border:1px solid #222;">
                    <h3 style="color:white;">Optimierte Gewichte</h3>
                    """, unsafe_allow_html=True)

                    st.table(weights)

                    st.markdown("</div>", unsafe_allow_html=True)

                # RIGHT COLUMN
                with col_right:
                    st.markdown("""
                    <div style="background-color:#111; padding:20px; border-radius:10px; border:1px solid #222;">
                    <h3 style="color:white;">Risiko/Rendite-Profil</h3>
                    """, unsafe_allow_html=True)

                    chart_data_rr = pd.DataFrame({
                        "Name": ["Optimiertes Portfolio"],
                        "Return": [port_return],
                        "Volatility": [port_vol]
                    })

                    scatter = alt.Chart(chart_data_rr).mark_circle(size=160, color="#3EA6FF").encode(
                        x="Volatility",
                        y="Return",
                        tooltip=["Name", "Return", "Volatility"]
                    )

                    st.altair_chart(scatter, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # Efficient Frontier (Schritt 21)
                # ---------------------------------------------------------
                st.subheader("Effizienzlinie (Efficient Frontier)")

                num_portfolios = 5000
                returns = df
                mean_returns = returns.mean() * 252
                cov_matrix = returns.cov() * 252
                num_assets = len(mean_returns)

                results = np.zeros((3, num_portfolios))

                for i in range(num_portfolios):
                    weights_rand = np.random.random(num_assets)
                    weights_rand /= np.sum(weights_rand)

                    portfolio_return = np.sum(mean_returns * weights_rand)
                    portfolio_vol = np.sqrt(weights_rand.T @ cov_matrix.values @ weights_rand)

                    results[0, i] = portfolio_vol
                    results[1, i] = portfolio_return

                frontier_df = pd.DataFrame({
                    "Volatility": results[0],
                    "Return": results[1]
                })

                frontier_chart = alt.Chart(frontier_df).mark_circle(size=20, color="#555").encode(
                    x="Volatility",
                    y="Return"
                )

                optimized_point = alt.Chart(pd.DataFrame({
                    "Volatility": [port_vol],
                    "Return": [port_return]
                })).mark_circle(size=200, color="#3EA6FF").encode(
                    x="Volatility",
                    y="Return"
                )

                st.altair_chart(frontier_chart + optimized_point, use_container_width=True)

                # ---------------------------------------------------------
                # Capital Market Line (Schritt 22)
                # ---------------------------------------------------------
                st.subheader("Capital Market Line (CML)")

                risk_free_rate = 0.02
                frontier_df["Sharpe"] = (frontier_df["Return"] - risk_free_rate) / frontier_df["Volatility"]
                tangency_point = frontier_df.iloc[frontier_df["Sharpe"].idxmax()]

                cml_df = pd.DataFrame({
                    "Volatility": [0, tangency_point["Volatility"]],
                    "Return": [risk_free_rate, tangency_point["Return"]]
                })

                cml_line = alt.Chart(cml_df).mark_line(color="#3EA6FF", strokeWidth=3).encode(
                    x="Volatility",
                    y="Return"
                )

                tangency_chart = alt.Chart(pd.DataFrame({
                    "Volatility": [tangency_point["Volatility"]],
                    "Return": [tangency_point["Return"]]
                })).mark_circle(size=200, color="#FFCC00").encode(
                    x="Volatility",
                    y="Return"
                )

                opt_point_cml = alt.Chart(pd.DataFrame({
                    "Volatility": [port_vol],
                    "Return": [port_return]
                })).mark_circle(size=200, color="#00FFAA").encode(
                    x="Volatility",
                    y="Return"
                )

                st.altair_chart(cml_line + tangency_chart + opt_point_cml, use_container_width=True)

                st.markdown(f"""
                **Sharpe Ratio (Optimiertes Portfolio):** {round((port_return-risk_free_rate)/port_vol, 4)}  
                **Sharpe Ratio (Tangency Portfolio):** {round(tangency_point['Sharpe'], 4)}  
                """)

                # ---------------------------------------------------------
                # Sharpe-Ratio-Heatmap (Schritt 23)
                # ---------------------------------------------------------
                st.subheader("Sharpe-Ratio-Heatmap")

                heatmap = alt.Chart(frontier_df).mark_rect().encode(
                    x=alt.X("Volatility:Q", bin=alt.Bin(maxbins=30)),
                    y=alt.Y("Return:Q", bin=alt.Bin(maxbins=30)),
                    color=alt.Color("mean(Sharpe):Q", scale=alt.Scale(scheme="viridis")),
                    tooltip=["mean(Sharpe):Q"]
                )

                st.altair_chart(heatmap, use_container_width=True)

                # ---------------------------------------------------------
                # Risk Contribution (Schritt 24)
                # ---------------------------------------------------------
                st.subheader("Risikobeitrag je Asset")

                cov_matrix = cov.values
                portfolio_variance = w @ cov_matrix @ w
                mcr = cov_matrix @ w
                risk_contribution = (w * mcr) / portfolio_variance

                risk_df = pd.DataFrame({
                    "Asset": weights.index,
                    "Risk Contribution": risk_contribution
                })

                st.table(risk_df)

                risk_chart = alt.Chart(risk_df).mark_bar(color="#3EA6FF").encode(
                    x="Asset",
                    y="Risk Contribution",
                    tooltip=["Asset", "Risk Contribution"]
                )

                st.altair_chart(risk_chart, use_container_width=True)
                # ---------------------------------------------------------
                # Portfolio Snapshot Widget – Schritt 25
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio Snapshot – Auf einen Blick")
                
                # Kennzahlen berechnen
                num_assets = len(weights)
                avg_return = mean_returns.mean()
                avg_vol = np.sqrt(np.mean(np.diag(cov.values)))
                max_risk_contrib = risk_contribution.max()
                diversification = 1 - max_risk_contrib  # je höher, desto besser
                
                # Snapshot-Container
                st.markdown("""
                <div style="
                    background-color:#111111;
                    padding:20px;
                    border-radius:10px;
                    border:1px solid #222222;
                    margin-top:10px;
                ">
                <h3 style="color:white; margin-top:0;">Portfolio Snapshot</h3>
                </div>
                """, unsafe_allow_html=True)
                
                colA, colB, colC = st.columns(3)
                
                with colA:
                    st.metric("Anzahl Assets", num_assets)
                    st.metric("Durchschn. Rendite", f"{round(avg_return,4)}")
                
                with colB:
                    st.metric("Durchschn. Volatilität", f"{round(avg_vol,4)}")
                    st.metric("Sharpe Ratio", f"{round((port_return - risk_free_rate)/port_vol,4)}")
                
                with colC:
                    st.metric("Max. Risikobeitrag", f"{round(max_risk_contrib,4)}")
                    st.metric("Diversifikation", f"{round(diversification,4)}")
                # ---------------------------------------------------------
                # Correlation Heatmap – Schritt 26
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Korrelationen zwischen Assets")
                
                # Korrelationen berechnen
                corr = df.corr()
                
                # Für Heatmap in lange Form bringen
                corr_long = corr.reset_index().melt(id_vars="index")
                corr_long.columns = ["Asset1", "Asset2", "Correlation"]
                
                # Heatmap
                corr_chart = alt.Chart(corr_long).mark_rect().encode(
                    x=alt.X("Asset1:O", title="Asset 1"),
                    y=alt.Y("Asset2:O", title="Asset 2"),
                    color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue"), title="Korrelation"),
                    tooltip=["Asset1", "Asset2", "Correlation"]
                )
                
                st.altair_chart(corr_chart, use_container_width=True)
                # ---------------------------------------------------------
                # Correlation Heatmap – Schritt 26
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Korrelationen zwischen Assets")
                
                # Korrelationen berechnen
                corr = df.corr()
                
                # Für Heatmap in lange Form bringen
                corr_long = corr.reset_index().melt(id_vars="index")
                corr_long.columns = ["Asset1", "Asset2", "Correlation"]
                
                # Heatmap
                corr_chart = alt.Chart(corr_long).mark_rect().encode(
                    x=alt.X("Asset1:O", title="Asset 1"),
                    y=alt.Y("Asset2:O", title="Asset 2"),
                    color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue"), title="Korrelation"),
                    tooltip=["Asset1", "Asset2", "Correlation"]
                )
                
                st.altair_chart(corr_chart, use_container_width=True)

                # -------------------------------------------------
                # EXECUTIVE SUMMARY
                # -------------------------------------------------
                st.markdown("""
                <div style="background-color:#111; padding:20px; border-radius:10px; border:1px solid #222; margin-top:20px;">
                <h3 style="color:white;">Kurzfassung für Entscheider</h3>
                <p style="color:#CCC;">
                <b>Zielrendite:</b> """ + str(target_return) + """<br>
                <b>Long-Only:</b> """ + str(long_only) + """<br>
                <b>Optimiertes Portfolio:</b><br>
                Rendite <b>""" + str(round(port_return,4)) + """</b>, 
                Volatilität <b>""" + str(round(port_vol,4)) + """</b>
                </p>
                </div>
                """, unsafe_allow_html=True)

                # -------------------------------------------------
                # DOWNLOAD BUTTON
                # -------------------------------------------------
                report_df = pd.DataFrame({
                    "Asset": weights.index,
                    "Weight": weights.values
                })

                csv = report_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Optimierungsreport herunterladen",
                    data=csv,
                    file_name="qaf_optimierungsreport.csv",
                    mime="text/csv"
                )

    else:
        st.info("Bitte zuerst eine Datei hochladen, um die Optimierung zu aktivieren.")

# =========================================================
# =======================  TAB 2  =========================
# =========================================================
with tab2:

    st.subheader("Rebalancing – Kauf/Verkauf-Empfehlungen")

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

# =========================================================
# =======================  TAB 3  =========================
# =========================================================
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
    """)
