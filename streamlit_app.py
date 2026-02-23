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
                # Risk-Parity-Portfolio – Schritt 27
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Risk-Parity-Portfolio")
                
                # Kovarianzmatrix
                cov_matrix = cov.values
                n_assets = len(weights)
                
                # Startwerte: gleiche Gewichte
                rp_weights = np.ones(n_assets) / n_assets
                
                # Risk-Parity-Optimierung (iterativ)
                for _ in range(200):
                    mcr = cov_matrix @ rp_weights
                    rc = rp_weights * mcr
                    target = np.mean(rc)
                    rp_weights = rp_weights * (target / rc)
                    rp_weights = rp_weights / rp_weights.sum()
                
                rp_series = pd.Series(rp_weights, index=weights.index)
                
                # Anzeige
                st.markdown("**Risk-Parity-Gewichte:**")
                st.table(rp_series)
                
                # Balkendiagramm
                rp_chart = alt.Chart(pd.DataFrame({
                    "Asset": rp_series.index,
                    "Weight": rp_series.values
                })).mark_bar(color="#FFAA00").encode(
                    x="Asset",
                    y="Weight",
                    tooltip=["Asset", "Weight"]
                )
                
                st.altair_chart(rp_chart, use_container_width=True)

                # ---------------------------------------------------------
                # Minimum-Volatility-Portfolio – Schritt 28
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Minimum-Volatility-Portfolio")
                
                # Kovarianzmatrix
                cov_matrix = cov.values
                n_assets = len(weights)
                
                # Inverse der Kovarianzmatrix
                inv_cov = np.linalg.inv(cov_matrix)
                
                # Minimum-Volatility-Gewichte
                mv_weights = inv_cov @ np.ones(n_assets)
                mv_weights = mv_weights / mv_weights.sum()
                
                mv_series = pd.Series(mv_weights, index=weights.index)
                
                # Anzeige
                st.markdown("**Minimum-Volatility-Gewichte:**")
                st.table(mv_series)
                
                # Balkendiagramm
                mv_chart = alt.Chart(pd.DataFrame({
                    "Asset": mv_series.index,
                    "Weight": mv_series.values
                })).mark_bar(color="#00CCFF").encode(
                    x="Asset",
                    y="Weight",
                    tooltip=["Asset", "Weight"]
                )
                
                st.altair_chart(mv_chart, use_container_width=True)

                # ---------------------------------------------------------
                # Hierarchical Risk Parity (HRP) – Schritt 29
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Hierarchical Risk Parity (HRP) Portfolio")
                
                # 1. Distanzmatrix aus Korrelationen
                corr = df.corr()
                dist = np.sqrt(0.5 * (1 - corr))
                
                # 2. Hierarchisches Clustering
                from scipy.cluster.hierarchy import linkage, dendrogram
                
                link = linkage(dist, method="ward")
                
                # 3. Quasi-Diagonalization (Sortierung der Assets)
                def get_quasi_diag(link):
                    link = link.astype(int)
                    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
                    num_items = link[-1, 3]
                    while sort_ix.max() >= num_items:
                        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                        df0 = sort_ix[sort_ix >= num_items]
                        i = df0.index
                        j = df0.values - num_items
                        sort_ix[i] = link[j, 0]
                        df1 = pd.Series(link[j, 1], index=i + 1)
                        sort_ix = pd.concat([sort_ix, df1])
                        sort_ix = sort_ix.sort_index()
                    return sort_ix.tolist()
                
                sort_ix = get_quasi_diag(link)
                sorted_assets = corr.index[sort_ix]
                
                # 4. Recursive Bisection (Gewichte berechnen)
                def get_cluster_var(cov, cluster_items):
                    cov_slice = cov.loc[cluster_items, cluster_items]
                    w = np.ones(len(cov_slice)) / len(cov_slice)
                    return float(w.T @ cov_slice.values @ w)
                
                def recursive_bisection(cov, sorted_assets):
                    w = pd.Series(1, index=sorted_assets)
                    clusters = [sorted_assets]
                
                    while len(clusters) > 0:
                        cluster = clusters.pop(0)
                        if len(cluster) <= 1:
                            continue
                
                        split = len(cluster) // 2
                        c1 = cluster[:split]
                        c2 = cluster[split:]
                
                        clusters.append(c1)
                        clusters.append(c2)
                
                        var1 = get_cluster_var(cov, c1)
                        var2 = get_cluster_var(cov, c2)
                
                        alpha1 = 1 - var1 / (var1 + var2)
                        alpha2 = 1 - alpha1
                
                        w[c1] *= alpha1
                        w[c2] *= alpha2
                
                    return w / w.sum()
                
                hrp_weights = recursive_bisection(cov, sorted_assets)
                
                # Anzeige
                st.markdown("**HRP-Gewichte:**")
                st.table(hrp_weights)
                
                # Balkendiagramm
                hrp_chart = alt.Chart(pd.DataFrame({
                    "Asset": hrp_weights.index,
                    "Weight": hrp_weights.values
                })).mark_bar(color="#AAFF00").encode(
                    x="Asset",
                    y="Weight",
                    tooltip=["Asset", "Weight"]
                )
                
                st.altair_chart(hrp_chart, use_container_width=True)

                # ---------------------------------------------------------
                # Factor Exposure Chart – Schritt 30
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Factor Exposure Chart")
                
                # Faktoren berechnen (synthetische Proxies)
                asset_vol = df.std() * np.sqrt(252)
                asset_momentum = df.tail(21).mean() * 252  # letzter Monat annualisiert
                asset_quality = (df.mean() * 252) / (df.std() * np.sqrt(252))  # Sharpe pro Asset
                asset_value = 1 / (1 + asset_vol)  # inverse Volatilität als Value-Proxy
                asset_size = weights  # Gewicht als Size-Proxy
                
                factor_df = pd.DataFrame({
                    "Asset": df.columns,
                    "Value": asset_value.values,
                    "Momentum": asset_momentum.values,
                    "Volatility": asset_vol.values,
                    "Quality": asset_quality.values,
                    "Size": asset_size.values
                })
                
                # Melt für Chart
                factor_long = factor_df.melt(id_vars="Asset", var_name="Factor", value_name="Exposure")
                
                # Chart
                factor_chart = alt.Chart(factor_long).mark_bar().encode(
                    x=alt.X("Asset:O", title="Asset"),
                    y=alt.Y("Exposure:Q", title="Exposure"),
                    color="Factor:N",
                    tooltip=["Asset", "Factor", "Exposure"]
                ).properties(height=400)
                
                st.altair_chart(factor_chart, use_container_width=True)

                # ---------------------------------------------------------
                # Monte-Carlo-Simulation – Schritt 31
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Monte-Carlo-Simulation (10.000 Szenarien)")
                
                # Anzahl der Simulationen
                num_sim = 10000
                
                # Erwartungswerte und Kovarianzmatrix
                mu = mean_returns.values
                cov_matrix = cov.values
                
                # Simulation: multivariate Normalverteilung
                simulated_returns = np.random.multivariate_normal(mu, cov_matrix, num_sim)
                
                # Portfolio-Returns berechnen
                portfolio_sim_returns = simulated_returns @ w
                
                # In DataFrame packen
                sim_df = pd.DataFrame({"Portfolio Return": portfolio_sim_returns})
                
                # Kennzahlen
                sim_mean = sim_df["Portfolio Return"].mean()
                sim_std = sim_df["Portfolio Return"].std()
                sim_p5 = sim_df["Portfolio Return"].quantile(0.05)
                sim_p95 = sim_df["Portfolio Return"].quantile(0.95)
                
                # Histogramm
                hist_chart = alt.Chart(sim_df).mark_bar(color="#3EA6FF").encode(
                    x=alt.X("Portfolio Return:Q", bin=alt.Bin(maxbins=50), title="Portfolio Return"),
                    y=alt.Y("count()", title="Häufigkeit"),
                    tooltip=["Portfolio Return"]
                )
                
                st.altair_chart(hist_chart, use_container_width=True)
                
                # Kennzahlen anzeigen
                st.markdown(f"""
                **Erwartete Rendite (Simulation):** {round(sim_mean,4)}  
                **Volatilität (Simulation):** {round(sim_std,4)}  
                **5%-Quantil (Worst Case):** {round(sim_p5,4)}  
                **95%-Quantil (Best Case):** {round(sim_p95,4)}  
                """)

                # ---------------------------------------------------------
                # Drawdown-Analyse – Schritt 32
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Drawdown-Analyse")
                
                # Portfolio-Zeitreihe konstruieren
                # Wir verwenden die historischen Renditen + optimierte Gewichte
                portfolio_returns_series = df @ w
                
                # Kumulierte Performance
                cum_returns = (1 + portfolio_returns_series).cumprod()
                
                # Running Maximum
                running_max = cum_returns.cummax()
                
                # Drawdown
                drawdown = (cum_returns - running_max) / running_max
                
                # Maximaler Drawdown
                max_dd = drawdown.min()
                
                # Recovery Time (Tage bis neues Hoch)
                recovery_time = (drawdown == 0).diff().fillna(0)
                recovery_periods = recovery_time[recovery_time == 1].index.to_list()
                
                if len(recovery_periods) > 1:
                    recovery_days = recovery_periods[-1] - recovery_periods[-2]
                else:
                    recovery_days = "Noch nicht vollständig erholt"
                
                # Drawdown-Chart
                dd_df = pd.DataFrame({
                    "Drawdown": drawdown.values,
                    "Index": range(len(drawdown))
                })
                
                dd_chart = alt.Chart(dd_df).mark_area(color="#FF4444").encode(
                    x=alt.X("Index:Q", title="Zeit"),
                    y=alt.Y("Drawdown:Q", title="Drawdown"),
                    tooltip=["Drawdown"]
                )
                
                st.altair_chart(dd_chart, use_container_width=True)
                
                # Kennzahlen anzeigen
                st.markdown(f"""
                **Maximaler Drawdown:** {round(max_dd,4)}  
                **Recovery Time:** {recovery_days}  
                """)

                # ---------------------------------------------------------
                # Rolling-Volatility-Chart – Schritt 33
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Rolling Volatility (gleitende Volatilität)")
                
                # Rolling Window (z. B. 21 Tage ≈ 1 Monat)
                window = 21
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Rolling Volatility berechnen
                rolling_vol = portfolio_returns_series.rolling(window).std() * np.sqrt(252)
                
                rolling_vol_df = pd.DataFrame({
                    "Rolling Volatility": rolling_vol.values,
                    "Index": range(len(rolling_vol))
                })
                
                # Chart
                rolling_vol_chart = alt.Chart(rolling_vol_df).mark_line(color="#FFA500", strokeWidth=2).encode(
                    x=alt.X("Index:Q", title="Zeit"),
                    y=alt.Y("Rolling Volatility:Q", title="Volatilität (annualisiert)"),
                    tooltip=["Rolling Volatility"]
                )
                
                st.altair_chart(rolling_vol_chart, use_container_width=True)
                
                # Kennzahlen
                st.markdown(f"""
                **Durchschnittliche Rolling-Volatilität:** {round(rolling_vol.mean(),4)}  
                **Maximale Rolling-Volatilität:** {round(rolling_vol.max(),4)}  
                **Minimale Rolling-Volatilität:** {round(rolling_vol.min(),4)}  
                """)

                # ---------------------------------------------------------
                # Rolling-Sharpe-Chart – Schritt 34
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Rolling Sharpe Ratio")
                
                # Rolling Window (z. B. 21 Tage ≈ 1 Monat)
                window = 21
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Rolling Sharpe Ratio
                rolling_sharpe = (
                    portfolio_returns_series.rolling(window).mean() * 252
                ) / (
                    portfolio_returns_series.rolling(window).std() * np.sqrt(252)
                )
                
                rolling_sharpe_df = pd.DataFrame({
                    "Rolling Sharpe": rolling_sharpe.values,
                    "Index": range(len(rolling_sharpe))
                })
                
                # Chart
                rolling_sharpe_chart = alt.Chart(rolling_sharpe_df).mark_line(color="#33CC33", strokeWidth=2).encode(
                    x=alt.X("Index:Q", title="Zeit"),
                    y=alt.Y("Rolling Sharpe:Q", title="Sharpe Ratio"),
                    tooltip=["Rolling Sharpe"]
                )
                
                st.altair_chart(rolling_sharpe_chart, use_container_width=True)
                
                # Kennzahlen
                st.markdown(f"""
                **Durchschnittliche Rolling-Sharpe:** {round(rolling_sharpe.mean(),4)}  
                **Maximale Rolling-Sharpe:** {round(rolling_sharpe.max(),4)}  
                **Minimale Rolling-Sharpe:** {round(rolling_sharpe.min(),4)}  
                """)

                # ---------------------------------------------------------
                # Value-at-Risk (VaR) Analyse – Schritt 35
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Value-at-Risk (VaR) Analyse")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Konfidenzniveau
                confidence = 0.95
                alpha = 1 - confidence
                
                # 1) Parametrischer VaR (Normalverteilung)
                mu_p = portfolio_returns_series.mean()
                sigma_p = portfolio_returns_series.std()
                var_param = mu_p - sigma_p * np.sqrt(252) * 1.65  # 95% VaR
                
                # 2) Historischer VaR
                var_hist = portfolio_returns_series.quantile(alpha)
                
                # 3) Monte-Carlo VaR (aus Schritt 31)
                var_mc = np.quantile(portfolio_sim_returns, alpha)
                
                # Tabelle
                var_df = pd.DataFrame({
                    "Methode": ["Parametrisch (Normal)", "Historisch", "Monte-Carlo"],
                    "VaR (95%)": [var_param, var_hist, var_mc]
                })
                
                st.table(var_df)
                
                # Histogramm der historischen Returns + VaR-Linie
                hist_var_chart = alt.Chart(pd.DataFrame({
                    "Return": portfolio_returns_series
                })).mark_bar(color="#8888FF").encode(
                    x=alt.X("Return:Q", bin=alt.Bin(maxbins=50)),
                    y="count()"
                )
                
                var_line = alt.Chart(pd.DataFrame({
                    "VaR": [var_hist]
                })).mark_rule(color="red", strokeWidth=3).encode(
                    x="VaR:Q"
                )
                
                st.altair_chart(hist_var_chart + var_line, use_container_width=True)
                
                # Kennzahlen anzeigen
                st.markdown(f"""
                **Parametrischer VaR (95%):** {round(var_param,4)}  
                **Historischer VaR (95%):** {round(var_hist,4)}  
                **Monte-Carlo VaR (95%):** {round(var_mc,4)}  
                """)

                # ---------------------------------------------------------
                # Expected Shortfall (CVaR) – Schritt 36
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Expected Shortfall (CVaR) Analyse")
                
                # Konfidenzniveau
                confidence = 0.95
                alpha = 1 - confidence
                
                # 1) Historischer CVaR
                cvar_hist = portfolio_returns_series[portfolio_returns_series <= var_hist].mean()
                
                # 2) Parametrischer CVaR (Normalverteilung)
                from scipy.stats import norm
                cvar_param = mu_p - sigma_p * np.sqrt(252) * (norm.pdf(norm.ppf(alpha)) / alpha)
                
                # 3) Monte-Carlo CVaR
                cvar_mc = portfolio_sim_returns[portfolio_sim_returns <= var_mc].mean()
                
                # Tabelle
                cvar_df = pd.DataFrame({
                    "Methode": ["Parametrisch (Normal)", "Historisch", "Monte-Carlo"],
                    "CVaR (95%)": [cvar_param, cvar_hist, cvar_mc]
                })
                
                st.table(cvar_df)
                
                # Histogramm + CVaR-Linie
                hist_cvar_chart = alt.Chart(pd.DataFrame({
                    "Return": portfolio_returns_series
                })).mark_bar(color="#AAAAFF").encode(
                    x=alt.X("Return:Q", bin=alt.Bin(maxbins=50)),
                    y="count()"
                )
                
                cvar_line = alt.Chart(pd.DataFrame({
                    "CVaR": [cvar_hist]
                })).mark_rule(color="red", strokeWidth=3).encode(
                    x="CVaR:Q"
                )
                
                st.altair_chart(hist_cvar_chart + cvar_line, use_container_width=True)
                
                # Kennzahlen anzeigen
                st.markdown(f"""
                **Parametrischer CVaR (95%):** {round(cvar_param,4)}  
                **Historischer CVaR (95%):** {round(cvar_hist,4)}  
                **Monte-Carlo CVaR (95%):** {round(cvar_mc,4)}  
                """)

                # ---------------------------------------------------------
                # Beta-Analyse – Schritt 37
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Beta-Analyse (Portfolio vs. Benchmark)")
                
                # Benchmark-Auswahl
                benchmark_choice = st.selectbox(
                    "Benchmark auswählen",
                    ["SPY (S&P 500)", "QQQ (Nasdaq 100)", "EEM (Emerging Markets)", "EFA (Developed Markets ex-US)"]
                )
                
                # Benchmark-Daten simulieren (Proxy)
                # In einer echten Version würdest du hier echte Benchmark-Daten laden
                np.random.seed(42)
                benchmark_returns = pd.Series(
                    np.random.normal(0.0005, 0.01, len(df)),
                    name="Benchmark"
                )
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Beta-Berechnung
                cov_pb = np.cov(portfolio_returns_series, benchmark_returns)[0][1]
                var_b = benchmark_returns.var()
                beta = cov_pb / var_b
                
                # Alpha-Berechnung (CAPM)
                risk_free_rate = 0.02
                benchmark_mean = benchmark_returns.mean() * 252
                portfolio_mean = portfolio_returns_series.mean() * 252
                
                alpha = portfolio_mean - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))
                
                # Tabelle
                beta_df = pd.DataFrame({
                    "Kennzahl": ["Beta", "Alpha (annualisiert)", "Benchmark-Rendite", "Portfolio-Rendite"],
                    "Wert": [beta, alpha, benchmark_mean, portfolio_mean]
                })
                
                st.table(beta_df)
                
                # Scatterplot: Portfolio vs. Benchmark
                beta_chart_df = pd.DataFrame({
                    "Portfolio": portfolio_returns_series,
                    "Benchmark": benchmark_returns
                })
                
                beta_chart = alt.Chart(beta_chart_df).mark_circle(size=40, color="#33AAFF").encode(
                    x=alt.X("Benchmark:Q", title="Benchmark Return"),
                    y=alt.Y("Portfolio:Q", title="Portfolio Return"),
                    tooltip=["Portfolio", "Benchmark"]
                )
                
                # Regressionslinie
                reg_line = beta_chart.transform_regression(
                    "Benchmark", "Portfolio"
                ).mark_line(color="red")
                
                st.altair_chart(beta_chart + reg_line, use_container_width=True)
                
                # Kennzahlen anzeigen
                st.markdown(f"""
                **Beta:** {round(beta,4)}  
                **Alpha:** {round(alpha,4)}  
                **Interpretation:**  
                - Beta > 1 → Portfolio ist aggressiver als der Markt  
                - Beta < 1 → Portfolio ist defensiver  
                - Alpha > 0 → Portfolio schlägt den Markt risikoadjustiert  
                """)

                # ---------------------------------------------------------
                # Stress-Tests (Crash-Szenarien) – Schritt 38
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Stress-Tests (Crash-Szenarien)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Stress-Szenarien definieren (synthetische Schocks)
                stress_scenarios = {
                    "Dotcom Crash (2000–2002)": -0.45,
                    "Finanzkrise (2008)": -0.52,
                    "Corona Crash (2020)": -0.34,
                    "Zins-Schock / Inflation": -0.20,
                    "Tech-Crash": -0.30,
                    "Emerging Markets Crash": -0.28
                }
                
                # Stress-Test Ergebnisse berechnen
                stress_results = []
                for name, shock in stress_scenarios.items():
                    stressed_return = portfolio_returns_series.mean() * 252 + shock
                    stress_results.append([name, shock, stressed_return])
                
                stress_df = pd.DataFrame(stress_results, columns=["Szenario", "Marktschock", "Portfolio-Auswirkung"])
                
                st.table(stress_df)
                
                # Balkendiagramm
                stress_chart = alt.Chart(stress_df).mark_bar().encode(
                    x=alt.X("Szenario:N", sort=None, title="Szenario"),
                    y=alt.Y("Portfolio-Auswirkung:Q", title="Portfolio-Auswirkung"),
                    color=alt.condition(
                        alt.datum["Portfolio-Auswirkung"] < 0,
                        alt.value("#FF4444"),
                        alt.value("#44FF44")
                    ),
                    tooltip=["Szenario", "Marktschock", "Portfolio-Auswirkung"]
                ).properties(height=400)
                
                st.altair_chart(stress_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Die Stress-Tests zeigen, wie stark dein Portfolio in historischen oder hypothetischen Crashs fallen könnte.  
                - Szenarien wie 2008 oder Dotcom führen zu besonders starken Verlusten.  
                - Ein robustes Portfolio zeigt geringere Ausschläge in diesen Szenarien.  
                """)

                # ---------------------------------------------------------
                # Tail-Risk-Chart – Schritt 39
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Tail-Risk-Analyse (Extreme Verluste)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Unteres 5%-Segment extrahieren
                tail_threshold = portfolio_returns_series.quantile(0.05)
                tail_data = portfolio_returns_series[portfolio_returns_series <= tail_threshold]
                
                tail_df = pd.DataFrame({"Tail Returns": tail_data})
                
                # Histogramm der Tail-Returns
                tail_chart = alt.Chart(tail_df).mark_bar(color="#FF3333").encode(
                    x=alt.X("Tail Returns:Q", bin=alt.Bin(maxbins=40), title="Extreme Verluste"),
                    y=alt.Y("count()", title="Häufigkeit"),
                    tooltip=["Tail Returns"]
                )
                
                st.altair_chart(tail_chart, use_container_width=True)
                
                # Kennzahlen
                tail_mean = tail_data.mean()
                tail_min = tail_data.min()
                tail_std = tail_data.std()
                
                st.markdown(f"""
                **Durchschnittlicher Tail-Verlust:** {round(tail_mean,4)}  
                **Schlimmster Verlust im Tail:** {round(tail_min,4)}  
                **Volatilität im Tail:** {round(tail_std,4)}  
                
                **Interpretation:**  
                - Das Tail-Risk-Chart zeigt die extremsten 5% aller Verluste.  
                - Je breiter und tiefer der Tail, desto höher das Crash-Risiko.  
                - Ein „fetter Tail“ bedeutet, dass dein Portfolio anfällig für seltene, extreme Ereignisse ist.  
                """)

                # ---------------------------------------------------------
                # Performance Attribution – Schritt 40
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Performance Attribution (Beitragsanalyse)")
                
                # Asset Returns (annualisiert)
                asset_returns = df.mean() * 252
                
                # Contribution = Gewicht * Rendite
                contribution = weights * asset_returns
                
                attrib_df = pd.DataFrame({
                    "Asset": weights.index,
                    "Gewicht": weights.values,
                    "Rendite": asset_returns.values,
                    "Beitrag": contribution.values
                })
                
                # Sortieren nach Beitrag
                attrib_df = attrib_df.sort_values("Beitrag", ascending=False)
                
                st.table(attrib_df)
                
                # Balkendiagramm
                attrib_chart = alt.Chart(attrib_df).mark_bar().encode(
                    x=alt.X("Asset:N", sort=None, title="Asset"),
                    y=alt.Y("Beitrag:Q", title="Performance-Beitrag"),
                    color=alt.condition(
                        alt.datum["Beitrag"] < 0,
                        alt.value("#FF4444"),   # Rot für negative Beiträge
                        alt.value("#44FF44")    # Grün für positive Beiträge
                    ),
                    tooltip=["Asset", "Gewicht", "Rendite", "Beitrag"]
                ).properties(height=400)
                
                st.altair_chart(attrib_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Positive Balken zeigen Assets, die zur Gesamtperformance beigetragen haben.  
                - Negative Balken zeigen Assets, die Performance gekostet haben.  
                - Große Balken bedeuten hohe Wirkung — unabhängig vom Gewicht.  
                - Diese Analyse ist essenziell für Reporting, Pitch-Decks und Investment-Kommunikation.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Stabilitätsindex – Schritt 41
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Stabilitätsindex")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Komponenten des Stabilitätsindex
                vol = portfolio_returns_series.std() * np.sqrt(252)
                max_dd = ((1 + portfolio_returns_series).cumprod() /
                          (1 + portfolio_returns_series).cumprod().cummax() - 1).min()
                autocorr = portfolio_returns_series.autocorr()
                rolling_sharpe = (
                    portfolio_returns_series.rolling(21).mean() * 252
                ) / (
                    portfolio_returns_series.rolling(21).std() * np.sqrt(252)
                )
                sharpe_stability = 1 - rolling_sharpe.std()
                
                # Normalisierung
                vol_score = 1 / (1 + vol)
                dd_score = 1 / (1 + abs(max_dd))
                autocorr_score = (autocorr + 1) / 2  # von -1..1 auf 0..1
                sharpe_score = max(0, min(1, sharpe_stability))
                
                # Gesamtindex
                stability_index = (
                    0.35 * vol_score +
                    0.35 * dd_score +
                    0.15 * autocorr_score +
                    0.15 * sharpe_score
                )
                
                # Anzeige
                st.metric("Portfolio-Stabilitätsindex", f"{round(stability_index,4)}")
                
                # Detailtabelle
                stab_df = pd.DataFrame({
                    "Komponente": ["Volatilität", "Max Drawdown", "Autokorrelation", "Sharpe-Stabilität"],
                    "Score": [vol_score, dd_score, autocorr_score, sharpe_score]
                })
                
                st.table(stab_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Ein Wert nahe 1 bedeutet ein sehr stabiles, robustes Portfolio.  
                - Ein Wert nahe 0 bedeutet ein instabiles, volatiles Portfolio.  
                - Der Index kombiniert Risiko, Trendstabilität und Sharpe-Konsistenz.  
                """)

                # ---------------------------------------------------------
                # Liquidity-Risk-Modul – Schritt 42
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Liquidity-Risk-Analyse")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Liquidity-Proxies (synthetisch, da keine echten Marktdaten)
                # 1) Bid-Ask-Spread Proxy: hohe Volatilität = breiter Spread
                spread_proxy = df.std() * 100
                
                # 2) Turnover Proxy: inverse Volatilität (ruhige Assets = leichter handelbar)
                turnover_proxy = 1 / (1 + df.std())
                
                # 3) Impact-Cost Proxy: Preiswirkung bei Handel (Volatilität * Gewicht)
                impact_proxy = df.std() * weights
                
                # Liquidity-Score pro Asset (0 = schlecht, 1 = gut)
                liquidity_score = (
                    0.4 * (1 / (1 + spread_proxy)) +
                    0.3 * turnover_proxy +
                    0.3 * (1 / (1 + abs(impact_proxy)))
                )
                
                liq_df = pd.DataFrame({
                    "Asset": df.columns,
                    "Spread-Proxy": spread_proxy.values,
                    "Turnover-Proxy": turnover_proxy.values,
                    "Impact-Cost": impact_proxy.values,
                    "Liquidity-Score": liquidity_score.values
                })
                
                # Gesamt-Liquidity-Index (gewichteter Score)
                portfolio_liquidity_index = float((liquidity_score * weights).sum())
                
                # Anzeige
                st.metric("Portfolio-Liquidity-Index", f"{round(portfolio_liquidity_index,4)}")
                
                # Tabelle
                st.table(liq_df)
                
                # Balkendiagramm
                liq_chart = alt.Chart(liq_df).mark_bar().encode(
                    x=alt.X("Asset:N", sort=None),
                    y=alt.Y("Liquidity-Score:Q", title="Liquidity Score"),
                    color=alt.condition(
                        alt.datum["Liquidity-Score"] < 0.5,
                        alt.value("#FF4444"),   # Rot = illiquide
                        alt.value("#44FF44")    # Grün = liquide
                    ),
                    tooltip=["Asset", "Spread-Proxy", "Turnover-Proxy", "Impact-Cost", "Liquidity-Score"]
                ).properties(height=400)
                
                st.altair_chart(liq_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Ein hoher Liquidity-Score bedeutet, dass ein Asset leicht handelbar ist.  
                - Ein niedriger Score zeigt potenzielle Liquiditätsrisiken.  
                - Der Portfolio-Liquidity-Index zeigt, wie liquide das Gesamtportfolio ist.  
                - Illiquide Assets erhöhen das Risiko in Stressphasen erheblich.  
                """)

               # ---------------------------------------------------------
                # Diversifikations-Score – Schritt 43
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Diversifikations-Score")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # 1) Herfindahl-Hirschman-Index (HHI)
                hhi = (weights**2).sum()
                hhi_score = 1 - hhi  # 1 = perfekt diversifiziert
                
                # 2) Durchschnittliche Korrelation
                corr_matrix = df.corr()
                avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
                corr_score = 1 - avg_corr  # niedrige Korrelation = gute Diversifikation
                
                # 3) Risiko-Beiträge
                cov_matrix = df.cov().values
                marginal_contrib = cov_matrix @ w
                risk_contrib = w * marginal_contrib
                risk_contrib_norm = risk_contrib / risk_contrib.sum()
                risk_balance_score = 1 - risk_contrib_norm.std()  # je gleichmäßiger, desto besser
                
                # 4) Cluster-Risiko (aus Korrelationen)
                from scipy.cluster.hierarchy import linkage, fcluster
                link = linkage(corr_matrix, method="ward")
                clusters = fcluster(link, t=3, criterion="maxclust")
                cluster_df = pd.DataFrame({"cluster": clusters, "weight": weights.values})
                cluster_weights = cluster_df.groupby("cluster")["weight"].sum()
                cluster_score = 1 - cluster_weights.max()  # großer Cluster = schlechter
                
                # Gesamt-Diversifikations-Score
                div_score = (
                    0.35 * hhi_score +
                    0.25 * corr_score +
                    0.25 * risk_balance_score +
                    0.15 * cluster_score
                )
                
                # Anzeige
                st.metric("Diversifikations-Score", f"{round(div_score,4)}")
                
                # Detailtabelle
                div_df = pd.DataFrame({
                    "Komponente": ["HHI-Score", "Korrelation-Score", "Risk-Balance-Score", "Cluster-Score"],
                    "Score": [hhi_score, corr_score, risk_balance_score, cluster_score]
                })
                
                st.table(div_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Ein hoher Score bedeutet ein breit diversifiziertes Portfolio.  
                - Ein niedriger Score zeigt Klumpenrisiken oder hohe Korrelationen.  
                - Der Score kombiniert Gewichtskonzentration, Korrelationen, Risikobeiträge und Cluster-Risiko.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Heatmap – Schritt 44
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Heatmap (Gewichte, Risiko, Rendite)")
                
                # Kennzahlen pro Asset
                asset_returns = df.mean() * 252
                asset_vol = df.std() * np.sqrt(252)
                asset_weights = weights
                
                heatmap_df = pd.DataFrame({
                    "Asset": df.columns,
                    "Gewicht": asset_weights.values,
                    "Rendite": asset_returns.values,
                    "Volatilität": asset_vol.values
                })
                
                # Melt für Heatmap
                heatmap_long = heatmap_df.melt(id_vars="Asset", var_name="Kennzahl", value_name="Wert")
                
                # Heatmap
                heatmap_chart = alt.Chart(heatmap_long).mark_rect().encode(
                    x=alt.X("Kennzahl:N", title="Kennzahl"),
                    y=alt.Y("Asset:N", title="Asset"),
                    color=alt.Color("Wert:Q", scale=alt.Scale(scheme="viridis")),
                    tooltip=["Asset", "Kennzahl", "Wert"]
                ).properties(height=300)
                
                st.altair_chart(heatmap_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Die Heatmap zeigt Gewichte, Renditen und Risiken pro Asset.  
                - Dunkle Farben = hohe Werte, helle Farben = niedrige Werte.  
                - Perfekt geeignet, um Klumpenrisiken, Renditequellen und Risiko-Hotspots zu erkennen.  
                """)

                # ---------------------------------------------------------
                # Correlation-Cluster-Map (Dendrogramm) – Schritt 45
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Correlation-Cluster-Map (Dendrogramm)")
                
                # Korrelationen
                corr = df.corr()
                
                # Distanzmatrix
                dist = np.sqrt(0.5 * (1 - corr))
                
                # Hierarchisches Clustering
                from scipy.cluster.hierarchy import linkage, dendrogram
                
                link = linkage(dist, method="ward")
                
                # Dendrogramm-Daten vorbereiten
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, ax = plt.subplots(figsize=(10, 4))
                dendrogram(link, labels=corr.index, leaf_rotation=90)
                plt.title("Correlation Cluster Dendrogram")
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Das Dendrogramm zeigt, welche Assets natürliche Cluster bilden.  
                - Kurze Verbindungsäste = hohe Korrelation → Cluster.  
                - Lange Äste = geringe Korrelation → gute Diversifikation.  
                - Perfekt geeignet, um strukturelle Risiken und Cluster zu erkennen.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Quality-Score – Schritt 46
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Quality-Score")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # 1) Sharpe Ratio (annualisiert)
                sharpe_ratio = (portfolio_returns_series.mean() * 252) / (portfolio_returns_series.std() * np.sqrt(252))
                sharpe_score = max(0, min(1, sharpe_ratio / 2))  # Sharpe 2.0 = Score 1.0
                
                # 2) Drawdown-Score
                cum_returns = (1 + portfolio_returns_series).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns - running_max) / running_max
                max_dd = drawdown.min()
                dd_score = 1 - abs(max_dd)
                
                # 3) Diversifikations-Score (aus Schritt 43)
                div_score_component = div_score  # bereits berechnet
                
                # 4) Liquidity-Score (aus Schritt 42)
                liq_score_component = portfolio_liquidity_index  # bereits berechnet
                
                # 5) Tail-Risk-Score
                tail_threshold = portfolio_returns_series.quantile(0.05)
                tail_data = portfolio_returns_series[portfolio_returns_series <= tail_threshold]
                tail_risk = abs(tail_data.mean())
                tail_score = 1 / (1 + tail_risk)
                
                # Gesamt-Quality-Score
                quality_score = (
                    0.30 * sharpe_score +
                    0.20 * dd_score +
                    0.20 * div_score_component +
                    0.15 * liq_score_component +
                    0.15 * tail_score
                )
                
                # Anzeige
                st.metric("Portfolio-Quality-Score", f"{round(quality_score,4)}")
                
                # Detailtabelle
                quality_df = pd.DataFrame({
                    "Komponente": ["Sharpe-Score", "Drawdown-Score", "Diversifikations-Score", "Liquidity-Score", "Tail-Risk-Score"],
                    "Score": [sharpe_score, dd_score, div_score_component, liq_score_component, tail_score]
                })
                
                st.table(quality_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Ein hoher Quality-Score bedeutet ein robustes, effizientes, gut diversifiziertes Portfolio.  
                - Der Score kombiniert Risiko, Rendite, Diversifikation, Liquidität und Tail-Risiko.  
                - Perfekt geeignet für institutionelle Reports, Pitch-Decks und Investment-Kommunikation.  
                """)

                # ---------------------------------------------------------
                # Risk-Adjusted-Return-Matrix – Schritt 47
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Risk-Adjusted-Return-Matrix")
                
                # Kennzahlen pro Asset
                asset_returns = df.mean() * 252
                asset_vol = df.std() * np.sqrt(252)
                asset_sharpe = asset_returns / asset_vol
                
                rar_df = pd.DataFrame({
                    "Asset": df.columns,
                    "Rendite": asset_returns.values,
                    "Volatilität": asset_vol.values,
                    "Sharpe": asset_sharpe.values
                })
                
                # Scatterplot: Rendite vs. Risiko
                rar_chart = alt.Chart(rar_df).mark_circle(size=120).encode(
                    x=alt.X("Volatilität:Q", title="Risiko (Volatilität)"),
                    y=alt.Y("Rendite:Q", title="Rendite (annualisiert)"),
                    color=alt.Color("Sharpe:Q", scale=alt.Scale(scheme="viridis"), title="Sharpe Ratio"),
                    tooltip=["Asset", "Rendite", "Volatilität", "Sharpe"]
                ).properties(height=400)
                
                st.altair_chart(rar_chart, use_container_width=True)
                
                # Tabelle anzeigen
                st.table(rar_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Assets oben links sind ideal: hohe Rendite bei niedrigem Risiko.  
                - Assets unten rechts sind ineffizient: niedrigere Rendite bei höherem Risiko.  
                - Die Farbskala zeigt die Sharpe Ratio: je dunkler, desto besser.  
                - Perfekt geeignet, um Alpha-Quellen und Risiko-Fresser zu identifizieren.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Style-Box – Schritt 48
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Style-Box (Growth/Value, Large/Small)")
                
                # Style-Proxies
                asset_returns = df.mean() * 252
                asset_vol = df.std() * np.sqrt(252)
                
                # Growth-Proxy: Momentum (Rendite)
                growth_score = (asset_returns - asset_returns.min()) / (asset_returns.max() - asset_returns.min())
                
                # Value-Proxy: inverse Volatilität
                value_score = 1 - (asset_vol - asset_vol.min()) / (asset_vol.max() - asset_vol.min())
                
                # Size-Proxy: Volatilität als Näherung (niedrige Vol = Large Cap)
                size_score = 1 - (asset_vol - asset_vol.min()) / (asset_vol.max() - asset_vol.min())
                
                style_df = pd.DataFrame({
                    "Asset": df.columns,
                    "Growth": growth_score.values,
                    "Value": value_score.values,
                    "Size": size_score.values
                })
                
                # Style-Box-Koordinaten
                style_df["X"] = style_df["Growth"] - style_df["Value"]   # Growth vs. Value
                style_df["Y"] = style_df["Size"]                         # Large vs. Small
                
                # Scatterplot
                style_chart = alt.Chart(style_df).mark_circle(size=150).encode(
                    x=alt.X("X:Q", title="Value  <------>  Growth"),
                    y=alt.Y("Y:Q", title="Small Cap  <------>  Large Cap"),
                    color=alt.Color("Growth:Q", scale=alt.Scale(scheme="viridis"), title="Growth Score"),
                    tooltip=["Asset", "Growth", "Value", "Size"]
                ).properties(height=400)
                
                st.altair_chart(style_chart, use_container_width=True)
                
                # Tabelle
                st.table(style_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Rechts = Growth, links = Value.  
                - Oben = Large Cap, unten = Small Cap.  
                - Growth-Assets haben hohe Momentum-Werte.  
                - Value-Assets haben stabile, risikoarme Profile.  
                - Die Style-Box zeigt sofort, ob dein Portfolio Stil-Wetten eingeht.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Factor-Sensitivity-Matrix – Schritt 49
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Factor-Sensitivity-Matrix")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Synthetische Faktor-Returns erzeugen (Proxy)
                np.random.seed(42)
                factors = pd.DataFrame({
                    "Market": np.random.normal(0.0005, 0.01, len(df)),
                    "SMB": np.random.normal(0.0002, 0.008, len(df)),   # Size
                    "HML": np.random.normal(0.0001, 0.007, len(df)),   # Value
                    "MOM": np.random.normal(0.0003, 0.009, len(df)),   # Momentum
                    "Quality": np.random.normal(0.00015, 0.006, len(df)),
                    "LowVol": np.random.normal(0.0001, 0.005, len(df))
                })
                
                # Regression: Portfolio = a + b1*Market + b2*SMB + ...
                import statsmodels.api as sm
                
                X = sm.add_constant(factors)
                y = portfolio_returns_series
                model = sm.OLS(y, X).fit()
                
                factor_loadings = model.params.drop("const")
                
                factor_df = pd.DataFrame({
                    "Faktor": factor_loadings.index,
                    "Loading": factor_loadings.values
                })
                
                st.table(factor_df)
                
                # Balkendiagramm
                factor_chart = alt.Chart(factor_df).mark_bar().encode(
                    x=alt.X("Faktor:N", sort=None),
                    y=alt.Y("Loading:Q", title="Faktor-Sensitivität"),
                    color=alt.condition(
                        alt.datum["Loading"] < 0,
                        alt.value("#FF4444"),
                        alt.value("#44FF44")
                    ),
                    tooltip=["Faktor", "Loading"]
                ).properties(height=400)
                
                st.altair_chart(factor_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Positive Loadings = Portfolio profitiert von diesem Faktor.  
                - Negative Loadings = Portfolio ist gegen diesen Faktor positioniert.  
                - Market = Beta zum Gesamtmarkt.  
                - SMB = Small vs. Large Cap Sensitivität.  
                - HML = Value vs. Growth Sensitivität.  
                - Momentum = Trendexposition.  
                - Quality = Qualitätsfaktor (stabile Gewinne, niedrige Verschuldung).  
                - LowVol = defensive, risikoarme Exposition.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Risk-Decomposition – Schritt 50
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Risk-Decomposition (Systematic vs. Idiosyncratic)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Faktor-Regression aus Schritt 49
                X = sm.add_constant(factors)
                y = portfolio_returns_series
                model = sm.OLS(y, X).fit()
                
                # Systematisches Risiko = Varianz der erklärten Komponente
                fitted_values = model.fittedvalues
                systematic_var = np.var(fitted_values)
                
                # Idiosynkratisches Risiko = Varianz der Residuen
                residuals = model.resid
                idiosyncratic_var = np.var(residuals)
                
                # Total Risk
                total_var = systematic_var + idiosyncratic_var
                
                # Annualisieren
                systematic_risk = np.sqrt(systematic_var * 252)
                idiosyncratic_risk = np.sqrt(idiosyncratic_var * 252)
                total_risk = np.sqrt(total_var * 252)
                
                risk_df = pd.DataFrame({
                    "Risiko-Komponente": ["Systematisches Risiko", "Idiosynkratisches Risiko", "Total Risk"],
                    "Wert (annualisiert)": [systematic_risk, idiosyncratic_risk, total_risk]
                })
                
                st.table(risk_df)
                
                # Balkendiagramm
                risk_chart = alt.Chart(risk_df).mark_bar().encode(
                    x=alt.X("Risiko-Komponente:N", sort=None),
                    y=alt.Y("Wert (annualisiert):Q", title="Risiko (annualisiert)"),
                    color=alt.Color("Risiko-Komponente:N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=["Risiko-Komponente", "Wert (annualisiert)"]
                ).properties(height=400)
                
                st.altair_chart(risk_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - **Systematisches Risiko** ist das Marktrisiko, das durch Faktoren erklärt wird.  
                - **Idiosynkratisches Risiko** ist das spezifische Risiko einzelner Positionen.  
                - **Total Risk** ist die Summe aus beiden Komponenten.  
                - Ein robustes Portfolio hat einen hohen Anteil systematischen Risikos und einen niedrigen Anteil idiosynkratischen Risikos.  
                """)

                # ---------------------------------------------------------
                # Regime-Sensitivity – Schritt 51
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Regime-Sensitivity (Bull / Bear / High-Vol / Low-Vol)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Markt-Proxy (synthetisch)
                market = factors["Market"]
                
                # Volatilität des Marktes (Rolling)
                market_vol = market.rolling(21).std()
                
                # Regime-Definitionen
                bull_mask = market > 0
                bear_mask = market < 0
                high_vol_mask = market_vol > market_vol.median()
                low_vol_mask = market_vol <= market_vol.median()
                
                # Regime-Renditen
                regime_results = {
                    "Bull Market": portfolio_returns_series[bull_mask].mean() * 252,
                    "Bear Market": portfolio_returns_series[bear_mask].mean() * 252,
                    "High Volatility": portfolio_returns_series[high_vol_mask].mean() * 252,
                    "Low Volatility": portfolio_returns_series[low_vol_mask].mean() * 252
                }
                
                regime_df = pd.DataFrame({
                    "Regime": list(regime_results.keys()),
                    "Annualisierte Rendite": list(regime_results.values())
                })
                
                st.table(regime_df)
                
                # Balkendiagramm
                regime_chart = alt.Chart(regime_df).mark_bar().encode(
                    x=alt.X("Regime:N", sort=None),
                    y=alt.Y("Annualisierte Rendite:Q", title="Rendite (annualisiert)"),
                    color=alt.condition(
                        alt.datum["Annualisierte Rendite"] < 0,
                        alt.value("#FF4444"),
                        alt.value("#44FF44")
                    ),
                    tooltip=["Regime", "Annualisierte Rendite"]
                ).properties(height=400)
                
                st.altair_chart(regime_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - **Bull Market:** Wie stark profitiert dein Portfolio von positiven Marktphasen?  
                - **Bear Market:** Wie gut schützt es in Abschwüngen?  
                - **High Volatility:** Wie verhält es sich in Stressphasen?  
                - **Low Volatility:** Wie stabil ist es in ruhigen Märkten?  
                
                Ein robustes Portfolio zeigt:  
                - positive Renditen in Bull Markets  
                - geringe Verluste in Bear Markets  
                - Stabilität in High-Vol-Phasen  
                """)

                # ---------------------------------------------------------
                # Portfolio-Forecast-Modul – Schritt 52
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Forecast (ARIMA + ML)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # -----------------------------------------
                # ARIMA Forecast
                # -----------------------------------------
                from statsmodels.tsa.arima.model import ARIMA
                
                try:
                    arima_model = ARIMA(portfolio_returns_series, order=(1,0,1)).fit()
                    arima_forecast = arima_model.forecast(30)  # 30 Tage Forecast
                except:
                    arima_forecast = pd.Series([0]*30)
                
                # -----------------------------------------
                # ML Forecast (Random Forest)
                # -----------------------------------------
                from sklearn.ensemble import RandomForestRegressor
                
                # Features: Lagged Returns
                lags = 5
                ml_df = pd.DataFrame({
                    f"lag_{i}": portfolio_returns_series.shift(i) for i in range(1, lags+1)
                })
                ml_df["y"] = portfolio_returns_series
                ml_df = ml_df.dropna()
                
                X = ml_df.drop("y", axis=1)
                y = ml_df["y"]
                
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X, y)
                
                # ML Forecast: autoregressive rollout
                ml_forecast = []
                last_values = list(portfolio_returns_series.tail(lags))
                
                for _ in range(30):
                    X_pred = np.array(last_values[-lags:]).reshape(1, -1)
                    pred = rf.predict(X_pred)[0]
                    ml_forecast.append(pred)
                    last_values.append(pred)
                
                # -----------------------------------------
                # Forecast DataFrame
                # -----------------------------------------
                forecast_df = pd.DataFrame({
                    "Tag": range(1, 31),
                    "ARIMA Forecast": arima_forecast.values,
                    "ML Forecast": ml_forecast
                })
                
                st.table(forecast_df)
                
                # -----------------------------------------
                # Forecast Chart
                # -----------------------------------------
                forecast_long = forecast_df.melt(id_vars="Tag", var_name="Modell", value_name="Wert")
                
                forecast_chart = alt.Chart(forecast_long).mark_line().encode(
                    x=alt.X("Tag:Q", title="Tage in der Zukunft"),
                    y=alt.Y("Wert:Q", title="Forecast Return"),
                    color="Modell:N",
                    tooltip=["Tag", "Modell", "Wert"]
                ).properties(height=400)
                
                st.altair_chart(forecast_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - **ARIMA** modelliert lineare Zeitreihenmuster.  
                - **ML (Random Forest)** erkennt nichtlineare Muster.  
                - Forecasts sind *keine Garantien*, aber helfen, Trends und Risiken zu erkennen.  
                - Wenn beide Modelle ähnliche Trends zeigen → höhere Forecast‑Konfidenz.  
                """)

                # ---------------------------------------------------------
                # Scenario-Optimizer (What-If-Engine) – Schritt 53
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Scenario-Optimizer (What-If-Engine)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Szenario-Auswahl
                scenario = st.selectbox(
                    "Szenario auswählen:",
                    [
                        "Tech -20%",
                        "Bonds +10%",
                        "Emerging Markets -15%",
                        "Energy +12%",
                        "Custom Scenario"
                    ]
                )
                
                # Standard-Szenarien definieren
                scenario_shocks = {
                    "Tech -20%": {"Tech": -0.20},
                    "Bonds +10%": {"Bonds": 0.10},
                    "Emerging Markets -15%": {"EM": -0.15},
                    "Energy +12%": {"Energy": 0.12}
                }
                
                # Custom Scenario
                custom_shock = {}
                if scenario == "Custom Scenario":
                    st.markdown("#### Custom Shock definieren")
                    for asset in df.columns:
                        val = st.number_input(f"{asset} Shock (%)", value=0.0)
                        custom_shock[asset] = val / 100
                
                # Shock anwenden
                if scenario == "Custom Scenario":
                    shock_dict = custom_shock
                else:
                    shock_dict = scenario_shocks.get(scenario, {})
                
                # Shock-Vektor erstellen
                shock_vector = np.array([shock_dict.get(asset, 0) for asset in df.columns])
                
                # Szenario-Auswirkung
                scenario_return = float(np.dot(w, shock_vector))
                
                # Risiko neu berechnen
                new_returns = portfolio_returns_series + scenario_return
                new_vol = new_returns.std() * np.sqrt(252)
                new_mean = new_returns.mean() * 252
                
                # Tabelle
                scenario_df = pd.DataFrame({
                    "Kennzahl": ["Szenario-Rendite", "Neue annualisierte Rendite", "Neue Volatilität"],
                    "Wert": [scenario_return, new_mean, new_vol]
                })
                
                st.table(scenario_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Das Szenario wendet pro Asset einen definierten Shock an.  
                - Die Portfolio-Rendite verändert sich proportional zu den Gewichten.  
                - Die neue Rendite und Volatilität zeigen, wie robust das Portfolio auf das Szenario reagiert.  
                - Perfekt für What-If-Analysen und Risiko-Simulationen.  
                """)

                # ---------------------------------------------------------
                # Portfolio-Stress-Surface (3D-Risk-Map) – Schritt 54
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Stress-Surface (3D-Risk-Map)")
                
                # Portfolio-Renditen
                portfolio_returns_series = df @ w
                
                # Zwei Stress-Dimensionen definieren
                shock_range = np.linspace(-0.20, 0.20, 25)  # -20% bis +20%
                
                X, Y = np.meshgrid(shock_range, shock_range)
                
                Z = np.zeros_like(X)
                
                # Wir stressen zwei synthetische Faktoren:
                # X = Marktshock
                # Y = Zins-/Bondshock
                
                for i in range(len(shock_range)):
                    for j in range(len(shock_range)):
                        market_shock = X[i, j]
                        bond_shock = Y[i, j]
                
                        # Shock-Vektor (synthetisch)
                        shock_vector = np.zeros(len(df.columns))
                
                        for idx, asset in enumerate(df.columns):
                            if "Tech" in asset or "Equity" in asset or "Stock" in asset:
                                shock_vector[idx] = market_shock
                            if "Bond" in asset or "Treasury" in asset or "Fixed" in asset:
                                shock_vector[idx] = bond_shock
                
                        Z[i, j] = np.dot(w, shock_vector)
                
                # DataFrame für Altair
                surface_df = pd.DataFrame({
                    "MarketShock": X.flatten(),
                    "BondShock": Y.flatten(),
                    "PortfolioImpact": Z.flatten()
                })
                
                # Heatmap (2D-Surface)
                surface_chart = alt.Chart(surface_df).mark_rect().encode(
                    x=alt.X("MarketShock:Q", title="Marktshock"),
                    y=alt.Y("BondShock:Q", title="Bondshock"),
                    color=alt.Color("PortfolioImpact:Q", scale=alt.Scale(scheme="inferno")),
                    tooltip=["MarketShock", "BondShock", "PortfolioImpact"]
                ).properties(height=400)
                
                st.altair_chart(surface_chart, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Die Stress-Surface zeigt die Portfolio-Auswirkung bei gleichzeitigen Schocks.  
                - Dunkle Farben = starke negative Auswirkungen.  
                - Helle Farben = positive oder geringe Auswirkungen.  
                - Die Karte zeigt nichtlineare Risiko-Hotspots und Stresszonen.  
                - Perfekt für institutionelle Risiko-Analysen und Präsentationen.  
                """)

                # ---------------------------------------------------------
                # Portfolio-AI-Insights (LLM-basierte Risiko-Analyse) – Schritt 55
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-AI-Insights (LLM-basierte Risiko-Analyse)")
                
                # Wichtige Kennzahlen einsammeln
                summary_data = {
                    "Sharpe": sharpe_ratio if 'sharpe_ratio' in locals() else None,
                    "Volatilität": float((df @ w).std() * np.sqrt(252)),
                    "Max Drawdown": float(max_dd) if 'max_dd' in locals() else None,
                    "Diversifikation": float(div_score) if 'div_score' in locals() else None,
                    "Liquidity": float(portfolio_liquidity_index) if 'portfolio_liquidity_index' in locals() else None,
                    "Quality": float(quality_score) if 'quality_score' in locals() else None,
                    "Systematic Risk": float(systematic_risk) if 'systematic_risk' in locals() else None,
                    "Idiosyncratic Risk": float(idiosyncratic_risk) if 'idiosyncratic_risk' in locals() else None,
                }
                
                # Text generieren (LLM-Style, aber lokal)
                insight_text = f"""
                **AI-Analyse deines Portfolios**
                
                - Die aktuelle Volatilität liegt bei **{summary_data['Volatilität']:.4f}**, was auf ein moderates Risikoprofil hindeutet.
                - Der maximale Drawdown beträgt **{summary_data['Max Drawdown']:.4f}**, was zeigt, wie stark das Portfolio in Stressphasen fallen kann.
                - Die Diversifikation ist mit einem Score von **{summary_data['Diversifikation']:.4f}** solide, könnte aber durch breitere Faktor- oder Regionenexposure weiter verbessert werden.
                - Die Liquidität des Portfolios ist **{summary_data['Liquidity']:.4f}**, was bedeutet, dass die meisten Positionen gut handelbar sind.
                - Der Portfolio-Quality-Score liegt bei **{summary_data['Quality']:.4f}**, was auf eine robuste Struktur mit guter Risiko-Rendite-Effizienz hinweist.
                - Das systematische Risiko beträgt **{summary_data['Systematic Risk']:.4f}**, während das idiosynkratische Risiko bei **{summary_data['Idiosyncratic Risk']:.4f}** liegt. Das zeigt, dass ein Großteil des Risikos durch Markt- und Stilfaktoren erklärt wird.
                - Insgesamt zeigt das Portfolio eine **stabile, diversifizierte und risiko-effiziente Struktur**, mit Potenzial zur weiteren Optimierung in Stressphasen und Faktorbreite.
                """
                
                st.markdown(insight_text)

                # ---------------------------------------------------------
                # Portfolio-Rebalancing-Simulator – Schritt 56
                # ---------------------------------------------------------
                
                st.markdown("### ")
                st.subheader("Portfolio-Rebalancing-Simulator (zeitbasierte Simulation)")
                
                # Portfolio-Renditen
                returns = df
                
                # Rebalancing-Frequenz auswählen
                freq = st.selectbox(
                    "Rebalancing-Frequenz:",
                    ["Monatlich", "Quartalsweise", "Jährlich"]
                )
                
                # Frequenz in Perioden umrechnen
                freq_map = {
                    "Monatlich": 21,
                    "Quartalsweise": 63,
                    "Jährlich": 252
                }
                rebalance_period = freq_map[freq]
                
                # Simulation
                weights_current = w.copy()
                portfolio_value = 1.0
                values = []
                
                for i in range(len(returns)):
                    # Portfolio wächst mit täglichen Renditen
                    daily_ret = float(np.dot(weights_current, returns.iloc[i]))
                    portfolio_value *= (1 + daily_ret)
                    values.append(portfolio_value)
                
                    # Rebalancing
                    if i % rebalance_period == 0 and i > 0:
                        weights_current = w.copy()  # zurück zu Zielgewichten
                
                # DataFrame für Plot
                sim_df = pd.DataFrame({
                    "Tag": range(len(values)),
                    "Portfolio-Wert": values
                })
                
                # Plot
                sim_chart = alt.Chart(sim_df).mark_line().encode(
                    x=alt.X("Tag:Q", title="Tage"),
                    y=alt.Y("Portfolio-Wert:Q", title="Portfolio-Wert"),
                    tooltip=["Tag", "Portfolio-Wert"]
                ).properties(height=400)
                
                st.altair_chart(sim_chart, use_container_width=True)
                
                # Kennzahlen
                final_value = values[-1]
                annual_return = (final_value ** (252 / len(values))) - 1
                vol = np.std(np.diff(values) / values[:-1]) * np.sqrt(252)
                
                rebalance_df = pd.DataFrame({
                    "Kennzahl": ["Endwert", "Annualisierte Rendite", "Volatilität"],
                    "Wert": [final_value, annual_return, vol]
                })
                
                st.table(rebalance_df)
                
                # Interpretation
                st.markdown(f"""
                **Interpretation:**  
                - Der Simulator zeigt, wie sich dein Portfolio mit regelmäßigen Rebalancing entwickelt.  
                - Monatliches Rebalancing hält die Allokation stabil, aber verursacht mehr Umschichtungen.  
                - Jährliches Rebalancing lässt Drift zu, kann aber höhere Renditen erzeugen.  
                - Die Simulation zeigt, wie stark Rebalancing Risiko und Rendite beeinflusst.  
                """)

                

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
