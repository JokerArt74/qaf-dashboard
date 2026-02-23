import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

st.title("QAF – Portfolio Optimizer Demo")

st.write("""
Willkommen im QAF Dashboard.

Hier wirst du später:
- Daten hochladen
- Optimierungen starten
- Ergebnisse sehen
- Reports generieren

Wir beginnen ganz oben: mit der Oberfläche.
""")

st.subheader("Upload Bereich (kommt später)")
uploaded_file = st.file_uploader("Portfolio-Datei hochladen (CSV)", type=["csv"])

if uploaded_file:
    st.success("Datei erfolgreich hochgeladen!")
    st.write("Vorschau der Daten:")
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
st.info("Hier wird später der Upload von Portfoliodaten erscheinen.")

st.subheader("Optimierungsbereich (kommt später)")
st.info("Hier wird später der Optimizer angezeigt.")
st.subheader("Optimierungsbereich")

if uploaded_file:
    st.write("Parameter für die Optimierung:")
        st.subheader("Aktuelles Portfolio (optional)")

    st.write("Wenn du hier dein aktuelles Portfolio eingibst, berechnen wir die notwendigen Trades.")

    current_portfolio_input = st.text_area(
        "Aktuelle Gewichte eingeben (Format: Ticker: Gewicht, z. B. AAPL: 0.2)",
        value="",
        height=100
    )

    target_return = st.number_input(
        "Zielrendite (z. B. 0.08 für 8%)",
        value=0.08,
        step=0.01
    )

    long_only = st.checkbox("Nur Long-Positionen erlauben", value=True)
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
        if run_opt:
    st.success("Optimierung ausgeführt (Demo-Modus)")

    # Fake-Ergebnis – später ersetzen wir das durch echte Optimierung
    fake_weights = {
        "AAPL": 0.25,
        "MSFT": 0.25,
        "GOOG": 0.20,
        "AMZN": 0.20,
        "TSLA": 0.10
    }
st.subheader("Optimierungsergebnis")

# Datei einlesen
df = pd.read_csv(uploaded_file).dropna()

# Optimierung ausführen
weights = mean_variance_optimizer(
    df,
    long_only=long_only,
    max_weight=max_weight,
    min_weight=min_weight
)

if len(weights) == 0:
    st.error("Keine gültigen Assets nach Bereinigung. Bitte Daten prüfen.")
else:
    # Jahreskennzahlen
    mean_returns = df.mean() * 252
    cov = df.cov() * 252

    # Portfolio-Kennzahlen
    w = weights.values
    port_return = float(mean_returns @ w)
    port_vol = float(np.sqrt(w @ cov.values @ w))

    st.write("Berechnete Gewichte:")
    st.table(weights)

    st.write("Geschätzte erwartete Jahresrendite:", round(port_return, 4))
    st.write("Geschätzte erwartete Volatilität:", round(port_vol, 4))

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
        st.subheader("Rebalancing – Kauf/Verkauf-Empfehlungen")

    if current_portfolio_input.strip():
        try:
            # Eingabe parsen
            current_dict = {}
            for line in current_portfolio_input.split(","):
                if ":" in line:
                    ticker, weight = line.split(":")
                    current_dict[ticker.strip()] = float(weight.strip())

            current_series = pd.Series(current_dict)

            # Nur Assets berücksichtigen, die im optimierten Portfolio vorkommen
            aligned_current = current_series.reindex(weights.index).fillna(0)

            # Trades berechnen
            trades = weights - aligned_current

            st.write("Positive Werte = Kaufen, Negative Werte = Verkaufen")
            st.table(trades.rename("Trade"))

        except Exception as e:
            st.error(f"Fehler beim Einlesen des aktuellen Portfolios: {e}")
    else:
        st.info("Kein aktuelles Portfolio eingegeben – Rebalancing wird übersprungen.")

    st.subheader("Kurzfassung für Entscheider")
    st.write(f"- Zielrendite-Einstellung: {target_return}")
    st.write(f"- Long-Only: {long_only}")
    st.write(f"- Optimiertes Portfolio: Rendite {round(port_return,4)}, Volatilität {round(port_vol,4)}")

    st.info("Im nächsten Schritt ersetzen wir dieses Fake-Ergebnis durch echte Optimierung.")
else:
    st.info("Bitte zuerst eine Datei hochladen, um die Optimierung zu aktivieren.")

st.subheader("Ergebnisbereich (kommt später)")
st.info("Hier werden später die Ergebnisse angezeigt.")
