import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def mean_variance_optimizer(returns_df, long_only=True):
    """
    Einfache Mean-Variance-Optimierung (ohne cvxpy):
    - Minimiert Risiko = w^T Σ w
    - Unter Nebenbedingungen:
        * Summe der Gewichte = 1
        * Optional: w >= 0 (Long Only)
    """

    # Kovarianzmatrix und Anzahl Assets
    cov = returns_df.cov().values
    n = cov.shape[0]

    # Inverse der Kovarianzmatrix
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Falls Matrix nicht invertierbar ist → leichte Regularisierung
        cov += np.eye(n) * 1e-6
        inv_cov = np.linalg.inv(cov)

    # Gleichgewichtung nach Risiko (Minimum Variance Portfolio)
    ones = np.ones(n)
    weights = inv_cov @ ones
    weights = weights / weights.sum()

    # Long-only erzwingen
    if long_only:
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()

    return pd.Series(weights, index=returns_df.columns).round(4)

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

    target_return = st.number_input(
        "Zielrendite (z. B. 0.08 für 8%)",
        value=0.08,
        step=0.01
    )

    long_only = st.checkbox("Nur Long-Positionen erlauben", value=True)

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
df = pd.read_csv(uploaded_file)

# Optimierung ausführen
weights = mean_variance_optimizer(df, long_only)

# Ergebnis anzeigen
st.write("Berechnete Gewichte:")
st.table(weights)
# Chart vorbereiten
chart_data = pd.DataFrame({
    "Asset": weights.index,
    "Weight": weights.values
})

st.subheader("Visualisierung der Gewichte")

bar_chart = alt.Chart(chart_data).mark_bar().encode(
    x="Asset",
    y="Weight",
    tooltip=["Asset", "Weight"]
)

st.altair_chart(bar_chart, use_container_width=True)

# Kleine Zusammenfassung
st.subheader("Zusammenfassung")
st.write(f"Zielrendite: {target_return}")
st.write(f"Long-Only: {long_only}")
st.write("Optimierung erfolgreich durchgeführt.")

# Erwartete Rendite (sehr einfache Schätzung)
expected_return = float((df.mean() * 252).mean())
st.write("Geschätzte erwartete Jahresrendite:", round(expected_return, 4))

    st.info("Im nächsten Schritt ersetzen wir dieses Fake-Ergebnis durch echte Optimierung.")
else:
    st.info("Bitte zuerst eine Datei hochladen, um die Optimierung zu aktivieren.")

st.subheader("Ergebnisbereich (kommt später)")
st.info("Hier werden später die Ergebnisse angezeigt.")
