import streamlit as st

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

    st.subheader("Optimierungsergebnis (Demo)")
    st.write("Dies ist ein Platzhalter. Später kommen hier echte Berechnungen hin.")

    st.table(fake_weights)

    st.info("Im nächsten Schritt ersetzen wir dieses Fake-Ergebnis durch echte Optimierung.")
else:
    st.info("Bitte zuerst eine Datei hochladen, um die Optimierung zu aktivieren.")

st.subheader("Ergebnisbereich (kommt später)")
st.info("Hier werden später die Ergebnisse angezeigt.")
