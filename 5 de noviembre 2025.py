import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import descargar_datos, calcular_retorno
from utils.portfolio import estadisticas_portafolio, simulacion_montecarlo
from utils.plots import plot_precios, plot_correlacion, plot_frontera

st.set_page_config(page_title="FinanSmart - Portafolio", page_icon="📈", layout="wide")
st.title("📊 FinanSmart - Análisis de Portafolio de Inversión")
st.sidebar.header("⚙️ Configuración")

tickers = st.sidebar.text_input("Tickers (coma separados)", "AAPL,MSFT,GOOGL,AMZN,META").split(",")
start = st.sidebar.date_input("Inicio", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("Fin", pd.to_datetime("2023-12-31"))
num_sim = st.sidebar.slider("Simulaciones Monte Carlo", 1000, 50000, 10000, 1000)

if st.sidebar.button("🚀 Ejecutar análisis"):
    data = descargar_datos(tickers, start, end)
    returns = calcular_retorno(data)

    st.header("📈 Datos Históricos")
    st.pyplot(plot_precios(data))
    st.header("📉 Correlación de Activos")
    st.pyplot(plot_correlacion(returns))

    st.header("📊 Simulación Monte Carlo")
    resultados, pesos = simulacion_montecarlo(returns, num_sim)
    st.pyplot(plot_frontera(resultados))

    max_idx = resultados["Sharpe"].idxmax()
    st.subheader("🏆 Portafolio Óptimo")
    st.metric("Sharpe Ratio", f"{resultados.iloc[max_idx]['Sharpe']:.2f}")
    st.metric("Retorno", f"{resultados.iloc[max_idx]['Retorno']:.2%}")
    st.metric("Riesgo", f"{resultados.iloc[max_idx]['Riesgo']:.2%}")
