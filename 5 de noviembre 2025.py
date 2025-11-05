# -*- coding: utf-8 -*-
"""
FinanSmart - Análisis de Portafolio de Inversión
Aplicación Streamlit unificada
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 🔹 Funciones auxiliares
# =====================================

def descargar_datos(tickers, start, end):
    """Descarga precios ajustados desde Yahoo Finance"""
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    data.dropna(inplace=True)
    return data

def calcular_retorno(data, freq='D'):
    """Calcula retornos porcentuales según frecuencia"""
    returns = data.pct_change().dropna()
    if freq == 'M':
        returns = data.resample('M').ffill().pct_change().dropna()
    elif freq == 'Y':
        returns = data.resample('Y').ffill().pct_change().dropna()
    return returns

def estadisticas_portafolio(returns, weights):
    """Calcula retorno y riesgo anualizados de un portafolio"""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_volatility if port_volatility != 0 else 0
    return port_return, port_volatility, sharpe

def simulacion_montecarlo(returns, num_portfolios=10000):
    """Simula miles de portafolios para generar la frontera eficiente"""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)
    
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol, sharpe = estadisticas_portafolio(returns, weights)
        results[:, i] = [vol, ret, sharpe]
        weights_record.append(weights)
        
    results_df = pd.DataFrame(results.T, columns=["Riesgo", "Retorno", "Sharpe"])
    return results_df, weights_record

def plot_precios(data):
    fig, ax = plt.subplots(figsize=(12,6))
    data.plot(ax=ax)
    ax.set_title("Evolución de precios ajustados")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio ($)")
    ax.grid(alpha=0.3)
    return fig

def plot_correlacion(returns):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Matriz de correlación")
    return fig

def plot_frontera(resultados):
    fig, ax = plt.subplots(figsize=(10,7))
    scatter = ax.scatter(
        resultados["Riesgo"], resultados["Retorno"],
        c=resultados["Sharpe"], cmap="viridis", alpha=0.6, s=10
    )
    plt.colorbar(scatter, label="Índice de Sharpe", ax=ax)
    ax.set_xlabel("Riesgo (Volatilidad)")
    ax.set_ylabel("Retorno esperado")
    ax.set_title("Frontera Eficiente")
    return fig

# =====================================
# 🚀 Interfaz Streamlit
# =====================================

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

else:
    st.info("👈 Configura los parámetros en el panel lateral y presiona **Ejecutar análisis**")

st.markdown("---")
st.markdown("Desarrollado con ❤️ usando Streamlit | Datos: Yahoo Finance")
