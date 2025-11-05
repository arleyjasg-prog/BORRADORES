# 💼 FinSight – Analizador de Rentabilidad y Riesgo Empresarial (Versión multiportafolio)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FinSight", page_icon="💼", layout="wide")

# 💠 Estilos personalizados
st.markdown("""
    <style>
    .main {
        background-color: #F9FAFB;
    }
    h1, h2, h3 {
        color: #002B5B;
    }
    .stButton>button {
        background-color: #0078D7;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 🧭 Encabezado principal
st.markdown("<h1 style='text-align: center;'>💼 FinSight</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Analizador de Rentabilidad y Riesgo Empresarial</h4>", unsafe_allow_html=True)
st.markdown("---")

# 📂 Navegación
opcion = st.sidebar.radio("Selecciona una vista:", ["Análisis individual", "Análisis multiempresa"])

# =====================================================
# 📈 VISTA 1: ANÁLISIS INDIVIDUAL
# =====================================================
if opcion == "Análisis individual":
    st.sidebar.header("⚙ Configuración de análisis individual")
    ticker = st.sidebar.text_input("📊 Ticker de la empresa:", "AAPL")
    start_date = st.sidebar.date_input("📅 Fecha inicial:", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("📅 Fecha final:", pd.to_datetime("2024-12-31"))

    if st.sidebar.button("Analizar empresa"):
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error("❌ No se encontraron datos para el ticker especificado.")
        else:
            st.success(f"✅ Datos descargados correctamente para *{ticker}*")

            # 🔧 Si viene con MultiIndex (a veces ocurre), tomar solo el nivel del ticker
            if isinstance(data.columns, pd.MultiIndex):
                data = data[ticker]

            # Cálculos
            price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
            data["Daily Return"] = data[price_col].pct_change()
            avg_return = data["Daily Return"].mean()
            std_dev = data["Daily Return"].std()
            sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0

            # 🎯 Mostrar resultados
            col1, col2, col3 = st.columns(3)
            col1.metric("Rentabilidad promedio", f"{avg_return*100:.2f}%")
            col2.metric("Riesgo (volatilidad)", f"{std_dev*100:.2f}%")
            col3.metric("Índice de Sharpe", f"{sharpe_ratio:.2f}")

            st.markdown("---")

            # 📉 Gráfico de precios
            st.subheader("📈 Evolución del precio ajustado")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data[price_col], color='#0078D7', linewidth=2)
            ax.set_title(f"Precio histórico de {ticker}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio ($)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            # 📊 Distribución de retornos
            st.subheader("📊 Distribución de los rendimientos diarios")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(data["Daily Return"].dropna(), bins=30, kde=True, ax=ax2, color='#009688')
            st.pyplot(fig2)

            # 🧾 Datos recientes
            st.subheader("📘 Últimos datos descargados")
            st.dataframe(data.tail(10), use_container_width=True)

# =====================================================
# 🏦 VISTA 2: ANÁLISIS MULTIEMPRESA (ILIMITADO)
# =====================================================
elif opcion == "Análisis multiempresa":
    st.sidebar.header("📊 Configuración comparativa múltiple")
    tickers_input = st.sidebar.text_input("Empresas (separadas por coma):", "AAPL, MSFT, TSLA, NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.sidebar.date_input("📅 Fecha inicial:", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("📅 Fecha final:", pd.to_datetime("2024-12-31"))

    if st.sidebar.button("Comparar empresas"):
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by="ticker")

        if data.empty:
            st.error("❌ No se encontraron datos para los tickers especificados.")
        else:
            st.success(f"✅ Datos descargados correctamente para {', '.join(tickers)}")

            resultados = []
            fig, ax = plt.subplots(figsize=(10, 5))

            # 📊 Procesamiento individual de cada ticker
            for ticker in tickers:
                try:
                    df = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                except KeyError:
                    st.warning(f"⚠ No se encontraron datos para {ticker}.")
                    continue

                price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                if price_col not in df.columns:
                    st.warning(f"⚠ {ticker}: No se encontró columna de precio válida.")
                    continue

                df["Daily Return"] = df[price_col].pct_change()
                avg = df["Daily Return"].mean()
                std = df["Daily Return"].std()
                sharpe = avg / std if std != 0 else 0
                resultados.append({"Ticker": ticker, "Rentabilidad": avg*100, "Volatilidad": std*100, "Sharpe": sharpe})

                ax.plot(df[price_col], linewidth=2, label=ticker)

            # 🧮 Mostrar tabla de resultados
            if resultados:
                resultados_df = pd.DataFrame(resultados).set_index("Ticker")
                st.subheader("📊 Métricas comparativas")
                st.dataframe(resultados_df.style.format({
                    "Rentabilidad": "{:.2f}%",
                    "Volatilidad": "{:.2f}%",
                    "Sharpe": "{:.2f}"
                }))

            # 📈 Gráfico comparativo de precios
            ax.set_title("Evolución de precios ajustados")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio ($)")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            # 📊 Matriz de correlación
            st.subheader("📈 Correlación entre rendimientos")
            returns = pd.DataFrame()

            for ticker in tickers:
                try:
                    df = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                    col = "Adj Close" if "Adj Close" in df.columns else "Close"
                    returns[ticker] = df[col].pct_change()
                except KeyError:
                    continue

            if not returns.empty:
                corr = returns.corr()
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                sns.heatmap(corr, annot=True, cmap="Blues", ax=ax2)
                ax2.set_title("Matriz de correlación")
                st.pyplot(fig2)
            else:
                st.warning("⚠ No se pudieron calcular las correlaciones por falta de datos válidos.")

# 🪪 Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 FinSight | Desarrollado por Angie</p>", unsafe_allow_html=True)
