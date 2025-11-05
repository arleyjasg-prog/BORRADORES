# 💼 FinSight – Analizador de Rentabilidad y Riesgo Empresarial (versión múltiple y robusta)
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
opcion = st.sidebar.radio("Selecciona una vista:", ["Análisis individual", "Análisis comparativo"])

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

            # 🔍 Si viene con MultiIndex (varios niveles), tomar solo el del ticker
            if isinstance(data.columns, pd.MultiIndex):
                data = data[ticker]

            # ✅ Asegurarse de tener una sola serie
            price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
            price_series = data[price_col].squeeze()

            # Calcular retornos diarios
            data["Daily Return"] = price_series.pct_change()

            # Métricas
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
            ax.plot(price_series, color='#0078D7', linewidth=2)
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
# 🏦 VISTA 2: ANÁLISIS COMPARATIVO MÚLTIPLE
# =====================================================
elif opcion == "Análisis comparativo":
    st.sidebar.header("📊 Configuración comparativa")
    tickers_input = st.sidebar.text_area("Ingresa los tickers separados por comas:", "AAPL, MSFT, GOOGL, AMZN")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.sidebar.date_input("📅 Fecha inicial:", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("📅 Fecha final:", pd.to_datetime("2024-12-31"))

    if st.sidebar.button("Comparar empresas"):
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error("❌ Verifica los tickers, no se encontraron datos.")
        else:
            st.success(f"✅ Comparando: {', '.join(tickers)}")

            # Si hay MultiIndex, quedarse con el nivel de precios ajustados o cerrados
            price_col = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"

            prices = data[price_col].copy()

            # Calcular retornos diarios
            daily_returns = prices.pct_change()

            # 📊 Estadísticas principales
            mean_returns = daily_returns.mean()
            volatilities = daily_returns.std()

            summary = pd.DataFrame({
                "Rentabilidad promedio (%)": mean_returns * 100,
                "Volatilidad (%)": volatilities * 100,
                "Sharpe Ratio": (mean_returns / volatilities).replace([np.inf, -np.inf], np.nan)
            }).dropna()

            st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

            # 📈 Gráfico comparativo de precios
            st.subheader("📉 Comparación de precios históricos")
            fig, ax = plt.subplots(figsize=(10, 5))
            for ticker in prices.columns:
                ax.plot(prices[ticker], label=ticker, linewidth=2)
            ax.legend()
            ax.set_title("Evolución de precios ajustados")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio ($)")
            st.pyplot(fig)

            # 📊 Matriz de correlación
            st.subheader("📊 Correlación entre rendimientos")
            corr = daily_returns.corr()
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax2)
            st.pyplot(fig2)

            # 🧠 Conclusión automática
            st.markdown("### 📈 Conclusión del análisis")
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    c = corr.loc[t1, t2]
                    if c > 0.7:
                        st.info(f"🔗 {t1} y {t2} están **fuertemente correlacionadas** — se mueven juntas.")
                    elif c > 0.3:
                        st.warning(f"⚖️ {t1} y {t2} tienen **correlación moderada**.")
                    else:
                        st.success(f"🌿 {t1} y {t2} están **poco correlacionadas** — buena opción para diversificar.")

# 🪪 Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 FinSight | Desarrollado por Angie</p>", unsafe_allow_html=True)
