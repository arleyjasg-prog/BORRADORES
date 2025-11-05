# -*- coding: utf-8 -*-
"""
FinanSmart - Análisis Completo de Portafolio de Inversión
Aplicación Streamlit para análisis financiero con todos los parámetros requeridos
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="FinanSmart - Análisis de Portafolio",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 FinanSmart - Análisis Completo de Portafolio de Inversión")
st.markdown("---")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Análisis")

# 1. Selección de empresas (tickers)
st.sidebar.subheader("1️⃣ Selección de Empresas")
default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
tickers_input = st.sidebar.text_area(
    "Tickers (uno por línea o separados por comas)",
    value="\n".join(default_tickers),
    height=100
)
# Procesar tickers (aceptar comas o saltos de línea)
tickers = [t.strip().upper() for t in tickers_input.replace(",", "\n").split("\n") if t.strip()]
st.sidebar.info(f"📈 {len(tickers)} empresas seleccionadas")

# 2. Pesos del portafolio
st.sidebar.subheader("2️⃣ Pesos del Portafolio")
peso_option = st.sidebar.radio(
    "Método de asignación:",
    ["Pesos iguales", "Pesos personalizados", "Optimización automática"]
)

weights_custom = None
if peso_option == "Pesos personalizados":
    st.sidebar.write("Ingresa los pesos (deben sumar 1.0):")
    weights_custom = []
    for ticker in tickers:
        peso = st.sidebar.number_input(
            f"{ticker}",
            min_value=0.0,
            max_value=1.0,
            value=1.0/len(tickers),
            step=0.01,
            key=f"peso_{ticker}"
        )
        weights_custom.append(peso)
    suma_pesos = sum(weights_custom)
    if abs(suma_pesos - 1.0) > 0.01:
        st.sidebar.warning(f"⚠️ Los pesos suman {suma_pesos:.2f}. Deben sumar 1.0")
    else:
        st.sidebar.success(f"✅ Pesos válidos (suma = {suma_pesos:.2f})")

# 3. Inversión inicial
st.sidebar.subheader("3️⃣ Inversión Inicial")
inversion_inicial = st.sidebar.number_input(
    "Monto de inversión ($)",
    min_value=100,
    max_value=10000000,
    value=10000,
    step=1000,
    format="%d"
)

# 4. Período de análisis
st.sidebar.subheader("4️⃣ Período de Análisis")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("Fecha fin", value=pd.to_datetime("2023-12-31"))

# 5. Frecuencia temporal
st.sidebar.subheader("5️⃣ Frecuencia Temporal")
frecuencia = st.sidebar.selectbox(
    "Frecuencia de datos:",
    ["Diaria", "Semanal", "Mensual"]
)
freq_map = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
intervalo = freq_map[frecuencia]

# Número de simulaciones para Monte Carlo
num_portfolios = st.sidebar.slider(
    "Simulaciones Monte Carlo",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)

# Botón para ejecutar análisis
if st.sidebar.button("🚀 Ejecutar Análisis Completo", type="primary"):
    
    with st.spinner("Descargando y procesando datos..."):
        try:
            # Descarga de datos
            data = yf.download(tickers, start=start_date, end=end_date, interval=intervalo, progress=False)['Close']
            
            if data.empty:
                st.error("No se pudieron descargar datos. Verifica los tickers y las fechas.")
                st.stop()
            
            # Si solo hay un ticker, convertir a DataFrame
            if len(tickers) == 1:
                data = pd.DataFrame(data, columns=tickers)
            
            st.success("✅ Datos descargados exitosamente")
            
            # ============================================
            # SECCIÓN 1: DATOS HISTÓRICOS
            # ============================================
            st.header("1️⃣ Datos Históricos de Precios")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(data.tail(10), use_container_width=True)
            with col2:
                st.metric("Período", f"{len(data)} períodos")
                st.metric("Empresas", len(tickers))
            
            # Gráfico de evolución de precios
            st.subheader("📈 Serie de Tiempo de Precios")
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            for ticker in tickers:
                ax1.plot(data.index, data[ticker], label=ticker, linewidth=2)
            ax1.set_title('Evolución Histórica de Precios Ajustados', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Fecha', fontsize=12)
            ax1.set_ylabel('Precio ($)', fontsize=12)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()
            
            # ============================================
            # SECCIÓN 2: CÁLCULO DE RETORNOS
            # ============================================
            st.header("2️⃣ Retornos por Período")
            
            # Calcular retornos
            returns = data.pct_change().dropna()
            
            # Retornos acumulados
            retornos_acumulados = (1 + returns).cumprod() - 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Estadísticas de Retornos")
                stats_df = returns.describe()
                st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
            
            with col2:
                st.subheader(f"📉 Retornos {frecuencia}s")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for ticker in tickers:
                    ax2.plot(returns.index, returns[ticker], label=ticker, alpha=0.7, linewidth=1)
                ax2.set_title(f'Retornos {frecuencia}s', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Fecha')
                ax2.set_ylabel('Retorno')
                ax2.legend(loc='best', fontsize=9)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            # Retornos acumulados
            st.subheader("📈 Retornos Acumulados")
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            for ticker in tickers:
                ax3.plot(retornos_acumulados.index, retornos_acumulados[ticker] * 100, 
                        label=ticker, linewidth=2)
            ax3.set_title('Evolución de Retornos Acumulados', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Fecha', fontsize=12)
            ax3.set_ylabel('Retorno Acumulado (%)', fontsize=12)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
            
            # ============================================
            # SECCIÓN 3: MÉTRICAS ANUALIZADAS
            # ============================================
            st.header("3️⃣ Volatilidad Histórica y Anualizada")
            
            # Factor de anualización según frecuencia
            if frecuencia == "Diaria":
                factor_anual = 252
            elif frecuencia == "Semanal":
                factor_anual = 52
            else:  # Mensual
                factor_anual = 12
            
            # Cálculos anualizados
            mean_returns_annual = returns.mean() * factor_anual
            volatilidad_annual = returns.std() * np.sqrt(factor_anual)
            
            st.subheader("📊 Métricas Anualizadas por Activo")
            metrics_df = pd.DataFrame({
                'Retorno Anual Esperado': mean_returns_annual,
                'Volatilidad Anual (σ)': volatilidad_annual,
                'Coef. Variación': volatilidad_annual / mean_returns_annual.replace(0, np.nan)
            })
            st.dataframe(metrics_df.style.format({
                'Retorno Anual Esperado': '{:.2%}',
                'Volatilidad Anual (σ)': '{:.2%}',
                'Coef. Variación': '{:.2f}'
            }), use_container_width=True)
            
            # ============================================
            # SECCIÓN 4: CORRELACIÓN
            # ============================================
            st.header("4️⃣ Matriz de Correlación")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                corr_matrix = returns.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, 
                           fmt='.2f', ax=ax4, square=True, linewidths=1,
                           cbar_kws={'label': 'Correlación'})
                ax4.set_title('Matriz de Correlación de Retornos', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()
            
            with col2:
                st.subheader("📋 Interpretación")
                st.write("""
                **Correlación:**
                - 🟩 Verde: Correlación positiva
                - 🟥 Rojo: Correlación negativa
                - ⬜ Blanco: No correlación
                
                **Diversificación:**
                - Valores cercanos a 0 o negativos indican mejor diversificación
                """)
                
                # Estadísticas de correlación
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                st.metric("Correlación Promedio", f"{corr_values.mean():.2f}")
                st.metric("Correlación Máxima", f"{corr_values.max():.2f}")
                st.metric("Correlación Mínima", f"{corr_values.min():.2f}")
            
            # ============================================
            # SECCIÓN 5: ANÁLISIS DE PORTAFOLIO
            # ============================================
            st.header("5️⃣ Análisis del Portafolio")
            
            # Determinar pesos según la opción seleccionada
            if peso_option == "Pesos iguales":
                weights = np.array([1/len(tickers)] * len(tickers))
            elif peso_option == "Pesos personalizados" and weights_custom:
                weights = np.array(weights_custom)
            else:
                # Se calculará en la optimización
                weights = np.array([1/len(tickers)] * len(tickers))
            
            # Cálculos del portafolio
            portfolio_return = np.dot(weights, mean_returns_annual)
            cov_matrix_annual = returns.cov() * factor_anual
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            
            # Valor final del portafolio
            retorno_total_periodo = np.dot(weights, retornos_acumulados.iloc[-1])
            valor_final = inversion_inicial * (1 + retorno_total_periodo)
            ganancia_perdida = valor_final - inversion_inicial
            
            st.subheader(f"💼 Portafolio Configurado ({peso_option})")
            
            # Mostrar pesos
            col1, col2 = st.columns([1, 1])
            with col1:
                pesos_df = pd.DataFrame({
                    'Empresa': tickers,
                    'Peso': weights,
                    'Inversión ($)': weights * inversion_inicial
                })
                st.dataframe(pesos_df.style.format({
                    'Peso': '{:.2%}',
                    'Inversión ($)': '${:,.2f}'
                }), use_container_width=True)
            
            with col2:
                # Gráfico de torta
                fig5, ax5 = plt.subplots(figsize=(8, 8))
                colors = plt.cm.Set3(range(len(tickers)))
                ax5.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=colors)
                ax5.set_title('Distribución de Pesos del Portafolio', fontsize=12, fontweight='bold')
                st.pyplot(fig5)
                plt.close()
            
            # Métricas principales
            st.subheader("📊 Métricas del Portafolio")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Retorno Anual Esperado", f"{portfolio_return:.2%}")
            with col2:
                st.metric("Volatilidad Anual", f"{portfolio_volatility:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col4:
                st.metric("Inversión Inicial", f"${inversion_inicial:,.2f}")
            
            # ============================================
            # SECCIÓN 6: EVOLUCIÓN DEL VALOR MONETARIO
            # ============================================
            st.header("6️⃣ Evolución del Valor Monetario del Portafolio")
            
            # Calcular valor del portafolio en el tiempo
            portfolio_returns = (returns * weights).sum(axis=1)
            portfolio_value = inversion_inicial * (1 + portfolio_returns).cumprod()
            
            fig6, ax6 = plt.subplots(figsize=(14, 6))
            ax6.plot(portfolio_value.index, portfolio_value, linewidth=2.5, color='darkblue', label='Valor del Portafolio')
            ax6.axhline(y=inversion_inicial, color='red', linestyle='--', linewidth=1.5, label='Inversión Inicial')
            ax6.fill_between(portfolio_value.index, inversion_inicial, portfolio_value, 
                            where=(portfolio_value >= inversion_inicial), alpha=0.3, color='green', label='Ganancia')
            ax6.fill_between(portfolio_value.index, inversion_inicial, portfolio_value, 
                            where=(portfolio_value < inversion_inicial), alpha=0.3, color='red', label='Pérdida')
            ax6.set_title('Evolución del Valor del Portafolio', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Fecha', fontsize=12)
            ax6.set_ylabel('Valor ($)', fontsize=12)
            ax6.legend(loc='best')
            ax6.grid(True, alpha=0.3)
            ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Valor Final", f"${valor_final:,.2f}", 
                         delta=f"${ganancia_perdida:,.2f}")
            with col2:
                st.metric("Retorno Total", f"{retorno_total_periodo:.2%}")
            with col3:
                max_value = portfolio_value.max()
                st.metric("Valor Máximo Alcanzado", f"${max_value:,.2f}")
            
            # ============================================
            # SECCIÓN 7: DIAGRAMA RIESGO-RETORNO
            # ============================================
            st.header("7️⃣ Diagrama Riesgo-Retorno (Frontera Eficiente)")
            
            with st.spinner(f"Simulando {num_portfolios:,} portafolios..."):
                results = np.zeros((4, num_portfolios))
                weights_record = []
                
                for i in range(num_portfolios):
                    w = np.random.random(len(tickers))
                    w /= np.sum(w)
                    weights_record.append(w)
                    
                    ret = np.dot(w, mean_returns_annual)
                    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix_annual, w)))
                    sharpe = ret / vol if vol != 0 else 0
                    
                    results[0, i] = vol
                    results[1, i] = ret
                    results[2, i] = sharpe
                    results[3, i] = i
            
            # Encontrar portafolios importantes
            max_sharpe_idx = np.argmax(results[2])
            min_vol_idx = np.argmin(results[0])
            
            # Gráfico de frontera eficiente
            fig7, ax7 = plt.subplots(figsize=(14, 8))
            scatter = ax7.scatter(
                results[0, :] * 100,
                results[1, :] * 100,
                c=results[2, :],
                cmap='viridis',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
            
            # Marcar portafolios especiales
            ax7.scatter(
                results[0, max_sharpe_idx] * 100,
                results[1, max_sharpe_idx] * 100,
                c='red',
                s=500,
                marker='*',
                edgecolors='black',
                linewidths=2,
                label='Máximo Sharpe',
                zorder=5
            )
            ax7.scatter(
                results[0, min_vol_idx] * 100,
                results[1, min_vol_idx] * 100,
                c='blue',
                s=300,
                marker='D',
                edgecolors='black',
                linewidths=2,
                label='Mínima Volatilidad',
                zorder=5
            )
            ax7.scatter(
                portfolio_volatility * 100,
                portfolio_return * 100,
                c='orange',
                s=300,
                marker='s',
                edgecolors='black',
                linewidths=2,
                label='Portafolio Actual',
                zorder=5
            )
            
            ax7.set_xlabel('Riesgo - Volatilidad (%)', fontsize=12)
            ax7.set_ylabel('Retorno Esperado (%)', fontsize=12)
            ax7.set_title('Frontera Eficiente - Simulación Monte Carlo', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Sharpe Ratio', ax=ax7)
            ax7.legend(loc='best', fontsize=10)
            ax7.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close()
            
            # ============================================
            # SECCIÓN 8: PORTAFOLIO ÓPTIMO
            # ============================================
            st.header("8️⃣ Portafolio Óptimo (Máximo Sharpe Ratio)")
            
            optimal_weights = weights_record[max_sharpe_idx]
            mejor_volatilidad = results[0, max_sharpe_idx]
            mejor_retorno = results[1, max_sharpe_idx]
            mejor_sharpe = results[2, max_sharpe_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Sharpe Ratio", f"{mejor_sharpe:.3f}")
            with col2:
                st.metric("📈 Retorno Esperado", f"{mejor_retorno:.2%}")
            with col3:
                st.metric("📊 Volatilidad", f"{mejor_volatilidad:.2%}")
            
            st.subheader("Composición del Portafolio Óptimo")
            optimal_df = pd.DataFrame({
                'Empresa': tickers,
                'Peso Óptimo': optimal_weights,
                'Inversión Sugerida ($)': optimal_weights * inversion_inicial
            })
            st.dataframe(optimal_df.style.format({
                'Peso Óptimo': '{:.2%}',
                'Inversión Sugerida ($)': '${:,.2f}'
            }), use_container_width=True)
            
            # Comparación con portafolio actual
            st.subheader("📊 Comparación: Portafolio Actual vs Óptimo")
            comparison_df = pd.DataFrame({
                'Métrica': ['Retorno Anual', 'Volatilidad', 'Sharpe Ratio'],
                'Portafolio Actual': [
                    f"{portfolio_return:.2%}",
                    f"{portfolio_volatility:.2%}",
                    f"{sharpe_ratio:.2f}"
                ],
                'Portafolio Óptimo': [
                    f"{mejor_retorno:.2%}",
                    f"{mejor_volatilidad:.2%}",
                    f"{mejor_sharpe:.2f}"
                ],
                'Diferencia': [
                    f"{(mejor_retorno - portfolio_return):.2%}",
                    f"{(mejor_volatilidad - portfolio_volatility):.2%}",
                    f"{(mejor_sharpe - sharpe_ratio):.2f}"
                ]
            })
            st.dataframe(comparison_df, use_container_width=True)
            
            # ============================================
            # SECCIÓN 9: HISTOGRAMAS Y DISTRIBUCIONES
            # ============================================
            st.header("9️⃣ Distribución de Retornos (Histogramas)")
            
            # Calcular número de filas necesarias
            n_cols = 2
            n_rows = (len(tickers) + n_cols - 1) // n_cols
            
            fig8, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for idx, ticker in enumerate(tickers):
                axes[idx].hist(returns[ticker] * 100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                axes[idx].set_title(f'{ticker}', fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Retorno (%)', fontsize=9)
                axes[idx].set_ylabel('Frecuencia', fontsize=9)
                axes[idx].axvline(returns[ticker].mean() * 100, color='red', linestyle='--', linewidth=2, label='Media')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend(fontsize=8)
            
            # Ocultar ejes sobrantes
            for idx in range(len(tickers), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig8)
            plt.close()
            
            # Histograma del portafolio
            st.subheader("Distribución de Retornos del Portafolio")
            fig9, ax9 = plt.subplots(figsize=(12, 6))
            ax9.hist(portfolio_returns * 100, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
            ax9.set_title('Distribución de Retornos del Portafolio', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Retorno (%)', fontsize=12)
            ax9.set_ylabel('Frecuencia', fontsize=12)
            ax9.axvline(portfolio_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Media = {portfolio_returns.mean()*100:.2f}%')
            ax9.grid(True, alpha=0.3)
            ax9.legend(fontsize=11)
            plt.tight_layout()
            st.pyplot(fig9)
            plt.close()
            
            # ============================================
            # SECCIÓN 10: BENCHMARK (OPCIONAL)
            # ============================================
            st.header("🔟 Benchmark - Comparación con el Mercado")
            
            try:
                # Descargar S&P 500 como benchmark
                benchmark_data = yf.download("^GSPC", start=start_date, end=end_date, 
                                           interval=intervalo, progress=False)['Close']
                benchmark_returns = benchmark_data.pct_change().dropna()
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                portfolio_cumulative = (1 + portfolio_returns).cumprod()
                
                fig10, ax10 = plt.subplots(figsize=(14, 6))
                ax10.plot(portfolio_cumulative.index, portfolio_cumulative, 
                         linewidth=2.5, label='Portafolio', color='darkblue')
                ax10.plot(benchmark_cumulative.index, benchmark_cumulative, 
                         linewidth=2.5, label='S&P 500 (Benchmark)', color='orange', linestyle='--')
                ax10.set_title('Comparación con el Mercado (S&P 500)', fontsize=14, fontweight='bold')
                ax10.set_xlabel('Fecha', fontsize=12)
                ax10.set_ylabel('Crecimiento (base 1)', fontsize=12)
                ax10.legend(loc='best', fontsize=11)
                ax10.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig10)
                plt.close()
                
                # Métricas de comparación
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Portafolio", f"{(portfolio_cumulative.iloc[-1] - 1):.2%}")
                with col2:
                    st.metric("Retorno S&P 500", f"{(benchmark_cumulative.iloc[-1] - 1):.2%}")
                with col3:
                    alpha = (portfolio_cumulative.iloc[-1] - 1) - (benchmark_cumulative.iloc[-1] - 1)
                    st.metric("Alpha", f"{alpha:.2%}", delta="vs Mercado")
                
            except Exception as e:
                st.warning("No se pudo descargar el benchmark (S&P 500). Continuando sin comparación.")
            
            # ============================================
            # SECCIÓN 11: EXPORTAR RESULTADOS
            # ============================================
            st.header("1️⃣1️⃣ Exportar Resultados")
            
            # Preparar datos para exportar
            df_simulaciones = pd.DataFrame({
                'Volatilidad': results[0, :],
                'Retorno': results[1, :],
                'Sharpe': results[2, :]
            })
            
            df_portafolio_actual = pd.DataFrame({
                'Ticker': tickers,
                'Peso': weights,
                'Inversion': weights * inversion_inicial
            })
            
            df_portafolio_optimo = pd.DataFrame({
                'Ticker': tickers,
                'Peso_Optimo': optimal_weights,
                'Inversion_Sugerida': optimal_weights * inversion_inicial
            })
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_simulaciones = df_simulaciones.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Simulaciones",
                    data=csv_simulaciones,
                    file_name="simulaciones_monte_carlo.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_actual = df_portafolio_actual.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Portafolio Actual",
                    data=csv_actual,
                    file_name="portafolio_actual.csv",
                    mime="text/csv"
                )
