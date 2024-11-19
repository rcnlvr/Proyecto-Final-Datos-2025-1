import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Funciones auxiliares
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_metricas(df):
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_rendimiento_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    return (1 + returns.iloc[-window:]).prod() - 1

# Nuevas funciones para VaR y CVaR
def calcular_var(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    return VaR

def calcular_var_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var(window_returns)

# Función para calcular VaR usando simulación de Monte Carlo
def calcular_var_montecarlo(normalized_prices, confidence_level=0.95, num_simulations=10000):
    simulated_returns = np.random.choice(normalized_prices, size=(num_simulations, len(normalized_prices)))
    portfolio_returns = np.sum(simulated_returns, axis=1)
    var_montecarlo = np.percentile(portfolio_returns, (1 - confidence_level))
    return var_montecarlo

def var_montecarlo_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var_montecarlo(window_returns)

def crear_histograma_distribucion(returns, var_95, title):
    # Crear el histograma base
    fig = go.Figure()
    
    # Calcular los bins para el histograma
    counts, bins = np.histogram(returns, bins=50)
    
    # Separar los bins en dos grupos: antes y después del VaR
    mask_before_var = bins[:-1] <= var_95
    
    # Añadir histograma para valores antes del VaR (rojo)
    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],
        y=counts[mask_before_var],
        width=np.diff(bins)[mask_before_var],
        name='Retornos < VaR',
        marker_color='rgba(255, 65, 54, 0.6)'
    ))
    
    # Añadir histograma para valores después del VaR (azul)
    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Retornos > VaR',
        marker_color='rgba(31, 119, 180, 0.6)'
    ))
    
    # Añadir líneas verticales para ubicar el VaR
    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Actualizar el diseño
    fig.update_layout(
        title=title,
        xaxis_title='Retornos',
        yaxis_title='Frecuencia',
        showlegend=True,
        barmode='overlay',
        bargap=0
    )
    
    return fig

# Configuración de la página
st.set_page_config(page_title="Analizador de Portafolio", layout="wide")
st.sidebar.title("Analizador de Portafolio de Inversión")

# Entrada de símbolos y pesos
simbolos_input = st.sidebar.text_input("Ingrese los símbolos de las acciones separados por comas (por ejemplo: AAPL,GOOGL,MSFT):", "AAPL,GOOGL,MSFT,AMZN,NVDA")
pesos_input = st.sidebar.text_input("Ingrese los pesos correspondientes separados por comas (deben sumar 1):", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip() for s in simbolos_input.split(',')]
pesos = [float(w.strip()) for w in pesos_input.split(',')]

# Selección del benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "IPC": "^MXX",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]

# Selección de la ventana de tiempo
end_date = datetime.now()
start_date_options = {
    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
}
selected_window = st.sidebar.selectbox("Seleccione la ventana de tiempo para el análisis:", list(start_date_options.keys()))
start_date = start_date_options[selected_window]

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.sidebar.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
else:
    # Obtener datos
    all_symbols = simbolos + [benchmark]
    df_stocks = obtener_datos_acciones(all_symbols, start_date, end_date)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    # Rendimientos del portafolio
    portfolio_returns = calcular_rendimientos_portafolio(returns[simbolos], pesos)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Crear pestañas
    tab1, tab2 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio"])

    with tab1:
        st.header("Análisis de Activos Individuales")
        
        selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)
        
        # Calcular VaR para el activo seleccionado
        var_95 = calcular_var(returns[selected_asset])
        # Calcular VaR con montecarlo para el activo seleccionado
        var_95_montecarlo = calcular_var_montecarlo(returns[selected_asset])
        
        col1, col2, col3 = st.columns(2)
        col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
        col2.metric("VaR 95%", f"{var_95:.2%}")
        col3.metric("VaR 95% MC", f"{var_95_montecarlo:.2%}")

        # Gráfico de precio normalizado del activo seleccionado vs benchmark
        fig_asset = go.Figure()
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[benchmark], name=selected_benchmark))
        fig_asset.update_layout(title=f'Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado')
        st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")
        
        st.subheader(f"Distribución de Retornos: {selected_asset} vs {selected_benchmark}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma para el activo seleccionado
            var_asset = calcular_var(returns[selected_asset])
            fig_hist_asset = crear_histograma_distribucion(
                returns[selected_asset],
                var_asset,
                f'Distribución de Retornos - {selected_asset}'
            )
            st.plotly_chart(fig_hist_asset, use_container_width=True, key="hist_asset")
            
        with col2:
            # Histograma para el benchmark
            var_bench = calcular_var(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_1")


    with tab2:
        st.header("Análisis del Portafolio")
        
        # Calcular VaR y CVaR para el portafolio
        portfolio_var_95 = calcular_var(portfolio_returns)

        col1, col2 = st.columns(2)
        col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
        col2.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")

        # Gráfico de rendimientos acumulados del portafolio vs benchmark
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio'))
        fig_cumulative.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[benchmark], name=selected_benchmark))
        fig_cumulative.update_layout(title=f'Rendimientos Acumulados: Portafolio vs {selected_benchmark}', xaxis_title='Fecha', yaxis_title='Rendimiento Acumulado')
        st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_returns")

        st.subheader("Distribución de Retornos del Portafolio vs Benchmark")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma para el portafolio
            var_port = calcular_var(portfolio_returns)
            fig_hist_port = crear_histograma_distribucion(
                portfolio_returns,
                var_port,
                'Distribución de Retornos - Portafolio'
            )
            st.plotly_chart(fig_hist_port, use_container_width=True, key="hist_port")
            
        with col2:
            # Histograma para el benchmark
            var_bench = calcular_var(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_2")

        # Rendimientos y métricas de riesgo en diferentes ventanas de tiempo
        st.subheader("Rendimientos y Métricas de Riesgo en Diferentes Ventanas de Tiempo")
        ventanas = [30, 90, 180, 252]
        
        # Crear DataFrames separados para cada métrica
        rendimientos_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        var_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])

        for ventana in ventanas:
            # Rendimientos
            rendimientos_ventanas[f'{ventana}d'] = pd.Series({
                'Portafolio': calcular_rendimiento_ventana(portfolio_returns, ventana),
                **{symbol: calcular_rendimiento_ventana(returns[symbol], ventana) for symbol in simbolos},
                selected_benchmark: calcular_rendimiento_ventana(returns[benchmark], ventana)
            })
            
            # VaR
            var_temp = {}
            
            # Para el portafolio
            port_var = calcular_var_ventana(portfolio_returns, ventana)
            var_temp['Portafolio'] = port_var
            
            # Para cada símbolo
            for symbol in simbolos:
                var = calcular_var_ventana(returns[symbol], ventana)
                var_temp[symbol] = var
            
            # Para el benchmark
            bench_var = calcular_var_ventana(returns[benchmark], ventana)
            var_temp[selected_benchmark] = bench_var
            
            var_ventanas[f'{ventana}d'] = pd.Series(var_temp)
        
        # Mostrar las tablas
        st.subheader("Rendimientos")
        st.dataframe(rendimientos_ventanas.style.format("{:.2%}"))
        
        st.subheader("VaR 95%")
        st.dataframe(var_ventanas.style.format("{:.2%}"))

        # Gráfico de comparación de rendimientos
        fig_comparison = go.Figure()
        for index, row in rendimientos_ventanas.iterrows():
            fig_comparison.add_trace(go.Bar(x=ventanas, y=row, name=index))
        fig_comparison.update_layout(title='Comparación de Rendimientos', xaxis_title='Días', yaxis_title='Rendimiento', barmode='group')
        # Gráfico de comparación de rendimientos
        st.plotly_chart(fig_comparison, use_container_width=True, key="returns_comparison")
