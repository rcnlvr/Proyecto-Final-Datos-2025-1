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
    # Cálculo de rendimientos
    returns = df.pct_change().dropna()
    # Cálculo de rendimientos acumulados de los activos
    cumulative_returns = (1 + returns).cumprod() - 1
    # Calculamos los precios normalizados de los activos
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_rendimiento_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    return (1 + returns.iloc[-window:]).prod() - 1

# Nuevas funciones para VaR
def calcular_var(returns, confidence=0.95):
    # Usamos el cuantil del nivel de confianza deseado para calcular el VaR paramétrico
    VaR = returns.quantile(1 - confidence)
    return VaR

def calcular_var_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var(window_returns)

# Función para calcular el VaR usando el método de Montecarlo
def calcular_var_mc(returns, num_simulaciones=100000, nivel_confianza=0.95):
    # Calculamos la media
    media = np.mean(returns)
    # Calculamos desviación estándar
    desviacion_estandar = np.std(returns)
    # Realizamos las simulaciones de Monte Carlo usando la media y la desviacion estándar de cada activo
    simulaciones = np.random.normal(media, desviacion_estandar, num_simulaciones)
    # Ordenamos las simulaciones
    simulaciones_ordenadas = np.sort(simulaciones)
    # Definimos el nivel de confianza que tendrá nuestro VaR
    percentil = int((1 - nivel_confianza) * num_simulaciones)
    # Calculamos el VaR viendo cuántas veces las simulaciones caen en el nivel de confianza que queremos
    var = simulaciones_ordenadas[percentil]
    return var

def var_mc_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var_mc(window_returns)

def crear_histograma_distribucion(returns, var_95, var_95_mc, title):
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

    # Añadir líneas verticales para ubicar el VaR con Montecarlo
    fig.add_trace(go.Scatter(
        x=[var_95_mc, var_95_mc],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95% (Montecarlo)',
        line=dict(color='purple', width=2, dash='dash')
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
    "1 mes": end_date - timedelta(days=30*1.5),
    "3 meses": end_date - timedelta(days=90*1.5),
    "6 meses": end_date - timedelta(days=180*1.5),
    "1 año": end_date - timedelta(days=365*1.5),
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
    tab1, tab2, tab3 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio", "Marco Teórico"])

    with tab1:
        st.header("Análisis de Activos Individuales")
        
        selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)
        
        # Calcular VaR para el activo seleccionado
        var_95 = calcular_var(returns[selected_asset])
        # Calcular VaR con montecarlo para el activo seleccionado
        var_95_mc = calcular_var_mc(returns[selected_asset])        
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
        col2.metric("VaR 95% (Histórico)", f"{var_95:.2%}")
        col3.metric("VaR 95% (Montecarlo)", f"{var_95_mc:.2%}")

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
            var_mc_asset = calcular_var_mc(returns[selected_asset])
            fig_hist_asset = crear_histograma_distribucion(
                returns[selected_asset],
                var_asset,
                var_mc_asset,
                f'Distribución de Retornos - {selected_asset}'
            )
            st.plotly_chart(fig_hist_asset, use_container_width=True, key="hist_asset")

        with col2:
            # Histograma para el benchmark
            var_bench = calcular_var(returns[benchmark])
            var_bench_mc = calcular_var_mc(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                var_bench_mc,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_1")
            
        col1, col2 = st.columns(2)

    with tab2:
        st.header("Análisis del Portafolio")
        
        # Calcular VaR del portafolio
        portfolio_var_95 = calcular_var(portfolio_returns)
        # Calcular VaR del portafolio con simulaciones de Montecarlo
        portfolio_var_mc = calcular_var_mc(portfolio_returns)
        

        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
        col2.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")
        col3.metric("VaR 95% del Portafolio con Montecarlo", f"{portfolio_var_mc:.2%}")


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
            var_port_mc = calcular_var_mc(portfolio_returns)
            fig_hist_port = crear_histograma_distribucion(
                portfolio_returns,
                var_port,
                var_port_mc,
                'Distribución de Retornos - Portafolio'
            )
            st.plotly_chart(fig_hist_port, use_container_width=True, key="hist_port")
            
        with col2:
            # Histograma para el benchmark
            var_bench = calcular_var(returns[benchmark])
            var_bench_mc = calcular_var_mc(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                var_bench_mc,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_2")

        # Rendimientos y métricas de riesgo en diferentes ventanas de tiempo
        st.subheader("Rendimientos y Métricas de Riesgo en Diferentes Ventanas de Tiempo")
        ventanas = [30, 91,184, 365]
        
        # Crear DataFrames separados para cada métrica
        rendimientos_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        var_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        var_mc_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])

        for ventana in ventanas:
            # Rendimientos
            rendimientos_ventanas[f'{ventana}d'] = pd.Series({
                'Portafolio': calcular_rendimiento_ventana(portfolio_returns, ventana),
                **{symbol: calcular_rendimiento_ventana(returns[symbol], ventana) for symbol in simbolos},
                selected_benchmark: calcular_rendimiento_ventana(returns[benchmark], ventana)
            })
            
            # VaR
            var_temp = {}
            var_mc_temp = {}
            
            # Para el portafolio
            port_var = calcular_var_ventana(portfolio_returns, ventana)
            var_temp['Portafolio'] = port_var
            port_var_mc = var_mc_ventana(portfolio_returns, ventana)
            var_mc_temp['Portafolio'] = port_var_mc
            
            # Para cada símbolo
            for symbol in simbolos:
                var = calcular_var_ventana(returns[symbol], ventana)
                var_temp[symbol] = var
                var_mc = var_mc_ventana(returns[symbol], ventana)
                var_mc_temp[symbol] = var_mc
            
            # Para el benchmark
            bench_var = calcular_var_ventana(returns[benchmark], ventana)
            var_temp[selected_benchmark] = bench_var
            bench_var_mc = var_mc_ventana(returns[benchmark], ventana)
            var_mc_temp[selected_benchmark] = bench_var_mc
            
            var_ventanas[f'{ventana}d'] = pd.Series(var_temp)
            var_mc_ventanas[f'{ventana}d'] = pd.Series(var_mc_temp)
            

        # Mostrar las tablas
        col1, col2, col3 = st.columns(3)
        col1.subheader("Rendimientos")
        col1.dataframe(rendimientos_ventanas.style.format("{:.2%}"))

        col2.subheader("VaR 95%")
        col2.dataframe(var_ventanas.style.format("{:.2%}"))
        
        col3.subheader("VaR 95% MC")
        col3.dataframe(var_mc_ventanas.style.format("{:.2%}"))

        with tab3:
            st.header("Marco Teórico")
            st.write("""
            **Abstract** 
            
El VaR o Valor en Riesgo, es una técnica estadística que usamos para medir el riesgo de pérdida que tiene un portafolio de inversión. En general, se expresa en términos de porcentaje o valor monetario.

Este modelo nos ayuda a cuantificar la máxima pérdida que puede experimentar nuestro portafolio en un horizonte de tiempo, con un nivel de confianza determinado. La importancia del VaR radica en la visión cuantificable que tenemos del riesgo, pues al conocer el peor escenario de pérdidas, se pueden tomar mejores decisiones sobre la asignación y diversificación de activos. 

Este modelo de riesgo es utilizado por varias instituciones financieras y reguladoras para evitar escenarios como el de la crisis económica del 2008. Resultados como la Teoría de Markowitz sobre la diversificación de portafolios, van de la mano con este concepto, pues al estudiar la relación riesgo-rendimiento nos permiten optimizar ganancias de acuerdo con un nivel de riesgo.

**I.	Introducción**

En un contexto de inversiones es importante cuantificar el riesgo ya que éste no puede ser eliminado en su totalidad y se permea con cada decisión que tomamos, pero ¿Qué es el riesgo?

El riesgo es visto como una métrica de posibilidad o amenaza de pérdidas monetarias en una inversión durante un tiempo específico a un nivel de confianza. El VaR es la métrica que cuantifica el nivel de riesgo de éstas pérdidas durante un período de tiempo específico y a un nivel de confianza.

Matemáticamente hablando, calcular el riesgo es calcular la desviación estándar de los rendimientos, mientras que el VaR es el percentil 1-α de la distribución de pérdidas. Así, el VaR al 100(1- α)% representa la pérdida máxima esperada con probabilidad de 1- α.

En general, esta métrica se expresa en términos de porcentaje o de valor monetario, lo que significa que proporciona una estimación de cuánto sería la pérdida máxima o el porcentaje de pérdida con relación al valor total de la inversión.

-	Tipos de VaR

Existen distintos métodos para calcular el VaR, entre los que podemos destacar:

o	VaR paramétrico: También conocido como VaR por covarianza. Se basa en la distribución de rendimientos de los activos. Usamos estadísticas históricas para estimar parámetros como la media y la desviación estándar y el VaR se calcula asumiendo que los rendimientos siguen una distribución normal o alguna otra esécífica.

o	Método histórico: Se basa únicamente en datos pasados. Se toma un período histórico y se ordena de menor a mayor. El VaR se calcula seleccionando el rendimiento que se encuentra en el percentil α, donde α es el nivel de confianza.

o	Simulación por Monte Carlo: tiene un enfoque computacional que modela miles de trayectorias posibles para los rendimientos de una cartera. Se usan las estadísticas históricas para modelar correlaciones entre activos. Luego, al ejecutar las simulaciones, el VaR se calcula observando cuántas veces la cartera tiene pérdidas que superan el umbral de VaR.

**II.	Justificación**

En este proyecto, veremos el funcionamiento de este último y lo compararemos con el VaR paramétrico, para observar la diferencia entre estos cálculos y así tener 2 visiones diferentes del riesgo que acompaña a nuestra inversión. La importancia radica en apoyarnos en la toma de decisiones para crear una cartera de inversión que tenga riesgo mínimo. 

De igual forma sabremos cuál es nuestra tolerancia al riesgo, ya que cada activo tiene un valor de VaR, pero al momento de seleccionar varios activos, el VaR cambia, lo que nos permitirá ajustar nuestras decisiones sobre qué activos queremos comprar. 

Esta es una herramienta fundamental al momento de crear portafolios en el ambiente de renta variable, donde los rendimientos suelen ser más volátiles.

**III.	Desarrollo**

Usaremos un dashboard de streamlit para llevar a cabo este proyecto.

Aquí tendremos un código que nos permitirá escoger 5 activos diferentes y el peso que tendrán en el portafolio, elegiremos un benchmark (marca que será la “meta” a superar de nuestro portafolio) y el horizonte de tiempo hasta un año. Toda esta información la vamos a importar directamente de la biblioteca de Yahoo! Finance, para hacer los cálculos con datos en vivo.

Dentro definiremos una función que calcule el VaR por simulaciones de Monte Carlo y por el método paramétrico con ayuda de la biblioteca Numpy, mostraremos ambos resultados y dado un portafolio, observaremos la gráfica de rendimientos comparada con el benchmark. Haremos un histograma de la distribución de los rendimientos tanto de cada activo como del portafolio en general y aquí podremos visualizar el VaR calculado.

La ventaja es que tendremos distintas ventanas de tiempo: 1, 3, 6 y 12 meses, para observar la variación del riesgo dependiendo del horizonte que tengamos para hacer la inversión, ya que es un dato importante para considerar al momento de elegir los activos que formarán nuestra cartera.

También incluiremos una tabla que nos indique el porcentaje de riesgo de nuestro portafolio en los horizontes de tiempo que establecimos, lo cual nos ayudará para conocer el comportamiento de los activos en el largo plazo y saber en qué plazo nos conviene más cada uno, lo que es crucial para la toma de decisiones.

**IV.	Conclusiones**

Como ya mencionamos, el VaR es una herramienta fundamental en la gestión de portafolios y con este trabajo podremos tener una visión clara del riesgo que involucra invertir en cierto activo o en varios.

Esto nos permite tomar decisiones con más seguridad, lo que es una gran ventaja en este ambiente ya que lo que queremos es hacer crecer nuestro dinero, entonces, conocer el riesgo que implica una inversión y ser capaz de visualizarlo gráficamente es algo que sin duda nos beneficiará a corto y largo plazo.
    
    """)
