#PARA GITHUB
#------------------------------------

#Librerias
import numpy as np
import pandas as pd
import seaborn as sna
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import os
from subprocess import check_output
import warnings
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from datetime import datetime as dt

import yfinance as yf
import datetime as dt
from datetime import date, timedelta

from dash import Dash,dcc,html,Input,Output

#-----------------------------------------------------------------------------
#PREPARACIÓN DE DATOS

# Acciones
tickers = ["KO", "PG", "PEP", "HON", "CAT", "MMM"]
data = yf.download(tickers, start="2022-01-01", end="2025-11-01",  auto_adjust=False)["Close"]

returns = data.pct_change() * 100

# Últimos 3 años
returns_3y = returns.loc[returns.index >= (returns.index[-1] - pd.DateOffset(years=3))]

# Cryptos
df = pd.read_csv("Crypto_historical_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ticker", "Date"])

#--------------------------------------------------------------------------------------------------------------------------------

#CREAR APP DASH


app = Dash(__name__)

#Github: agregar linea de server que usa git
server=app.server

app.layout = html.Div([
    html.H1("Dashboard Fondo de Inversión: Stocks & Criptomonedas", style={"textAlign": "center"}),

    html.P("Usa el menú para cambiar entre las gráficas.",
           style={"textAlign": "center", "fontSize": "18px"}),

    dcc.Tabs([

        #-------------------------------------------------------
        # Precios & Retornos de Acciones
        
        dcc.Tab(label="Precios & Retornos de Acciones", children=[
            html.Br(),
            html.H2("Visualización de Precios y Retornos de Acciones"),

            html.Label("Acción:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': t, 'value': t} for t in tickers],
                value=['KO'],
                multi=True
            ),

            html.Label("Selecciona tipo de gráfico:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Precios de Cierre', 'value': 'price'},
                    {'label': 'Retornos Mensuales (%)', 'value': 'return'}
                ],
                value='price'
            ),

            html.Label("Selecciona rango de fechas:"),
            dcc.RangeSlider(
                id='slider',
                min=0,
                max=len(data.index) - 1,
                step=1,
                value=[0, len(data.index) - 1],
                marks={i: str(date.strftime('%Y-%m')) for i, date in enumerate(data.index[::60])}
            ),

            dcc.Graph(id='graph-output')
        ]),

        # ----------------------------------------------------
        # Histograma Retornos de Acciones
        
        dcc.Tab(label="Distribución de Retornos de Acciones", children=[
            html.Br(),
            html.H2("Distribución de Retornos por Acción (Últimos 3 años)"),

            html.Label("Selecciona una acción:"),
            dcc.Dropdown(
                id='dist-dropdown',
                options=[{'label': t, 'value': t} for t in tickers],
                value='KO'
            ),

            dcc.Graph(id='dist-graph')
        ]),

        # --------------------------------------------------
        # Bollinger Cryptos
        
        dcc.Tab(label="Gráfica de Bollinger (Criptos)", children=[
            html.Br(),
            html.H1("Gráfica de Bollinger - Criptomonedas", style={"textAlign": "center"}),

            html.P("Selecciona una criptomoneda:", style={"fontWeight": "bold"}),

            dcc.Dropdown(
                id="selected_ticker",
                options=[{"label": t, "value": t} for t in df["ticker"].unique()],
                value=df["ticker"].unique()[0],
                multi=False
            ),

            dcc.Graph(id="bollinger_chart")
        ]),

        #
        # Evolución Cryptos
        
        dcc.Tab(label="Evolución de Precios (Criptos)", children=[
            html.Br(),
            html.H1("Evolución de precios de criptomonedas", style={"textAlign": "center"}),

            html.P("Selecciona una o varias criptomonedas para visualizar su evolución:"),

            dcc.Dropdown(
                id="crypto-dropdown",
                options=[{"label": name, "value": ticker} 
                         for name, ticker in zip(df["name"].unique(), df["ticker"].unique())],
                value="BTC-USD",
                multi=False,
                style={"width": "70%"}
            ),

            dcc.Graph(id="line-chart")
        ]),
    ])
])

#---------------------------------------------------------------
#  CALLBACKS


# Grafica 1
@app.callback(
    Output('graph-output', 'figure'),
    [Input('ticker-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('slider', 'value')]
)
def update_graph(selected_tickers, metric, date_range):

    start_idx, end_idx = date_range
    start_date = data.index[start_idx]
    end_date = data.index[end_idx]

    mask = (data.index >= start_date) & (data.index <= end_date)
    filtered_data = data.loc[mask, selected_tickers]
    filtered_returns = returns.loc[mask, selected_tickers]

    if metric == 'price':
        fig = px.line(
            filtered_data,
            x=filtered_data.index,
            y=filtered_data.columns,
            title="Precios de Cierre",
            labels={'value': 'Precio (USD)', 'index': 'Fecha'}
        )
    else:
        fig = px.line(
            filtered_returns,
            x=filtered_returns.index,
            y=filtered_returns.columns,
            title="Retornos Mensuales (%)",
            labels={'value': 'Retorno (%)', 'index': 'Fecha'}
        )

    fig.update_layout(legend_title_text='Acción', template='plotly_white')
    return fig

# Grafica 2
@app.callback(
    Output('dist-graph', 'figure'),
    Input('dist-dropdown', 'value')
)
def update_distribution(selected_ticker):
    fig = px.histogram(
        returns_3y,
        x=selected_ticker,
        nbins=30,
        histnorm='probability density',
        title=f"Distribución de Retornos - {selected_ticker} (Últimos 3 años)",
    )
    fig.update_layout(template='plotly_white')
    return fig

# Grafica 3 
@app.callback(
    Output("bollinger_chart", "figure"),
    Input("selected_ticker", "value")
)
def update_bollinger(selected_ticker):

    data_crypto = df[df["ticker"] == selected_ticker].copy()

    window = 20
    data_crypto["MA"] = data_crypto["Close"].rolling(window).mean()
    data_crypto["STD"] = data_crypto["Close"].rolling(window).std()

    data_crypto["Upper"] = data_crypto["MA"] + (2 * data_crypto["STD"])
    data_crypto["Lower"] = data_crypto["MA"] - (2 * data_crypto["STD"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_crypto["Date"], y=data_crypto["Close"], name="Precio"))
    fig.add_trace(go.Scatter(x=data_crypto["Date"], y=data_crypto["MA"], name="Media Móvil"))
    fig.add_trace(go.Scatter(x=data_crypto["Date"], y=data_crypto["Upper"], name="Upper Band"))
    fig.add_trace(go.Scatter(x=data_crypto["Date"], y=data_crypto["Lower"], name="Lower Band",
                             fill="tonexty", fillcolor="rgba(128,128,128,0.3)"))

    fig.update_layout(
        title=f"{data_crypto['name'].iloc[0]}",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        template="plotly_white"
    )

    return fig

# Grafica 4
@app.callback(
    Output("line-chart", "figure"),
    Input("crypto-dropdown", "value")
)
def update_chart(selected_crypto):

    filtered_df = df[df["ticker"] == selected_crypto]

    fig = px.line(
        filtered_df,
        x="Date",
        y="Close",
        color="name",
        title="Evolución del precio (USD)",
        labels={"Close": "Precio (USD)", "Date": "Fecha"}
    )

    fig.update_traces(line=dict(width=3))
    fig.update_layout(template="plotly_white")

    return fig


#------------------------------------------------------------------
#  Correr APP


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",port=10000)
