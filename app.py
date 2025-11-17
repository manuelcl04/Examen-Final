# ============================================================
#                   IMPORTS
# ============================================================
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

from dash import Dash, dcc, html, Input, Output

import dash_bootstrap_components as dbc

# ============================================================
#               CARGA DE DATOS (asumiendo que ya existen)
# ============================================================

tickers = ["KO", "PG", "PEP", "HON", "CAT", "MMM"]
data = yf.download(tickers, start="2022-01-01", end="2025-11-01", auto_adjust=False)["Close"]

df1 = pd.read_csv("Crypto1.csv")
df2 = pd.read_csv("Crypto2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# -----------------  TAB 1 ----------------------
returns = data.pct_change() * 100
returns_3y = returns.loc[returns.index >= (returns.index[-1] - pd.DateOffset(years=3))]

# -----------------  TAB 3 ----------------------
df_crypto_raw = pd.concat([df1, df2], ignore_index=True)
df_crypto_raw["Date"] = pd.to_datetime(df_crypto_raw["Date"]).dt.tz_localize(None)

# ============================================================
#          *** NUEVO PROCESAMIENTO PARA TAB 4 ***
# ============================================================
df_tab4 = pd.concat([df1, df2], ignore_index=True)
df_tab4["Date"] = pd.to_datetime(df_tab4["Date"]).dt.tz_localize(None)

# (mantengo tu lista top_cryptos tal cual — omitida aquí por brevedad en este bloque)
top_cryptos = ["DoubleZero", "Vaulta", "Aave", "AB", "Cardano", "Aerodrome Finance",
    "Algorand", "ApeCoin", "Ape and Pepe", "Aptos", "Arbitrum", "ARK",
    "Aster Staked BNB", "Aster", "Aethir", "Cosmos Hub", "Avalanche",
    "Bybit Staked SOL", "Bitcoin Cash", "Beldex", "BFUSD", "Bitget Token", "BNB",
    "Binance Staked SOL", "Bonk", "SwissBorg", "Bitcoin SV", "Bitcoin",
    "Avalanche Bridged BTC (Avalanche)", "BitTorrent",
    "BlackRock USD Institutional Digital Liquidity Fund", "Binance-Peg BUSD",
    "PancakeSwap", "Coinbase Wrapped BTC", "Coinbase Wrapped Staked ETH",
    "Concordium", "Conflux", "Chiliz", "Mantle Restaked ETH", "Compound", "Cronos",
    "Curve DAO", "crvUSD", "Cap USD", "Dai",
    "Polygon PoS Bridged DAI (Polygon POS)", "Dash", "Decred", "DeXe", "Dogecoin",
    "Binance-Peg Dogecoin", "Polkadot", "ether.fi Staked ETH",
    "EigenCloud (prev. EigenLayer)", "Ethena", "Ethereum Name Service",
    "Ethereum Classic", "Ethereum", "Ether.fi", "Stader ETHx",
    "Renzo Restaked ETH", "Fartcoin", "First Digital USD",
    "Artificial Superintelligence Alliance", "Filecoin", "FLOKI", "Flow", "Flare",
    "Fluid", "Legacy Frax Dollar", "Frax Ether", "Fasttoken", "GALA", "GHO",
    "Gnosis", "The Graph", "Gate", "Humanity", "Provenance Blockchain", "Hedera",
    "Helium", "HTX DAO", "Hyperliquid", "Internet Computer", "Immutable",
    "Injective", "IOTA", "Story", "JasmyCoin", "Jito Staked SOL",
    "Jupiter Perpetuals Liquidity Provider Token", "JUST", "Jito", "Jupiter",
    "Jupiter Staked SOL", "Kaia", "Kaspa", "KuCoin", "Lombard Staked BTC",
    "Lido DAO", "LEO Token", "Chainlink", "Loaded Lions", "Liquid Staked ETH",
    "Litecoin", "MemeCore", "Decentraland", "Merlin Chain", "Mantle Staked Ether",
    "Mantle", "Morpho", "Marinade Staked SOL", "MYX Finance", "NEAR Protocol",
    "NEO", "NEXO", "AINFT", "Olympus", "OKB", "Ondo", "Optimism",
    "StakeWise Staked ETH", "OUSG", "PAX Gold", "Pendle", "Pudgy Penguins", "Pepe",
    "Pi Network", "POL (ex-MATIC)", "Pyth Network", "PayPal USD", "Quant",
    "Raydium", "Render", "Rocket Pool ETH", "Ripple USD", "Kelp DAO Restaked ETH",
    "Reserve Rights", "THORChain", "Sonic", "The Sandbox",
    "BENQI Liquid Staked AVAX", "sBTC", "Savings Dai", "Sei", "Shiba Inu", "Sky",
    "Synthetix", "Solana", "Wrapped SOL", "Solv Protocol BTC", "SPX6900",
    "Lido Staked Ether", "Starknet", "Stacks", "Sui", "Sun Token",
    "Ethena Staked USDe", "Swell Ethereum", "Maple Finance", "syrupUSDC", "tBTC",
    "Treehouse ETH", "Theta Network", "Celestia", "Ribbita by Virtuals",
    "Toncoin", "Official Trump", "TRON", "TrueUSD", "Trust Wallet", "Unit Bitcoin",
    "Uniswap", "Usual USD", "USD1", "USDai", "USDB", "USDC",
    "Binance Bridged USDC (BNB Smart Chain)", "USDD", "Ethena USDe",
    "Falcon USD", "Global Dollar", "USDS", "Tether",
    "Mantle Bridged USDT (Mantle)", "USDT0", "USDtb", "Stables Labs USDX",
    "Ondo US Dollar Yield", "VeChain", "Virtuals Protocol", "Vision", "Wormhole",
    "Walrus", "Wrapped AVAX", "Wrapped Beacon ETH", "Wrapped BNB",
    "WhiteBIT Coin", "Wrapped Bitcoin", "Arbitrum Bridged WBTC (Arbitrum One)",
    "Wrapped eETH", "Arbitrum Bridged Wrapped eETH (Arbitrum)", "WETH",
    "Binance-Peg WETH", "Arbitrum Bridged WETH (Arbitrum One)",
    "L2 Standard Bridged WETH (Base)",
    "Polygon PoS Bridged WETH (Polygon POS)", "Mantle Bridged WETH (Mantle)",
    "Wrapped HYPE", "dogwifhat", "Worldcoin", "World Liberty Financial",
    "Wrapped stETH", "Tether Gold", "XDC Network", "Stellar", "Monero", "Plasma",
    "XRP", "Solv Protocol Staked BTC", "Tezos", "Zebec Network", "Zcash", "ZKsync",
    "Zora"]



df_filtered = df_tab4[df_tab4["name"].isin(top_cryptos)].copy()
df_filtered = df_filtered[df_filtered["Date"] >= "2022-01-01"]

df_filtered = (
    df_filtered.groupby([
        pd.Grouper(key="Date", freq="W"),
        "ticker", "name"
    ])
    .agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    .reset_index()
)

df_filtered = df_filtered.sort_values(["Date", "ticker"])

# ============================================================
#               FUNCIÓN ANIMACIÓN (NUEVA)
# ============================================================
def create_animated_figure(selected_cryptos):
    if isinstance(selected_cryptos, str):
        selected_cryptos = [selected_cryptos]

    if not selected_cryptos:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No se han seleccionado criptomonedas.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(template="plotly_white")
        return empty_fig

    df_selection = df_filtered[df_filtered["name"].isin(selected_cryptos)].copy()
    if df_selection.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No hay datos para las criptomonedas seleccionadas en el rango.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(template="plotly_white")
        return empty_fig

    dates_unique = sorted(df_selection['Date'].unique())
    frames = []
    for i, d in enumerate(dates_unique):
        frame_data = df_selection[df_selection["Date"] <= d].copy()
        frame_data["frame"] = i
        frames.append(frame_data)

    df_animated = pd.concat(frames, ignore_index=True)

    # uso melt para asegurar formato "long" consistente (aunque aquí no es estrictamente necesario)
    # pero px.line con animation_frame funciona mejor si cada fila es una observación
    fig = px.line(
        df_animated,
        x="Date",
        y="Close",
        color="name",
        animation_frame="frame",
        range_x=[df_selection["Date"].min(), df_selection["Date"].max()],
        range_y=[
            df_selection["Close"].min() * 0.90,
            df_selection["Close"].max() * 1.10,
        ],
        title=f"Evolución Acumulativa del Precio - {', '.join(selected_cryptos)}",
        labels={"Close": "Precio (USD)", "Date": "Fecha", "name": "Criptomoneda"},
        template="plotly_white"
    )

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=650,
        hovermode="x unified",
        showlegend=True,
        title=dict(
            text=f"Evolución Acumulativa del Precio - {', '.join(selected_cryptos)}",
            x=0.5,
            xanchor="center",
            font=dict(size=22, family="Arial, sans-serif", color="#1f2c56", weight="bold"),
        ),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        yaxis=dict(title="Precio (USD)", showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)"),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    # Ajustar velocidad si existen los controles
    try:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 50
    except Exception:
        pass

    return fig

# ============================================================
#               APP & TABS
# ============================================================
app10 = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server=app10.server

tabs = dbc.Tabs([

    # ======================================================
    # TAB 1 CORPORATIVO
    # ======================================================
    dbc.Tab(label="1. Precios & Retornos Acciones", children=[
        html.Div(
            style={"backgroundColor": "#f7f9fc", "padding": "40px", "fontFamily": "Segoe UI"},
            children=[
                html.H1(
                    "Análisis de Precios y Retornos de Acciones",
                    style={"textAlign": "center", "color": "#1C2340", "fontWeight": "700", "marginBottom": "40px", "fontSize": "36px"}
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "center", "gap": "30px", "marginBottom": "35px"},
                    children=[
                        html.Div(
                            style={"width": "350px", "backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0px 2px 8px rgba(0,0,0,0.1)"},
                            children=[
                                html.Label("Selecciona Acción", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="ticker-dropdown",
                                    options=[{'label': t, 'value': t} for t in tickers],
                                    value=["KO"],
                                    multi=True,
                                    style={"marginTop": "10px"}
                                ),
                            ]
                        ),
                        html.Div(
                            style={"width": "350px", "backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0px 2px 8px rgba(0,0,0,0.1)"},
                            children=[
                                html.Label("Tipo de Métrica", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="metric-dropdown",
                                    options=[
                                        {"label": "Precios de Cierre", "value": "price"},
                                        {"label": "Retornos Mensuales (%)", "value": "return"}
                                    ],
                                    value="price",
                                    style={"marginTop": "10px"}
                                ),
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={"width": "80%", "margin": "0 auto", "backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "boxShadow": "0px 2px 8px rgba(0,0,0,0.08)", "marginBottom": "30px"},
                    children=[
                        html.Label("Rango de Fechas", style={"fontWeight": "600"}),
                        dcc.RangeSlider(
                            id="slider",
                            min=0,
                            max=len(data.index) - 1,
                            step=1,
                            value=[0, len(data.index) - 1],
                            marks={i: str(d.strftime("%Y-%m")) for i, d in enumerate(data.index[::60])}
                        )
                    ]
                ),
                html.Div(
                    children=[dcc.Graph(id="graph-output", style={"height": "650px"})],
                    style={"width": "90%", "margin": "0 auto", "backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0px 2px 10px rgba(0,0,0,0.1)"}
                )
            ]
        )
    ]),

    # ======================================================
    # TAB 2 - DISTRIBUCIÓN DE RETORNOS
    # ======================================================
    dbc.Tab(label="2. Distribución de Retornos", children=[
        html.Div(
            style={"backgroundColor": "#f7f9fc", "padding": "40px", "fontFamily": "Segoe UI"},
            children=[
                html.H1(
                    "Distribución de Retornos por Acción (Últimos 3 años)",
                    style={"textAlign": "center", "color": "#1C2340", "fontWeight": "700", "marginBottom": "40px", "fontSize": "34px"}
                ),
                html.Div(
                    style={"width": "350px", "margin": "0 auto", "backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "boxShadow": "0px 2px 8px rgba(0,0,0,0.10)", "marginBottom": "35px"},
                    children=[
                        html.Label("Selecciona una acción:", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id='dist-dropdown',
                            options=[{'label': t, 'value': t} for t in tickers],
                            value='KO',
                            style={"marginTop": "10px"}
                        )
                    ]
                ),
                html.Div(
                    children=[dcc.Graph(id='dist-graph', style={"height": "650px"})],
                    style={"width": "90%", "margin": "0 auto", "backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "boxShadow": "0px 2px 12px rgba(0,0,0,0.12)"}
                )
            ]
        )
    ]),

    # ======================================================
    # TAB 3 - BOLLINGER CRIPTOS
    # ======================================================
    dbc.Tab(label="3. Bollinger Criptomonedas", children=[
        html.Div([
            html.H1("Gráfica de Bollinger - Criptomonedas", style={"textAlign": "center"}),

            html.P("Selecciona una criptomoneda:", style={"fontWeight": "bold"}),

            dcc.Dropdown(
                id="selected_ticker",
                options=[{"label": t, "value": t} for t in df["ticker"].unique()],
                value=df["ticker"].unique()[0] if len(df["ticker"].unique()) > 0 else None,
                multi=False
            ),

            dcc.Graph(id="bollinger_chart")
        ])
    ]),

    # ======================================================
    # TAB 4 - *** NUEVO LAYOUT ***
    # ======================================================
    dbc.Tab(label="4. Animación Criptomonedas", children=[
        html.Div([
            html.H1("Dashboard - Evolución de Precios de Criptomonedas",
                    style={"textAlign": "center", "marginBottom": 25,
                           "color": "#1f2c56", "fontFamily": "Arial, sans-serif"}),

            html.Div([
                html.Label("Selecciona una o varias criptomonedas:",
                           style={"fontWeight": "bold", "fontSize": "16px",
                                  "color": "#1f2c56"}),

                dcc.Dropdown(
                    id="crypto-dropdown",
                    options=[{"label": name, "value": name}
                             for name in sorted(df_filtered["name"].unique())],
                    value=["Bitcoin"] if "Bitcoin" in df_filtered["name"].unique() else (sorted(df_filtered["name"].unique())[:1] if len(df_filtered)>0 else []),
                    multi=True,
                    placeholder="Selecciona criptomonedas...",
                    style={"width": "80%", "margin": "0 auto"},
                ),
            ],
            style={"textAlign": "center", "marginBottom": "30px"}),

            dcc.Graph(id="animated-crypto-chart", style={"height": "650px"})
        ])
    ])
])

app10.layout = html.Div(
    [
        dbc.Card(
            [
                # tabs ya es un componente dbc.Tabs; no lo anidamos de nuevo.
                dbc.CardHeader(tabs),
                dbc.CardBody(
                    html.Div(id="tab-content", className="p-4")
                ),
            ],
            class_name="shadow-lg",
            style={
                "borderRadius": "15px",
                "padding": "10px",
                "backgroundColor": "white"
            }
        )
    ],
    style={"padding": "20px"}
)

# ============================================================
#               CALLBACKS
# ============================================================

# ---------- TAB 1 ----------
@app10.callback(
    Output("graph-output", "figure"),
    Input("ticker-dropdown", "value"),
    Input("metric-dropdown", "value"),
    Input("slider", "value")
)
def update_graph(ticker_list, metric, slider_range):
    # normalizar input
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]
    if not ticker_list:
        empty = go.Figure()
        empty.add_annotation(text="Selecciona al menos una acción.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty.update_layout(template="plotly_white")
        return empty

    start, end = slider_range
    if end < start:
        start, end = end, start

    # extraer sub-data seguro
    df_plot = data.iloc[start:end + 1].copy()
    if df_plot.empty:
        empty = go.Figure()
        empty.add_annotation(text="No hay datos en el rango seleccionado.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty.update_layout(template="plotly_white")
        return empty

    corporate_colors = ["#00528C", "#1A73E8", "#4C8BF5", "#7FAAFD", "#A6C4FF", "#D4E1FF"]

    hover_template = "<b>%{fullData.name}</b><br>Fecha: %{x}<br>Valor: %{y:.2f}<extra></extra>"

    if metric == "price":
        # PRECAUCIÓN: usar formato long para evitar problemas internos de plotly
        try:
            filtered = df_plot[ticker_list].copy()
        except KeyError:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Ticker(s) no encontrados en los datos.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            empty_fig.update_layout(template="plotly_white")
            return empty_fig

        filtered = filtered.reset_index().rename(columns={"index": "Date"})
        # melt a formato long
        df_long = filtered.melt(id_vars="Date", value_vars=[c for c in filtered.columns if c != "Date"], var_name="Ticker", value_name="Price")
        if df_long["Price"].dropna().empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No hay datos de precio en el rango seleccionado.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            empty_fig.update_layout(template="plotly_white")
            return empty_fig

        fig = px.line(
            df_long,
            x="Date",
            y="Price",
            color="Ticker",
            title="Evolución de Precios de Cierre",
            labels={"Price": "Precio (USD)", "Date": "Fecha"},
            color_discrete_sequence=corporate_colors
        )
    else:
        # retornos mensuales
        monthly_returns = returns[ticker_list].resample("ME").mean() * 100
        start_date = data.index[start]
        end_date = data.index[end]
        monthly_filtered = monthly_returns.loc[start_date:end_date]
        if monthly_filtered.empty or monthly_filtered.dropna(how="all").empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No hay datos mensuales en el rango seleccionado.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            empty_fig.update_layout(template="plotly_white")
            return empty_fig

        monthly_filtered = monthly_filtered.reset_index().rename(columns={"index": "Date"})
        df_long = monthly_filtered.melt(id_vars="Date", value_vars=[c for c in monthly_filtered.columns if c != "Date"], var_name="Ticker", value_name="Retorno")
        fig = px.line(
            df_long,
            x="Date",
            y="Retorno",
            color="Ticker",
            title="Evolución de Retornos Mensuales (%)",
            labels={"Retorno": "Retorno (%)", "Date": "Fecha"},
            color_discrete_sequence=corporate_colors
        )

    fig.update_traces(line=dict(width=3), hovertemplate=hover_template)

    fig.update_layout(
        template="plotly_white",
        title=dict(x=0.5, font=dict(size=28, color="#1C2340", family="Segoe UI", weight=700)),
        font=dict(size=15, family="Segoe UI"),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(title="Acciones", orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1, font=dict(size=14)),
        xaxis=dict(showgrid=True, gridcolor="#e3e6eb", gridwidth=0.7),
        yaxis=dict(showgrid=True, gridcolor="#e3e6eb", gridwidth=0.7)
    )

    return fig

# ---------- TAB 2 ----------
@app10.callback(
    Output('dist-graph', 'figure'),
    Input('dist-dropdown', 'value')
)
def update_distribution(selected_ticker):
    corporate_color = "#1A73E8"
    if selected_ticker not in returns_3y.columns:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Ticker no disponible en retornos.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(template="plotly_white")
        return empty_fig

    fig = px.histogram(
        returns_3y,
        x=selected_ticker,
        nbins=40,
        histnorm='probability density',
        color_discrete_sequence=[corporate_color],
        opacity=0.85,
        title=f"Distribución de Retornos - {selected_ticker} (Últimos 3 años)"
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color="#1C2340")),
        hovertemplate="<b>%{x:.4f}</b><br>Densidad: %{y:.4f}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(x=0.5, font=dict(size=26, color="#1C2340", weight=700)),
        xaxis=dict(title="Retorno", showgrid=True, gridcolor="#e3e6eb", gridwidth=0.7),
        yaxis=dict(title="Densidad", showgrid=True, gridcolor="#e3e6eb", gridwidth=0.7),
        margin=dict(l=60, r=40, t=60, b=50),
        font=dict(family="Segoe UI", size=15)
    )
    return fig

# ---------- TAB 3 ----------

# ---------- TAB 3 ----------
@app10.callback(
    Output("bollinger_chart", "figure"),
    Input("selected_ticker", "value")
)
def update_bollinger(selected_ticker):
    data_sel = df[df["ticker"] == selected_ticker].copy()
    data_sel = data_sel.sort_values("Date")
    window = 20
    data_sel["MA"] = data_sel["Close"].rolling(window).mean()
    data_sel["STD"] = data_sel["Close"].rolling(window).std()
    data_sel["Upper"] = data_sel["MA"] + (2 * data_sel["STD"])
    data_sel["Lower"] = data_sel["MA"] - (2 * data_sel["STD"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_sel["Date"], y=data_sel["Close"], mode="lines", name="Precio de Cierre", line=dict(color="royalblue", width=1.5)))
    fig.add_trace(go.Scatter(x=data_sel["Date"], y=data_sel["MA"], mode="lines", name=f"Media Móvil ({window})", line=dict(color="black", width=1.5)))
    fig.add_trace(go.Scatter(x=data_sel["Date"], y=data_sel["Upper"], mode="lines", name="Banda Superior (2σ)", line=dict(color="gray", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=data_sel["Date"], y=data_sel["Lower"], mode="lines", name="Banda Inferior (2σ)", line=dict(color="gray", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(128,128,128,0.3)"))
    
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"{data_sel['name'].iloc[0]}", x=0.5, xanchor='center', font=dict(size=22, family="Segoe UI", color="#1C2340", weight="bold")),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)"),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig



# ---------- TAB 4 ----------
@app10.callback(
    Output("animated-crypto-chart", "figure"),
    Input("crypto-dropdown", "value")
)
def update_animation(selected_cryptos):
    return create_animated_figure(selected_cryptos)

# ============================================================
#               RUN
# ============================================================
if __name__ == "__main__":
    app10.run(debug=False, host="0.0.0.0",port=10000)
