from dash import html, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from visualizations.wildfire_trends import create_county_trends_plot
from visualizations.ca_choropleth import create_wildfire_choropleth
from visualizations.acres_by_county import create_acres_bar_chart
from visualizations.correlation_heatmap import create_correlation_heatmap

register_page(__name__, path="/findings")

COLORS = {
    "header": "#D05F33",
    "highlight": "#FF7621",
    "card_border": "#FF7621"
}

layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader(
            "Wildfire Findings Dashboard",
            className="h4",
            style={"background": COLORS["header"], "color": "white"}
        ),
        
        dbc.CardBody([
            # Geographic visualization
            dbc.Row([
                dbc.Col([
                    html.H3("Geographic Distribution", style={"color": COLORS["highlight"]}),
                    create_wildfire_choropleth()
                ])
            ], className="mb-4"),

            # Acres burned visualization
            dbc.Row([
                dbc.Col([
                    html.H3("Acres Burned by County", style={"color": COLORS["highlight"]}),
                    create_acres_bar_chart()
                ])
            ], className="mb-4"),

            # Correlation heatmap
            dbc.Row([
                dbc.Col([
                    html.H3("Feature Correlations", style={"color": COLORS["highlight"]}),
                    create_correlation_heatmap(),
                    html.Div([
                        html.P("Key observations:", className="mt-2"),
                        html.Ul([
                            html.Li("Temperature shows strong positive correlation with fire acreage"),
                            html.Li("Humidity has negative correlation with fire severity"),
                            html.Li("Wind speed shows moderate positive correlation")
                        ])
                    ], className="p-3 bg-light rounded mt-2")
                ])
            ], className="mb-4"),

            # Temporal trends
            dbc.Row([
                dbc.Col([
                    html.H3("Monthly Trends", style={"color": COLORS["highlight"]}),
                    create_county_trends_plot()
                ])
            ])
        ], style={"padding": "20px"})
    ], className="mt-4 shadow", style={"borderTop": f"3px solid {COLORS['card_border']}"})
], fluid=True)