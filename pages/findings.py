from dash import html, register_page, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from visualizations.wildfire_trends import create_county_trends_plot
from visualizations.ca_choropleth import create_wildfire_choropleth
from visualizations.acres_by_county import create_acres_bar_chart
from visualizations.correlation_heatmap import create_correlation_heatmap
from visualizations.california_wildfire_dist import create_ca_wildfire_map
from visualizations.eto_scatter import eto_scatter_plot
from visualizations.climate_parallel import create_climate_parallel_plot 
from visualizations.monthly_patterns import MonthlyPatternsPlot
from visualizations.drought_fire_scatter import drought_fire_scatter_plot

# Load your data
df = pd.read_csv('datasets/new_merged_df.csv')

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
            # County-level choropleth
            dbc.Row([
                dbc.Col([
                    html.H3("County-Level Choropleth", style={"color": COLORS["highlight"]}),
                    create_wildfire_choropleth()
                ])
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.H3("California Wildfire Distribution", style={"color": COLORS["highlight"]}),
                    html.P("Interactive map showing wildfire locations and severity across California", 
                          className="text-muted mb-3"),
                    dcc.Graph(
                        figure=create_ca_wildfire_map(df),
                        id='ca-wildfire-map',
                        style={'height': '800px'}
                    )
                ], width=12, className="mb-4")
            ]),

            # Temporal trends
            dbc.Row([
                dbc.Col([
                    html.H3("Monthly Trends", style={"color": COLORS["highlight"]}),
                    create_county_trends_plot()
                ])
            ]),

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

            dbc.Row([
                dbc.Col([
                    html.H3("Monthly Fire Patterns", style={"color": COLORS["highlight"]}),
                    dcc.Graph(
                        figure=MonthlyPatternsPlot(df).get_figure(),
                        id='monthly-patterns-plot',
                        style={'height': '500px'}
                    )
                ], width=12, className="mb-4")
            ]),

            # Environmental factors section
            dbc.Row([
                dbc.Col([
                    html.H3("Environmental Factors Analysis", style={"color": COLORS["highlight"]}),
                    html.P("Relationship between evapotranspiration, temperature and fire size", 
                          className="text-muted mb-3"),
                    dcc.Graph(
                        figure=eto_scatter_plot(df),
                        id='eto-scatter-plot',
                        style={'height': '600px'}
                    ),
                    html.Div([
                        html.P("Key insights:", className="mt-3"),
                        html.Ul([
                            html.Li("Higher evapotranspiration correlates with larger fires"),
                            html.Li("Warmer temperatures (orange/red dots) tend to have more severe fires"),
                            html.Li("Larger bubbles indicate more populated counties affected")
                        ])
                    ], className="p-3 bg-light rounded mt-2")
                ], width=12, className="mb-4")
            ]),

            # Drought analysis section
            dbc.Row([
                dbc.Col([
                    html.H3("Drought Impact Analysis", style={"color": COLORS["highlight"]}),
                    html.P("Relationship between drought severity (D4) and wildfire size over time", 
                          className="text-muted mb-3"),
                    dcc.Graph(
                        figure=drought_fire_scatter_plot(df),
                        id='drought-scatter-plot',
                        style={'height': '600px'}
                    ),
                    html.Div([
                        html.P("Key insights:", className="mt-3"),
                        html.Ul([
                            html.Li("Darker colors indicate more severe drought conditions (D4)"),
                            html.Li("Larger bubbles represent larger fires (acres burned)"),
                            html.Li("Hover for county-specific details and exact drought values"),
                            html.Li("Note the logarithmic scale on the y-axis for better visualization")
                        ])
                    ], className="p-3 bg-light rounded mt-2")
                ], width=12, className="mb-4")
            ]),            
            
            # Climate patterns section
            dbc.Row([
                dbc.Col([
                    html.H3("Climate Patterns Analysis", style={"color": COLORS["highlight"]}),
                    html.P("Parallel coordinates plot showing relationships between climate variables and fire severity",
                          className="text-muted mb-3"),
                    dcc.Graph(
                        figure=create_climate_parallel_plot(df),
                        id='climate-parallel-plot',
                        style={'height': '600px'}
                    ),
                    html.Div([
                        html.P("Key observations:", className="mt-3"),
                        html.Ul([
                            html.Li("Lines colored by fire severity (darker = more acres burned)"),
                            html.Li("Examine relationships between multiple climate variables simultaneously"),
                            html.Li("Hover over lines to see exact values for each variable")
                        ])
                    ], className="p-3 bg-light rounded mt-2")
                ], width=12, className="mb-4")
            ])

        ], style={"padding": "20px"})
    ], className="mt-4 shadow", style={"borderTop": f"3px solid {COLORS['card_border']}"})
], fluid=True)