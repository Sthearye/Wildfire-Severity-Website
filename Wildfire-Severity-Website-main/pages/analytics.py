from dash import html, dcc, register_page, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd

from models.random_forest_classifier_model import run_rf_classifier_model
from models.stacking_regressor_model import run_stacking_regressor_model
from models.statistical_analysis import generate_acf_pacf_plot, generate_scatter_plots
from models.decomposition import run_time_series_decomposition  # <-- Make sure this accepts (county, feature)

register_page(__name__, path="/analytics")

COLORS = {
    "bg": "#16060C05",
    "card": "#FFFFFF",
    "header": "#D05F33",
    "text": "#16060C",
    "highlight": "#FF7621"
}

TEXT_STYLE = {"fontSize": "15px", "color": COLORS["text"]}
H5_STYLE = {"fontSize": "18px", "color": COLORS["text"], "marginBottom": "10px"}
H6_STYLE = {"fontSize": "16px", "color": COLORS["text"]}

# Load dataset once for dropdowns
df = pd.read_csv("datasets/cimis_merged.csv")
df["Date"] = pd.to_datetime(df["Date"])
county_options = sorted(df["County"].dropna().unique())
feature_options = sorted([
    'ETo (in)', 'Precip (in)', 'Sol Rad (Ly/day)', 'Avg Vap Pres (mBars)',
    'Max Air Temp (F)', 'Min Air Temp (F)', 'Max Rel Hum (%)',
    'Avg Wind Speed (mph)', 'Wind Run (miles)', 'Avg Soil Temp (F)'
])

layout = dbc.Container([
    dbc.ButtonGroup([
        dbc.Button("Analytical Methods", id="btn-analytics", n_clicks=0, color="primary", outline=True),
        dbc.Button("Model Results", id="btn-model-results", n_clicks=0, color="primary", outline=True),
    ], className="mt-4"),

    dbc.Card([
        dbc.CardHeader("Data Processing Pipeline", className="h4", style={"background": COLORS["header"], "color": "white"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("📅 Data Collection", style=H5_STYLE),
                    html.P(
                        "Data is collected from CIMIS weather stations and CAL FIRE incident databases. "
                        "The CIMIS dataset includes temperature, humidity, wind speed, solar radiation, and precipitation. "
                        "Wildfire incident records provide spatial and temporal burn area data. "
                        "Through the US Census, we were able to gather the population of each county from 2014–2024.",
                        style=TEXT_STYLE
                    )
                ], md=4),
                dbc.Col([
                    html.H5("🛠️ Feature Engineering", style=H5_STYLE),
                    html.P(
                        "Additional features such as 'FireSeason' (May–October), and temporal components like year, month, and day "
                        "are derived to enrich the dataset. We also added lag features for weather variables up to 7 days.",
                        style=TEXT_STYLE
                    )
                ], md=4),
                dbc.Col([
                    html.H5("🪟 Data Preprocessing", style=H5_STYLE),
                    html.P(
                        "The dataset undergoes cleaning by handling missing values, "
                        "dropping irrelevant columns, and applying log transformations to normalize skewed features.",
                        style=TEXT_STYLE
                    )
                ], md=4)
            ])
        ], style={"background": COLORS["card"]})
    ], className="shadow my-4"),

    html.Div(id="analytics-content", children=[
        dbc.Card([
            dbc.CardHeader("Analytical Methods", className="h4", style={"background": COLORS["header"], "color": "white"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("📉 Scatter Plot with Wildfire Size", style={**H6_STYLE, "marginTop": "20px"}),
                        dcc.Dropdown(
                            id="scatter-feature-dropdown",
                            options=[{"label": f, "value": f} for f in [
                                'Max Air Temp (F)', 'Min Air Temp (F)', 'Avg Wind Speed (mph)', 'Avg Soil Temp (F)', 'Weather Intensity'
                            ]],
                            value="Max Air Temp (F)",
                            placeholder="Select a feature",
                            style={"marginBottom": "15px"}
                        ),
                        html.Div(id="scatter-plot-container"),

                        html.H6("📈 ACF/PACF Explorer", style={**H6_STYLE, "marginTop": "40px"}),
                        dcc.Dropdown(id="acf-county-dropdown", options=[{"label": c, "value": c} for c in county_options], value=county_options[0], placeholder="Select County", style={"marginBottom": "10px"}),
                        dcc.Dropdown(id="acf-type-dropdown", options=[{"label": "ACF", "value": "ACF"}, {"label": "PACF", "value": "PACF"}], value="ACF", style={"marginBottom": "10px"}),
                        dcc.Dropdown(id="acf-feature-dropdown", options=[{"label": f, "value": f} for f in feature_options], value="Max Air Temp (F)", style={"marginBottom": "10px"}),
                        html.Div(id="acf-pacf-plot-container", style={"marginTop": "20px"}),
                    ])
                ])
            ], style={"background": COLORS["card"]})
        ], className="shadow")
    ]),

    html.Div(id="model-content", style={"display": "none"}, children=[
        dbc.Card([
            dbc.CardHeader("Model Results", className="h4", style={"background": COLORS["header"], "color": "white"}),
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab(label="Random Forest Classifier", children=run_rf_classifier_model()),
                    dbc.Tab(label="Stacking Regressor", children=run_stacking_regressor_model()),
                    dbc.Tab(label="Decomposition", children=[
                        html.Div([
                            html.H6("🔎 Decomposition Viewer", style=H6_STYLE),
                            dcc.Dropdown(
                                id="tab-decomp-county",
                                options=[{"label": c, "value": c} for c in county_options],
                                value=county_options[0],
                                placeholder="Select County",
                                style={"marginBottom": "10px"}
                            ),
                            dcc.Dropdown(
                                id="tab-decomp-feature",
                                options=[{"label": f, "value": f} for f in feature_options],
                                value=feature_options[0],
                                placeholder="Select Feature",
                                style={"marginBottom": "20px"}
                            ),
                            html.Div(id="tab-decomp-output", style={"marginTop": "20px"})
                        ])
                    ])
                ])
            ], style={"background": COLORS["card"]})
        ], className="shadow mt-4")
    ])
], style={"background": COLORS["bg"], "minHeight": "100vh"})


# === CALLBACKS ===

@callback(
    Output("scatter-plot-container", "children"),
    Input("scatter-feature-dropdown", "value")
)
def update_scatter_plot(feature):
    if not feature:
        return html.P("Please select a feature to view the scatter plot.", style=TEXT_STYLE)
    df = pd.read_csv("datasets/new_merged_df.csv")
    visuals = generate_scatter_plots(df, features=[feature])
    if not visuals:
        return html.P("No data available for this feature.", style=TEXT_STYLE)
    label, img = visuals[0]
    return html.Div([
        html.H6(f"{label}", style={**H6_STYLE, "marginTop": "15px"}),
        html.Img(src=f"data:image/png;base64,{img}", style={"width": "100%", "borderRadius": "5px"})
    ])

@callback(
    Output("acf-pacf-plot-container", "children"),
    Input("acf-county-dropdown", "value"),
    Input("acf-type-dropdown", "value"),
    Input("acf-feature-dropdown", "value")
)
def update_acf_single_plot(county, plot_type, feature):
    if not county or not plot_type or not feature:
        return html.P("Please select all fields.", style=TEXT_STYLE)
    img_str, error = generate_acf_pacf_plot(county, plot_type, feature)
    if error:
        return html.P(error, style=TEXT_STYLE)
    return html.Div([
        html.H6(f"{plot_type} - {feature} in {county}", style=H6_STYLE),
        html.Img(src=f"data:image/png;base64,{img_str}", style={"width": "100%", "borderRadius": "5px"})
    ])

@callback(
    Output("tab-decomp-output", "children"),
    Input("tab-decomp-county", "value"),
    Input("tab-decomp-feature", "value")
)
def update_decomposition_output(county, feature):
    if not county or not feature:
        return html.P("Please select both a county and a feature.", style=TEXT_STYLE)
    return run_time_series_decomposition(county, feature)

@callback(
    Output("analytics-content", "style"),
    Output("model-content", "style"),
    Input("btn-analytics", "n_clicks"),
    Input("btn-model-results", "n_clicks")
)
def toggle_views(n_analytics, n_model):
    if n_model > n_analytics:
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}
