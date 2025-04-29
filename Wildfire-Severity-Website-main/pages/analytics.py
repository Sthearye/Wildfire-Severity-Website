from dash import html, dcc, register_page, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd

# Importing model functions
from models.linear_model import run_linear_model
from models.xgboost_model import run_xgboost_model
from models.random_forest_classifier_model import run_rf_classifier_model
from models.stacking_regressor_model import run_stacking_regressor_model

from models.statistical_analysis import (
    get_qqplot_img,
    generate_acf_pacf_plot,
    generate_scatter_plots,
    linear_regression_with_diagnostics
)

register_page(__name__, path="/analytics")

COLORS = {
    "bg": "#16060C05",
    "card": "#FFFFFF",
    "header": "#D05F33",
    "text": "#16060C",
    "highlight": "#FF7621"
}

county_options = sorted(pd.read_csv("datasets/cimis_merged.csv")["County"].unique())

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
                    html.H5("📅 Data Collection", style={"color": COLORS["text"], "marginBottom": "10px"}),
                    html.P(
                        "Data is collected from CIMIS weather stations and CAL FIRE incident databases. "
                        "The CIMIS dataset includes temperature, humidity, wind speed, solar radiation, and precipitation. "
                        "Wildfire incident records provide spatial and temporal burn area data."
                        "Through the US Census, we were able to gather the population of each county from 2014-2024",
                        style={"fontSize": "15px", "color": COLORS["text"]}
                    )
                ], md=4),

                dbc.Col([
                    html.H5("🛠️ Feature Engineering", style={"color": COLORS["text"], "marginBottom": "10px"}),
                    html.P(
                        "Additional features such as 'FireSeason' which we assigned as in between May through October "
                        "and temporal components like year, month, and day are derived to enrich the modeling dataset."
                        "On top of that, we added many different lag features for all the weather for up to 7",
                        style={"fontSize": "15px", "color": COLORS["text"]}
                    )
                ], md=4),

                dbc.Col([
                    html.H5("🪟 Data Preprocessing", style={"color": COLORS["text"], "marginBottom": "10px"}),
                    html.P(
                        "The dataset undergoes cleaning by handling missing values, "
                        "dropping irrelevant columns, and applying log transformations to normalize skewed features.",
                        style={"fontSize": "15px", "color": COLORS["text"]}
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
                        html.Div([
                            html.H6("📊 Q-Q Plot of Residuals", style={"marginTop": "20px", "color": COLORS["text"]}),
                            html.P("This plot compares the distribution of residuals to a normal distribution. If the residuals are normally distributed, the points will fall approximately along the red line.", style={"fontSize": "13px", "color": COLORS["text"]}),
                            html.Img(id="qqplot-img", style={"width": "100%", "borderRadius": "5px"}),

                            html.H6("📈 ACF/PACF Explorer", style={"marginTop": "25px", "color": COLORS["text"]}),
                            html.P("Explore autocorrelation in weather data using ACF (overall correlation) and PACF (direct correlation) across lags.", style={"fontSize": "13px", "color": COLORS["text"]}),
                            dcc.Dropdown(id="acf-county-dropdown", options=[{"label": c, "value": c} for c in county_options], value=county_options[0], placeholder="Select County", style={"marginBottom": "10px"}),
                            dcc.Dropdown(id="acf-type-dropdown", options=[{"label": "ACF", "value": "ACF"}, {"label": "PACF", "value": "PACF"}], value="ACF", style={"marginBottom": "10px"}),
                            dcc.Dropdown(id="acf-feature-dropdown", options=[{"label": f, "value": f} for f in [
                                'ETo (in)', 'Precip (in)', 'Sol Rad (Ly/day)', 'Avg Vap Pres (mBars)',
                                'Max Air Temp (F)', 'Min Air Temp (F)', 'Max Rel Hum (%)',
                                'Avg Wind Speed (mph)', 'Wind Run (miles)', 'Avg Soil Temp (F)'
                            ]], value="Max Air Temp (F)", style={"marginBottom": "10px"}),
                            html.Div(id="acf-pacf-plot-container", style={"marginTop": "20px"}),

                            html.H6("📉 Select Weather Feature for Scatter Plot:", style={"marginTop": "20px", "color": COLORS["text"]}),
                            html.P("Visualize the relationship between selected weather features and wildfire size using a scatter plot.", style={"fontSize": "13px", "color": COLORS["text"]}),
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

                            html.H6("🧪 Partial Regression & Rainbow Test", style={"marginTop": "30px", "color": COLORS["text"]}),
                            html.P("Partial regression plots the effect of one variable while holding others constant. Rainbow test checks linearity assumptions statistically.", style={"fontSize": "13px", "color": COLORS["text"]}),
                            dcc.Dropdown(
                                id="regression-feature-dropdown",
                                options=[{"label": f, "value": f} for f in [
                                    'Max Air Temp (F)', 'Min Air Temp (F)', 'Avg Wind Speed (mph)', 'Avg Soil Temp (F)', 'Weather Intensity'
                                ]],
                                value=['Max Air Temp (F)', 'Avg Wind Speed (mph)', 'Avg Soil Temp (F)'],
                                multi=True,
                                placeholder="Select features for regression",
                                style={"marginBottom": "10px"}
                            ),
                            html.Img(id="partial-reg-img", style={"width": "100%", "borderRadius": "5px"}),
                            html.P(id="rainbow-test-output", style={"marginTop": "10px"}),
                        ])
                    ], md=12)
                ])
            ], style={"background": COLORS["card"]})
        ], className="shadow")
    ]),

    html.Div(id="model-content", style={"display": "none"}, children=[
        dbc.Card([
            dbc.CardHeader("Model Results", className="h4", style={"background": COLORS["header"], "color": "white"}),
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab(label="Linear Regression", children=run_linear_model()),
                    dbc.Tab(label="XGBoost", children=run_xgboost_model()),
                    dbc.Tab(label="Random Forest Classifier", children=run_rf_classifier_model()),
                    dbc.Tab(label="Stacking Regressor", children=run_stacking_regressor_model()),
                ])
            ], style={"background": COLORS["card"]})
        ], className="shadow mt-4")
    ])
], style={"background": COLORS["bg"], "minHeight": "100vh"})

# --- CALLBACKS ---

@callback(
    Output("scatter-plot-container", "children"),
    Input("scatter-feature-dropdown", "value")
)
def update_scatter_plot(feature):
    if not feature:
        return html.P("Please select a feature to view the scatter plot.")
    df = pd.read_csv("datasets/new_merged_df.csv")
    visuals = generate_scatter_plots(df, features=[feature])
    if not visuals:
        return html.P("No data available for this feature.")
    label, img = visuals[0]
    return html.Div([
        html.H6(f"{label}", style={"marginTop": "15px"}),
        html.Img(src=f"data:image/png;base64,{img}", style={"width": "100%", "borderRadius": "5px"})
    ])

@callback(
    Output("partial-reg-img", "src"),
    Output("rainbow-test-output", "children"),
    Output("qqplot-img", "src"),
    Input("regression-feature-dropdown", "value")
)
def update_statistical_analysis(selected_features):
    df = pd.read_csv("datasets/new_merged_df.csv")
    diag = linear_regression_with_diagnostics(df, selected_features)
    rainbow_msg = f"Rainbow Test p-value: {diag['rainbow_p_value']:.4f} → " + (
        "✅ No evidence against linearity." if diag['rainbow_p_value'] > 0.05 else "⚠️ Linearity assumption may be violated."
    )
    partial_img = f"data:image/png;base64,{diag['partial_regression_img']}"
    qq_img = f"data:image/png;base64,{get_qqplot_img()}"
    return partial_img, rainbow_msg, qq_img

@callback(
    Output("acf-pacf-plot-container", "children"),
    Input("acf-county-dropdown", "value"),
    Input("acf-type-dropdown", "value"),
    Input("acf-feature-dropdown", "value")
)
def update_acf_single_plot(county, plot_type, feature):
    if not county or not plot_type or not feature:
        return html.P("Please select all fields.")
    img_str, error = generate_acf_pacf_plot(county, plot_type, feature)
    if error:
        return html.P(error)
    return html.Div([
        html.H6(f"{plot_type} - {feature} in {county}"),
        html.Img(src=f"data:image/png;base64,{img_str}", style={"width": "100%", "borderRadius": "5px"})
    ])

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