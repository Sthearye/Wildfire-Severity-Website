from dash import html, dcc, register_page
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

register_page(__name__, path="/objectives")

COLORS = {
    "bg": "#FFF8F0",
    "card_bg": "#FFFFFF",
    "header": "linear-gradient(135deg, #9F5244 0%, #D05F33 100%)",
    "text": "#16060C",
    "accent": "#D05F33",
    "secondary": "#E88D67",
    "light_accent": "#FFE8D6",
}

ICONS = {
    "model": "carbon:model-alt",
    "explore": "carbon:chart-relationship",
    "ml": "carbon:machine-learning-model",
    "socio": "carbon:user-data",
    "insights": "carbon:data-vis-4"
}

layout = dbc.Container([
    html.Div([
        html.Div([
            DashIconify(icon="carbon:fire", width=40, height=40, color="white", style={"marginRight": "15px"}),
            html.H2("Project Objectives", className="mb-0 d-inline")
        ], className="d-flex align-items-center"),
    ], className="text-white p-4 rounded-top", style={"background": COLORS["header"]}),

    dbc.Card([
        dbc.CardBody([
            html.Ul([
                html.Li([
                    html.Div([
                        DashIconify(icon=ICONS["model"], width=28, height=28, color=COLORS["accent"], style={"marginRight": "15px"}),
                        html.Span("Develop predictive models to estimate wildfire severity in California using historical fire data and environmental indicators.")
                    ], className="d-flex align-items-start")
                ], className="mb-4", style={
                    "color": COLORS["text"], "borderLeft": f"4px solid {COLORS['accent']}",
                    "padding": "12px 15px", "backgroundColor": COLORS["light_accent"],
                    "borderRadius": "0 8px 8px 0", "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
                }),

                html.Li([
                    html.Div([
                        DashIconify(icon=ICONS["explore"], width=28, height=28, color=COLORS["accent"], style={"marginRight": "15px"}),
                        html.Span("Explore correlations between wildfire severity and factors such as temperature, wind speed, soil moisture, precipitation, humidity, and seasonality.")
                    ], className="d-flex align-items-start")
                ], className="mb-4", style={
                    "color": COLORS["text"], "borderLeft": f"4px solid {COLORS['accent']}",
                    "padding": "12px 15px", "backgroundColor": COLORS["light_accent"],
                    "borderRadius": "0 8px 8px 0", "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
                }),

                html.Li([
                    html.Div([
                        DashIconify(icon=ICONS["ml"], width=28, height=28, color=COLORS["accent"], style={"marginRight": "15px"}),
                        html.Span("Apply machine learning techniques—such as Random Forests and Linear Regression—to predict the logarithmic scale of acres burned during wildfires.")
                    ], className="d-flex align-items-start")
                ], className="mb-4", style={
                    "color": COLORS["text"], "borderLeft": f"4px solid {COLORS['accent']}",
                    "padding": "12px 15px", "backgroundColor": COLORS["light_accent"],
                    "borderRadius": "0 8px 8px 0", "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
                }),

                html.Li([
                    html.Div([
                        DashIconify(icon=ICONS["insights"], width=28, height=28, color=COLORS["accent"], style={"marginRight": "15px"}),
                        html.Span("Provide data-driven insights to support fire prevention, preparedness, and equitable resource allocation for emergency response.")
                    ], className="d-flex align-items-start")
                ], className="mb-4", style={
                    "color": COLORS["text"], "borderLeft": f"4px solid {COLORS['accent']}",
                    "padding": "12px 15px", "backgroundColor": COLORS["light_accent"],
                    "borderRadius": "0 8px 8px 0", "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
                }),
            ], className="list-unstyled"),

            html.Hr(className="my-4", style={"borderTop": f"2px solid {COLORS['secondary']}"}),

            html.Div([
                DashIconify(icon="carbon:document", width=24, height=24, color=COLORS["accent"]),
                html.H5("Project Summary", className="ml-2 mb-0")
            ], className="d-flex align-items-center mb-3"),

            html.Div([
                html.P(
                    "This project aims to uncover the key environmental and socio-economic drivers of wildfire severity across California. "
                    "By using machine learning and statistical analysis, we will assess how climate variables—like drought, temperature, humidity, and wind—interact with social factors, such as income level and infrastructure access, to influence fire outcomes. "
                    "The ultimate objective is to build interpretable models that not only forecast wildfire damage but also inform policy makers, fire management agencies, and local communities about where resources and interventions are most needed.",
                    className="mb-0",
                    style={"color": COLORS["text"], "lineHeight": "1.6", "fontSize": "1.05rem"}
                )
            ], className="p-4 rounded", style={
                "backgroundColor": COLORS["light_accent"],
                "border": f"1px solid {COLORS['secondary']}"
            }),

            html.Hr(className="my-4", style={"borderTop": f"2px solid {COLORS['secondary']}"}),

            html.Div([
                DashIconify(icon="carbon:idea", width=24, height=24, color=COLORS["accent"]),
                html.H5("Project Hypothesis", className="ml-2 mb-0")
            ], className="d-flex align-items-center mb-3"),

            html.Div([
                html.P("Based on environmental and demographic variables in the dataset, we propose the following hypotheses regarding factors that influence wildfire severity across California:",
                       style={"color": COLORS["text"], "lineHeight": "1.6", "fontSize": "1.05rem"}),
                html.Ul([
                    html.Li("Higher temperatures combined with lower humidity and vapor pressure increase wildfire severity."),
                    html.Li("Higher average wind speed and wind run are positively correlated with wildfire spread."),
                    html.Li("Severe drought classifications (D2–D4) result in larger wildfires compared to minimal or no drought."),
                    html.Li("Urbanized counties with high population density may show lower severity due to better infrastructure and response.")
                ], style={"paddingLeft": "20px", "lineHeight": "1.6", "fontSize": "1.01rem", "color": COLORS["text"]})
            ], className="p-4 rounded", style={
                "backgroundColor": COLORS["light_accent"],
                "border": f"1px solid {COLORS['secondary']}"
            })
        ])
    ], className="shadow-lg border-0"),

    html.Div([
        html.Small("California Wildfire Severity Prediction Project", className="text-muted")
    ], className="text-center mt-4 mb-2")

], fluid=True, className="py-4", style={"background": COLORS["bg"], "minHeight": "100vh"})
