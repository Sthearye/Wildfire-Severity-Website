from dash import html, register_page
import dash_bootstrap_components as dbc

register_page(__name__, path="/objectives")

COLORS = {
    "bg": "#16060C10",  # Dark with 10% opacity
    "card": f"linear-gradient(135deg, #FFFFFF 0%, #F8F8F8 100%)",
    "header": "#9F5244",
    "text": "#16060C",
    "accent": "#D05F33"
}

layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader("Project Objectives", 
                      className="h4",
                      style={"background": COLORS["header"], 
                             "color": "white"}),

        dbc.CardBody([

            html.Ul([
                html.Li("Analyze key patterns and trends within wildfire severity data.",
                        style={"color": COLORS["text"], 
                               "borderLeft": f"4px solid {COLORS['accent']}",
                               "padding": "8px",
                               "marginBottom": "8px"}),

                html.Li("Identify significant correlations between environmental factors and wildfire spread.",
                        style={"color": COLORS["text"],
                               "borderLeft": f"4px solid {COLORS['accent']}",
                               "padding": "8px",
                               "marginBottom": "8px"}),

                html.Li("Apply machine learning models to predict wildfire severity based on climate and population data.",
                        style={"color": COLORS["text"],
                               "borderLeft": f"4px solid {COLORS['accent']}",
                               "padding": "8px",
                               "marginBottom": "8px"}),

                html.Li("Determine the influence of weather patterns, drought conditions, and temperature fluctuations.",
                        style={"color": COLORS["text"],
                               "borderLeft": f"4px solid {COLORS['accent']}",
                               "padding": "8px",
                               "marginBottom": "8px"}),

                html.Li("Provide actionable insights for wildfire prevention, preparedness, and resource allocation.",
                        style={"color": COLORS["text"],
                               "borderLeft": f"4px solid {COLORS['accent']}",
                               "padding": "8px",
                               "marginBottom": "8px"}),

            ], className="list-unstyled"),

            html.Hr(style={"borderTop": f"2px solid {COLORS['accent']}"}),

            html.P(
                "The objective of this project is to leverage environmental, climatic, and demographic data "
                "to analyze the key drivers of wildfire severity across California counties. Through statistical "
                "analysis and machine learning models, this project aims to identify significant patterns and "
                "correlations between wildfire size and various factors such as temperature, humidity, wind speed, "
                "drought levels, and population density. The ultimate goal is to provide actionable insights "
                "that can assist in risk assessment, resource allocation, and wildfire prevention strategies.",
                className="lead",
                style={"color": COLORS["accent"]}
            )

        ], style={"background": COLORS["card"]})

    ], className="mt-4 shadow-lg", style={"border": f"1px solid {COLORS['accent']}"}),

], style={"background": COLORS["bg"], "minHeight": "100vh"})
