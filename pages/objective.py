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
                html.Li("Analyze key patterns in the dataset",
                       style={"color": COLORS["text"], 
                              "borderLeft": f"4px solid {COLORS['accent']}",
                              "padding": "8px",
                              "marginBottom": "8px"}),
                html.Li("Identify significant correlations",
                       style={"color": COLORS["text"],
                              "borderLeft": f"4px solid {COLORS['accent']}",
                              "padding": "8px",
                              "marginBottom": "8px"}),
                # More list items...
            ], className="list-unstyled"),
            
            html.Hr(style={"borderTop": f"2px solid {COLORS['accent']}"}),
            
            html.P("This research aims to provide actionable insights through rigorous data analysis.", 
                  className="lead",
                  style={"color": COLORS["accent"]})
        ], style={"background": COLORS["card"]})
    ], className="mt-4 shadow-lg", style={"border": f"1px solid {COLORS['accent']}"})
], style={"background": COLORS["bg"], "minHeight": "100vh"})