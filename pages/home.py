from dash import html, dcc, register_page, callback, Output, Input
import dash_bootstrap_components as dbc

register_page(__name__, path="/") 
# Warm color palette
colors = {
    "background": "#fff9f2",  # Cream
    "card": "#ffffff",        # White
    "primary": "#ff7e5f",     # Coral
    "secondary": "#feb47b",   # Light orange
    "text": "#5a3e36"        # Brown
}


layout = dbc.Container([
    html.H1("Home Page", className="display-4 text-center my-4", style={"color": colors["primary"]}),
    
    dbc.Card([
        dbc.CardBody([
            html.P("Welcome to our colorful dashboard!", 
                  className="lead",
                  style={"color": colors["text"]}),
            dbc.Button("Click Me", id="home-btn", 
                       color="primary", 
                       className="mt-3",
                       style={"backgroundColor": colors["primary"]}),
            html.Div(id="home-output", className="mt-3")
        ])
    ], className="shadow", style={"backgroundColor": colors["card"]}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature 1", style={"backgroundColor": colors["secondary"], "color": "white"}),
                dbc.CardBody([
                    html.P("Enjoy the warm color scheme!", style={"color": colors["text"]}),
                ])
            ], className="mt-4 shadow-sm")
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature 2", style={"backgroundColor": colors["secondary"], "color": "white"}),
                dbc.CardBody([
                    html.P("All colors are carefully coordinated.", style={"color": colors["text"]}),
                ])
            ], className="mt-4 shadow-sm")
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature 3", style={"backgroundColor": colors["secondary"], "color": "white"}),
                dbc.CardBody([
                    html.P("Hover effects would look great here!", style={"color": colors["text"]}),
                ])
            ], className="mt-4 shadow-sm")
        ], md=4),
    ], className="mt-4")
], style={"backgroundColor": colors["background"], "padding": "2rem"})

@callback(
    Output("home-output", "children"),
    Input("home-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_output(n_clicks):
    return dbc.Alert(
        f"Button clicked {n_clicks} times!",
        color="primary",
        style={"backgroundColor": colors["primary"], "color": "white"}
    )