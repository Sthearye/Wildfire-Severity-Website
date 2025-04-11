from dash import Dash, html, dcc, page_container, callback, Output, Input
import dash_bootstrap_components as dbc
import time
# Load flame animation HTML
with open("flame_loader_cleaned.html", "r", encoding="utf-8") as f:
    flame_html = f.read()

# Color palette
COLOR_PALETTE = {
    "dark": "#16060C",
    "primary": "#D05F33",
    "secondary": "#9F5244",
    "accent1": "#FF7621",
    "accent2": "#FF9818",
    "accent3": "#FEB504",
    "bg_opacity": "#16060C10"
}

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder="assets"
)

server = app.server

# Flame loader as iframe
flame_loader = html.Div(
    id="loader",
    children=[
        html.Iframe(
            srcDoc=flame_html,
            style={"width": "100%", "height": "100vh", "border": "none"},
            id="flame-iframe"
        )
    ],
    style={"zIndex": "9999", "position": "fixed", "top": 0, "left": 0, "width": "100%", "height": "100vh"}
)

# Navbar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Research Dashboard", href="/", className="ms-2", style={"color": COLOR_PALETTE["accent3"]}),
        dbc.Nav([
            dbc.NavLink("Home", href="/", id="home-link", style={"color": COLOR_PALETTE["accent3"]}),
            dbc.NavLink("Objectives", href="/objectives", id="objectives-link", style={"color": COLOR_PALETTE["accent3"]}),
            dbc.NavLink("Methods", href="/analytics", id="analytics-link", style={"color": COLOR_PALETTE["accent3"]}),
            dbc.NavLink("Findings", href="/findings", id="findings-link", style={"color": COLOR_PALETTE["accent3"]})
        ], navbar=True)
    ]),
    color="dark",
    dark=True,
    sticky="top",
    className="mb-4",
    style={
        "background": f"linear-gradient(90deg, {COLOR_PALETTE['dark']} 0%, {COLOR_PALETTE['secondary']} 100%)",
        "boxShadow": f"0 4px 20px 0 {COLOR_PALETTE['primary']}33"
    }
)

# App layout
app.layout = html.Div([
    dcc.Location(id='url'),
    flame_loader,
    html.Div(
        id="page-content",
        children=[
            navbar,
            dbc.Container(page_container, fluid=True),
            html.Footer(
                dbc.Container(
                    html.P("© 2023 Research Dashboard", className="text-center", style={"color": COLOR_PALETTE["accent2"]}),
                    className="mt-5 py-3",
                    style={"backgroundColor": COLOR_PALETTE["dark"]}
                )
            )
        ],
        style={"display": "none"}  # Hidden until loader finishes
    )
])

# Fade out flame and show app after delay
@callback(
    Output("loader", "style"),
    Output("page-content", "style"),
    Input("url", "pathname"),
    prevent_initial_call=True
)
def finish_loader(pathname):
    time.sleep(5)  # Duration flame plays before hiding it
    return {"display": "none"}, {"display": "block"}

# Highlight active navbar link
@callback(
    Output("home-link", "active"),
    Output("objectives-link", "active"),
    Output("analytics-link", "active"),
    Output("findings-link", "active"),
    Input("url", "pathname")
)
def update_nav(path):
    return (
        path == "/",
        path == "/objectives",
        path == "/analytics",
        path == "/findings"
    )

if __name__ == '__main__':
    app.run(debug=True)
