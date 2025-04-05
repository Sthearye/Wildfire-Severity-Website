from dash import Dash, html, dcc, page_container, callback, Output, Input, no_update, clientside_callback
import dash_bootstrap_components as dbc
import time

# Custom color palette
COLOR_PALETTE = {
    "dark": "#16060C",    # Deep black
    "primary": "#D05F33", # Rust orange
    "secondary": "#9F5244", # Muted red
    "accent1": "#FF7621",  # Bright orange
    "accent2": "#FF9818",  # Golden orange
    "accent3": "#FEB504",   # Yellow
    "bg_opacity": "#16060C10"  # Dark with 10% opacity
}

app = Dash(__name__, 
           use_pages=True,
           pages_folder="pages",
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           assets_folder="assets")

server = app.server

# Loading screen with custom fire colors
loading_screen = html.Div(
    id="loading-screen",
    className="fire-container",
    children=[
        html.Div(
            className="fire-wrapper",
            children=[
                html.Div(className="flame flame-main"),
                html.Div(className="flame flame-left"),
                html.Div(className="flame flame-right"),
                html.Div(id="fire-sparks")
            ]
        ),
        html.P("Igniting your experience...", className="loading-text")
    ],
    style={
        "position": "fixed",
        "top": "0",
        "left": "0",
        "width": "100%",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "center",
        "alignItems": "center",
        "zIndex": "9999",
        "backgroundColor": COLOR_PALETTE["dark"]
    }
)

# Navigation bar with gradient
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand(
            "Research Dashboard", 
            href="/", 
            className="ms-2",
            style={"color": COLOR_PALETTE["accent3"]}
        ),
        dbc.Nav([
            dbc.NavLink(
                "Home", 
                href="/", 
                id="home-link",
                style={"color": COLOR_PALETTE["accent3"]}
            ),
            dbc.NavLink(
                "Objectives", 
                href="/objectives", 
                id="objectives-link",
                style={"color": COLOR_PALETTE["accent3"]}
            ),
            dbc.NavLink(
                "Methods", 
                href="/analytics", 
                id="analytics-link",
                style={"color": COLOR_PALETTE["accent3"]}
            ),
            dbc.NavLink(
                "Findings", 
                href="/findings", 
                id="findings-link",
                style={"color": COLOR_PALETTE["accent3"]}
            ),
        ], navbar=True)
    ]),
    color="dark",
    dark=True,
    sticky="top",
    className="mb-4",
    style={
        "background": f"linear-gradient(90deg, {COLOR_PALETTE['dark']} 0%, {COLOR_PALETTE['secondary']} 100%)",
        "boxShadow": f"0 4px 20px 0 {COLOR_PALETTE['primary']}33"  # 20% opacity
    }
)

app.layout = html.Div([
    dcc.Location(id='url'),
    dcc.Store(id='load-complete', data=False),
    loading_screen,
    html.Div(
        id="page-content",
        children=[
            navbar,
            dbc.Container(
                page_container,
                fluid=True,
                className="py-3",
                style={"backgroundColor": COLOR_PALETTE["bg_opacity"]}
            ),
            html.Footer(
                dbc.Container(
                    html.P(
                        "© 2023 Research Dashboard", 
                        className="text-center",
                        style={"color": COLOR_PALETTE["accent2"]}
                    ),
                    className="mt-5 py-3",
                    style={"backgroundColor": COLOR_PALETTE["dark"]}
                )
            )
        ],
        style={"display": "none"}
    ),
    dcc.Interval(
        id='spark-interval',
        interval=100,
        n_intervals=0
    )
])

# Spark animation callback
clientside_callback(
    """
    function(n_intervals) {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen && loadingScreen.style.display !== 'none') {
            const fireWrapper = document.querySelector('.fire-wrapper');
            if (fireWrapper) {
                const spark = document.createElement('div');
                spark.className = 'flame-spark';
                
                const leftPos = 50 + (Math.random() * 20 - 10);
                const size = 2 + Math.random() * 4;
                
                spark.style.width = `${size}px`;
                spark.style.height = `${size}px`;
                spark.style.left = `${leftPos}%`;
                spark.style.bottom = '0';
                spark.style.animation = `spark ${0.5 + Math.random() * 1}s forwards`;
                
                fireWrapper.appendChild(spark);
                
                setTimeout(() => {
                    if (spark.parentNode) {
                        spark.parentNode.removeChild(spark);
                    }
                }, 1500);
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-output', 'children'),
    Input('spark-interval', 'n_intervals')
)

# Loading screen control
@callback(
    [Output("loading-screen", "style"),
     Output("page-content", "style")],
    Input("url", "pathname"),
    prevent_initial_call=True
)
def update_loading(pathname):
    time.sleep(3)  # 3-second loading screen
    return {"display": "none"}, {"display": "block"}

# Active link highlighting
@callback(
    [Output("home-link", "active"),
     Output("objectives-link", "active"),
     Output("analytics-link", "active"), 
     Output("findings-link", "active")],
    Input("url", "pathname")
)
def update_active_links(pathname):
    return (
        pathname == "/",
        pathname == "/objectives",
        pathname == "/analytics",
        pathname == "/findings"
    )

if __name__ == '__main__':
    app.run(debug=True)