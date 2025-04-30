from dash import html, dcc, register_page, callback, Output, Input
import dash_bootstrap_components as dbc

register_page(__name__, path="/")

# Wildfire-themed color palette
colors = {
    "background": "#fff9f2",
    "card": "#ffffff",
    "primary": "#e76f51",
    "secondary": "#f4a261",
    "accent": "#2a9d8f",
    "text": "#5a3e36",
    "alert": "#e9c46a"
}

hero = dbc.Row([
    dbc.Col([
        html.H1("California Wildfire Severity Prediction",
                className="display-4 fw-bold mb-3",
                style={"color": colors["primary"]}),
        html.P("An approach utilizing historical and environmental data to predict the severity of wildfires.",
               className="lead",
               style={"color": colors["text"]}),
        dbc.Button("Explore Our Objectives →",
                   id="explore-btn",
                   href="/objectives",
                   color="primary",
                   className="mt-3",
                   style={"backgroundColor": colors["primary"]})
    ], className="text-center py-5")
], style={
    "background": "linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), "
                  "url('https://images.unsplash.com/photo-1516054575922-f0b8eeadec1a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80')",
    "backgroundSize": "cover",
    "borderRadius": "10px"
})

# Project summary card
summary_card = dbc.Card([
    dbc.CardHeader("Why This Matters",
                   style={"backgroundColor": colors["secondary"], "color": "white"}),
    dbc.CardBody([
        html.P("Since the 1980s, there has been a significant rise in both the size and intensity of wildfires in California. Since 2000, fifteen of California's 20 largest wildfires have happened, and since 2015, ten of the most damaging and expensive fires affecting life and property in the state have taken place.",
               style={"color": colors["text"]}),
        html.P("Our initiative uses machine learning methods to forecast wildfire intensity—not only incidence—by analyzing environmental factors such as temperature, humidity, and soil moisture.",
               className="mt-3",
               style={"color": colors["text"]}),
    ])
], className="shadow my-4")

# Image gallery section
image_gallery = dbc.Row([
    dbc.Col([
        html.H2("", className="text-center mb-4", style={"color": colors["primary"]}),
        dbc.Row([
            dbc.Col([
                html.Img(
                    src="/assets/38A06FAD-787A-4C6E-8EAF7961B4A15286_source.png",
                    className="img-fluid rounded shadow",
                    style={"maxHeight": "400px", "width": "100%", "objectFit": "cover"}
                ),
                html.P("", className="text-center mt-2", style={"color": colors["text"]})
            ], md=6, className="mb-4"),

            dbc.Col([
                html.Img(
                    src="/assets/merlin_176700381_709524cd-fbfe-499c-af77-596f80821067-superJumbo.png",
                    className="img-fluid rounded shadow",
                    style={"maxHeight": "400px", "width": "100%", "objectFit": "cover"}
                ),
                html.P("", className="text-center mt-2", style={"color": colors["text"]})
            ], md=6, className="mb-4")
        ])
    ])
], className="my-5")

# Key features cards
features = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Severity Prediction", style={"backgroundColor": colors["secondary"], "color": "white"}),
            dbc.CardBody([
                html.P("Go beyond binary 'fire/no fire' models to predict how destructive a wildfire will be.",
                       style={"color": colors["text"]}),
            ])
        ], className="h-100 shadow-sm")
    ], md=4),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Environmental Factors", style={"backgroundColor": colors["secondary"], "color": "white"}),
            dbc.CardBody([
                html.P("Analyze 10+ variables: temperature, humidity, wind speed, soil moisture, and more.",
                       style={"color": colors["text"]}),
            ])
        ], className="h-100 shadow-sm")
    ], md=4),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Actionable Insights", style={"backgroundColor": colors["secondary"], "color": "white"}),
            dbc.CardBody([
                html.P("Assist fire departments in distributing resources and aiding communities in getting ready for high-risk situations.",
                       style={"color": colors["text"]}),
            ])
        ], className="h-100 shadow-sm")
    ], md=4),
], className="my-4")

ways_to_help = html.Div(
    dbc.Button("Ways to Help",
               href="https://www.cafirefoundation.org/how-to-help/ways-to-give",
               color="primary",
               className="mt-4 mb-4 d-block mx-auto",
               style={"backgroundColor": colors['primary'], "border": "none", "fontSize": "1.1rem"}),
    className="text-center"
)

# Footer
footer = html.Footer([
    html.Hr(),
    html.P("CS 163 Project | Sokuntheary Em & Devin Chau",
           className="text-center",
           style={"color": colors["text"]})
], className="mt-3")

# Combine all components
layout = dbc.Container([
    hero,
    summary_card,
    image_gallery,
    features,
    ways_to_help,
    footer
], style={"backgroundColor": colors["background"], "padding": "2rem"})

# Optional callback for interactivity
@callback(
    Output("home-output", "children"),
    Input("explore-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_output(n_clicks):
    return dcc.Location(pathname="/objectives", id="redirect")
