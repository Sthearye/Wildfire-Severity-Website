from dash import html, dcc, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

register_page(__name__, path="/findings")

COLOR_SEQUENCE = ["#9F5244", "#D05F33", "#FF7621", "#FF9818", "#FEB504"]

# Sample data
np.random.seed(42)
data = pd.DataFrame({
    "Category": ["A", "B", "C", "D", "E"],
    "Value": np.random.randint(10, 100, 5),
    "Impact": np.random.uniform(0.5, 1.5, 5)
})

# Create visualizations with custom palette
fig1 = px.bar(data, x="Category", y="Value", 
             color="Category", color_discrete_sequence=COLOR_SEQUENCE,
             title="Key Metrics by Category")

fig2 = px.scatter(data, x="Value", y="Impact", 
                 color="Category", size="Value",
                 color_discrete_sequence=COLOR_SEQUENCE,
                 title="Value vs Impact Correlation")

layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader("Major Findings", 
                      style={"background": "#16060C", 
                             "color": "#FEB504"}),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(figure=fig1),
                    label="Metrics",
                    tabClassName="border-top border-start border-end",
                    label_style={"color": "#D05F33", "fontWeight": "bold"}
                ),
                dbc.Tab(
                    dcc.Graph(figure=fig2),
                    label="Correlations",
                    label_style={"color": "#D05F33", "fontWeight": "bold"}
                )
            ]),
            
            html.Div([
                html.H5("Key Insights:", style={"color": "#FF7621"}),
                html.P("Our analysis revealed...", style={"color": "#16060C"})
            ], className="mt-4 p-3", 
               style={"background": "#FF981820",  # 20% opacity
                      "borderLeft": f"4px solid {COLOR_SEQUENCE[1]}"})
        ], style={"background": "#FFFFFF"})
    ], className="mt-4 shadow-lg")
], style={"background": "#16060C10"})