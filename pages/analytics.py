from dash import html, register_page
import dash_bootstrap_components as dbc

register_page(__name__, path="/analytics")

COLORS = {
    "bg": "#16060C05",
    "card": "#FFFFFF",
    "header": "#D05F33",
    "text": "#16060C",
    "highlight": "#FF7621"
}

layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader("Analytical Methods", 
                      className="h4",
                      style={"background": COLORS["header"],
                             "color": "white"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Statistical Approaches:", 
                           style={"color": COLORS["highlight"]}),
                    html.Ul([
                        html.Li("Descriptive statistics for data profiling"),
                        html.Li("Hypothesis testing with p-value < 0.05 threshold"),
                        html.Li("Regression analysis for trend identification")
                    ])
                ], md=4),
                
                dbc.Col([
                    html.H5("Machine Learning Methods:", 
                           style={"color": COLORS["highlight"]}),
                    html.Ul([
                        html.Li("Random Forest for classification tasks"),
                        html.Li("XGBoost for feature importance ranking"),
                        html.Li("K-means clustering for segmentation")
                    ])
                ], md=4),
                
                dbc.Col([
                    html.H5("Validation Techniques:", 
                           style={"color": COLORS["highlight"]}),
                    html.Ul([
                        html.Li("10-fold cross-validation"),
                        html.Li("Holdout validation with 30% test set"),
                        html.Li("AUC-ROC for model performance")
                    ])
                ], md=4)
            ])
        ], style={"background": COLORS["card"]})
    ], className="mt-4 shadow", 
       style={"borderTop": f"3px solid {COLORS['highlight']}"})
], style={"background": COLORS["bg"]})