import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import xgboost as xgb

# Define color constants
COLORS = {
    "accent": "#D05F33",
    "secondary": "#E88D67",
    "light_accent": "#FFE8D6",
    "text": "#16060C",
}

# This function prepares and caches visuals
def prepare_model_outputs():
    df = pd.read_csv('datasets/new_merged_df.csv')
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)
    df["log_Precip"] = np.log1p(df["Precip (in)"])
    df["Drought_Severity_Index"] = df[["D0", "D1", "D2", "D3", "D4"]].sum(axis=1)
    df.drop(columns=["D0", "D1", "D2", "D3", "D4"], inplace=True)
    df = df.dropna(subset=["incident_acres_burned"])
    q_low = df["incident_acres_burned"].quantile(0.01)
    q_hi = df["incident_acres_burned"].quantile(0.99)
    df = df[(df["incident_acres_burned"] > q_low) & (df["incident_acres_burned"] < q_hi)]

    y = np.log1p(df["incident_acres_burned"])
    X = df.drop(columns=["incident_acres_burned"])
    drop_cols = ["Date", "incident_latitude", "incident_longitude", "severity"]
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    categorical_features = ["County"] if "County" in X.columns else []
    numeric_features = X.select_dtypes(include=[np.number]).columns.difference(categorical_features)

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    # Stacking Regressor
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    stack_model = StackingRegressor(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model)], final_estimator=LinearRegression())
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", stack_model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Histogram
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(np.log1p(df["incident_acres_burned"]), bins=50, kde=True, ax=ax1, color="#ff7043")
    ax1.set_title("Distribution of Incident Acres Burned (Log Transformed)", fontsize=14)
    ax1.set_xlabel("Log(1 + Acres Burned)", fontsize=12)
    ax1.grid(alpha=0.3)
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf1.seek(0)
    hist_image = base64.b64encode(buf1.read()).decode("utf-8")

    # Scatter plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax2, alpha=0.6, color="#4caf50")
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax2.set_title("Predicted vs Actual (Log Transformed)", fontsize=14)
    ax2.set_xlabel("Actual", fontsize=12)
    ax2.set_ylabel("Predicted", fontsize=12)
    ax2.grid(alpha=0.3)
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf2.seek(0)
    scatter_image = base64.b64encode(buf2.read()).decode("utf-8")

    return mse, r2, hist_image, scatter_image

# Cache outputs once
mse, r2, hist_image, scatter_image = prepare_model_outputs()

def run_stacking_regressor_model():
    # Define styles using inline style dictionaries instead of html.Style
    toggle_container_style = {
        'display': 'flex',
        'justifyContent': 'center',
        'margin': '20px 0',
        'gap': '15px'
    }
    
    selected_button_style = {
        'padding': '12px 20px',
        'border': 'none',
        'borderRadius': '8px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'backgroundColor': '#2c3e50',
        'color': 'white'
    }
    
    unselected_button_style = {
        'padding': '12px 20px',
        'border': 'none',
        'borderRadius': '8px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'backgroundColor': '#ecf0f1',
        'color': '#34495e'
    }
    
    viz_container_style = {
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'padding': '20px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
        'marginTop': '20px'
    }
    
    viz_description_style = {
        'backgroundColor': '#f8f9fa',
        'borderLeft': '4px solid #2c3e50',
        'padding': '12px 15px',
        'margin': '15px 0 25px 0',
        'borderRadius': '0 8px 8px 0',
        'fontStyle': 'italic',
        'color': '#495057'
    }
    
    metrics_container_style = {
        'display': 'flex',
        'gap': '20px',
        'marginBottom': '25px'
    }
    
    metric_card_style = {
        'flex': '1',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '8px',
        'padding': '15px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
    }
    
    metric_title_style = {
        'fontSize': '14px',
        'color': '#6c757d',
        'marginBottom': '8px'
    }
    
    metric_value_style = {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'color': '#2c3e50'
    }
    
    dashboard_title_style = {
        'color': '#2c3e50',
        'textAlign': 'center',
        'marginBottom': '30px',
        'fontWeight': '600',
        'borderBottom': '2px solid #ecf0f1',
        'paddingBottom': '15px'
    }
    
    return html.Div([
        html.H2("🔥 Wildfire Acreage Prediction Dashboard", style=dashboard_title_style),
        
        # Metrics cards
        html.Div([
            html.Div([
                html.Div("Mean Squared Error (Log Scale)", style=metric_title_style),
                html.Div(f"{mse:.4f}", style=metric_value_style)
            ], style=metric_card_style),
            
            html.Div([
                html.Div("R² Score", style=metric_title_style),
                html.Div(f"{r2:.4f}", style=metric_value_style)
            ], style=metric_card_style)
        ], style=metrics_container_style),
        
        # Custom toggle buttons
        html.Div([
            html.Button([
                html.I(className="fas fa-chart-bar"), " Distribution"
            ], id="btn-hist", style=selected_button_style, n_clicks=0),
            
            html.Button([
                html.I(className="fas fa-bullseye"), " Predicted vs Actual"
            ], id="btn-scatter", style=unselected_button_style, n_clicks=0)
        ], style=toggle_container_style),
        
        # Hidden radio for state management
        dcc.RadioItems(
            id="viz-selector",
            options=[
                {"label": "Distribution", "value": "hist"},
                {"label": "Predicted vs Actual", "value": "scatter"}
            ],
            value="hist",
            style={"display": "none"}
        ),
        
        # Visualization container
        html.Div([
            html.Div(id="viz-description", style=viz_description_style),
            html.Img(id="viz-image", style={"width": "100%", "margin": "auto", "display": "block"})
        ], style=viz_container_style)
    ])

# Callback to update the hidden radio when buttons are clicked
@callback(
    Output("viz-selector", "value"),
    Output("btn-hist", "style"),
    Output("btn-scatter", "style"),
    Input("btn-hist", "n_clicks"),
    Input("btn-scatter", "n_clicks")
)
def update_selection(hist_clicks, scatter_clicks):
    # Define button styles
    selected_style = {
        'padding': '12px 20px',
        'border': 'none',
        'borderRadius': '8px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'backgroundColor': '#2c3e50',
        'color': 'white'
    }
    
    unselected_style = {
        'padding': '12px 20px',
        'border': 'none',
        'borderRadius': '8px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'backgroundColor': '#ecf0f1',
        'color': '#34495e'
    }
    
    # Determine which button was clicked
    from dash import callback_context
    ctx = callback_context
    if not ctx.triggered:
        # Default to histogram
        return "hist", selected_style, unselected_style
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-hist":
        return "hist", selected_style, unselected_style
    else:
        return "scatter", unselected_style, selected_style

@callback(
    Output("viz-image", "src"),
    Output("viz-description", "children"),
    Input("viz-selector", "value")
)
def update_image(selected):
    if selected == "hist":
        return (
            f"data:image/png;base64,{hist_image}",
            html.Div([
                html.Div([
                    html.H4("Histogram Analysis", className="mb-3", style={"color": "#D05F33", "fontWeight": "600"}),
                    html.Div([
                        html.Span("📊", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                        html.P([
                            "Raw acres burned are extremely right-skewed and heteroscedastic: a few massive fires dominate both plots and least-squares fits, leading to poor performance.",
                        ], style={"fontSize": "1.05rem", "lineHeight": "1.6", "margin": "0"})
                    ], className="d-flex align-items-start mb-3"),
                    
                    html.Div([
                        html.P([
                            "In this case, we use ", html.Strong("Log(1 + x)"), " transformation which accomplishes many tasks:"
                        ], style={"fontSize": "1.05rem", "lineHeight": "1.6", "marginBottom": "10px"}),
                        
                        html.Ul([
                            html.Li("Compresses the tail so the wildfires appear closer to the bulk of the data", 
                                   style={"marginBottom": "8px"}),
                            html.Li("Stabilizes variance which helps satisfy the homoscedasticity (variance is constant throughout) assumptions", 
                                   style={"marginBottom": "8px"}),
                            html.Li("Approximates normality which helps prove regression diagnostics", 
                                   style={"marginBottom": "8px"})
                        ], style={"paddingLeft": "25px", "marginBottom": "20px"})
                    ], className="mb-4", style={"backgroundColor": "#FFE8D6", "padding": "15px", "borderRadius": "8px"})
                ], className="mb-4"),
                
                html.Div([
                    html.H4("Model Performance", className="mb-3", style={"color": "#D05F33", "fontWeight": "600"}),
                    
                    html.Div([
                        html.Div([
                            html.H5("Mean Squared Error (log-scale)", className="mb-2", style={"fontWeight": "600"}),
                            html.Div([
                                html.Span("3.24", style={"fontSize": "1.8rem", "fontWeight": "700", "color": "#D05F33"}),
                            ], className="mb-2"),
                            html.Ul([
                                html.Li("This is the average of the squared differences between our predicted and actual values", 
                                       style={"marginBottom": "8px"}),
                                html.Li("An MSE of 3.24 which is roughly 1.80 with the Root-mean-square deviation on the log scale. This shows that the predictions deviate about ±1.8 log units", 
                                       style={"marginBottom": "8px"}),
                                html.Li("On the raw-acre scale, it translates to an average multiplicative error of roughly e¹·⁸ ≈ 6×—so a 1,000-acre fire might be predicted anywhere between ~170 and 6,000 acres.", 
                                       style={"marginBottom": "8px"})
                            ], style={"paddingLeft": "25px"})
                        ], className="mb-4 p-3", style={"backgroundColor": "#F8F9FA", "borderRadius": "8px", "border": "1px solid #E9ECEF"}),
                        
                        html.Div([
                            html.H5("R² Score", className="mb-2", style={"fontWeight": "600"}),
                            html.Div([
                                html.Span("0.165", style={"fontSize": "1.8rem", "fontWeight": "700", "color": "#D05F33"}),
                            ], className="mb-2"),
                            html.P([
                                "Our model explains only about ", 
                                html.Strong("16.5%"), 
                                " of the variance in log-transformed fire severity. The remaining 83.5% of variability is unexplained which means that we were unsuccessful in capturing the key drivers of wildfire sizes or perhaps other relationships that we were unable to capture."
                            ], style={"fontSize": "1.05rem", "lineHeight": "1.6"})
                        ], className="mb-3 p-3", style={"backgroundColor": "#F8F9FA", "borderRadius": "8px", "border": "1px solid #E9ECEF"})
                    ])
                ])
            ])
        )
    else:
        return (
            f"data:image/png;base64,{scatter_image}",
            html.Div([
                html.Div([
                    html.H4("Scatter Plot Analysis", className="mb-3", style={"color": "#D05F33", "fontWeight": "600"}),
                    html.Div([
                        html.Span("🔍", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                        html.Div([
                            html.H5("Predicted vs. Actual Values (Log Scale)", className="mb-2", style={"fontWeight": "600"}),
                            html.P([
                                "This scatter plot compares the model's predicted fire sizes to the actual observed values. ",
                                "A perfectly accurate model would produce points that fall along the diagonal dashed line. ",
                                "Deviations from the line reflect prediction errors, while clustering around it indicates good model calibration."
                            ], style={"fontSize": "1.05rem", "lineHeight": "1.6"})
                        ])
                    ], className="d-flex align-items-start")
                ], className="p-3", style={"backgroundColor": "#FFE8D6", "borderRadius": "8px"})
            ])
        )
