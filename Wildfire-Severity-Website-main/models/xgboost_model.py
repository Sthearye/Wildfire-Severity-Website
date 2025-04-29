import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from dash import html
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def run_xgboost_model():
    # Style Definitions
    dashboard_title_style = {
        'color': '#2c3e50',
        'textAlign': 'center',
        'marginBottom': '30px',
        'fontWeight': '700',
        'fontSize': '36px',
        'borderBottom': '2px solid #ecf0f1',
        'paddingBottom': '15px'
    }

    section_title_style = {
        'color': '#2c3e50',
        'marginTop': '35px',
        'marginBottom': '20px',
        'fontWeight': '600',
        'fontSize': '28px',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '10px'
    }

    paragraph_style = {
        "fontSize": "28px",
        "lineHeight": "1.65",
        "color": "#2c3e50",
        "marginBottom": "25px"
    }

    subheader_style = {
        'color': '#34495e',
        'marginBottom': '10px',
        'fontSize': '22px',
        'fontWeight': '500'
    }

    metrics_container_style = {
        'display': 'flex',
        'gap': '20px',
        'marginBottom': '25px',
        'flexWrap': 'wrap'
    }

    metric_card_style = {
        'flex': '1',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '8px',
        'padding': '15px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
        'minWidth': '150px'
    }

    metric_title_style = {
        'fontSize': '16px',
        'color': '#6c757d',
        'marginBottom': '8px'
    }

    metric_value_style = {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'color': '#2c3e50'
    }

    viz_container_style = {
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'padding': '20px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
        'marginTop': '20px'
    }

    # Load and preprocess data
    df = pd.read_csv('datasets/new_merged_df.csv')
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)
    df["log_Precip"] = np.log1p(df["Precip (in)"])

    df = df.drop(columns=["County", "incident_longitude", "incident_latitude", "Date"])
    df = df.fillna(df.mean(numeric_only=True))

    y = np.log1p(df["incident_acres_burned"])
    X = df.drop(columns=["incident_acres_burned"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    mse_log = mean_squared_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)

    y_pred_raw = np.expm1(y_pred_log)
    y_test_raw = np.expm1(y_test)

    mse = mean_squared_error(y_test_raw, y_pred_raw)
    mae = mean_absolute_error(y_test_raw, y_pred_raw)
    r2 = r2_score(y_test_raw, y_pred_raw)

    # Feature importance plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, ax=ax, color='#3498db', grid=False)
    ax.set_title("Top 10 Feature Importances - XGBoost", fontsize=14, fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    buf.seek(0)
    feature_plot = base64.b64encode(buf.read()).decode("utf-8")

    # Actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_raw, y_pred_raw, alpha=0.5, color='#3498db')
    max_val = max(max(y_test_raw), max(y_pred_raw))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    plt.title('Actual vs Predicted Fire Size', fontsize=14, fontweight='bold', color='#2c3e50')
    plt.xlabel('Actual Acres Burned', fontsize=12, color='#555')
    plt.ylabel('Predicted Acres Burned', fontsize=12, color='#555')
    plt.grid(True, alpha=0.3)
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    buf2.seek(0)
    prediction_plot = base64.b64encode(buf2.read()).decode("utf-8")

    return html.Div([
        html.H1("🔥 Wildfire Severity Prediction Dashboard", style=dashboard_title_style),

        html.Div(style=viz_container_style, children=[
            html.H3("Model Performance Metrics", style=section_title_style),

            html.Div([
                html.H4("Log-Space Metrics", style=subheader_style),
                html.Div(style=metrics_container_style, children=[
                    html.Div(style=metric_card_style, children=[
                        html.Div("MSE (Log)", style=metric_title_style),
                        html.Div(f"{mse_log:.4f}", style=metric_value_style)
                    ]),
                    html.Div(style=metric_card_style, children=[
                        html.Div("R² Score (Log)", style=metric_title_style),
                        html.Div(f"{r2_log:.4f}", style=metric_value_style)
                    ])
                ])
            ]),

            html.Div([
                html.H4("Raw-Space Metrics", style=subheader_style),
                html.Div(style=metrics_container_style, children=[
                    html.Div(style=metric_card_style, children=[
                        html.Div("MSE", style=metric_title_style),
                        html.Div(f"{mse:.2f}", style=metric_value_style)
                    ]),
                    html.Div(style=metric_card_style, children=[
                        html.Div("MAE", style=metric_title_style),
                        html.Div(f"{mae:.2f}", style=metric_value_style)
                    ]),
                    html.Div(style=metric_card_style, children=[
                        html.Div("R² Score", style=metric_title_style),
                        html.Div(f"{r2:.4f}", style=metric_value_style)
                    ])
                ])
            ]),

            html.H3("Model Performance Analysis", style=section_title_style),
            html.Div(style={
                'backgroundColor': '#fff8f8', 
                'borderLeft': '4px solid #C0392B',
                'padding': '20px',
                'borderRadius': '0 8px 8px 0',
                'marginBottom': '25px'
            }, children=[
                html.P(
                    "Although the XGBoost model leverages powerful boosting techniques, the evaluation metrics indicate relatively poor performance. "
                    "The R² scores — particularly in raw space — are low, suggesting the model struggles to explain variance in wildfire sizes. "
                    "This could be due to insufficient feature relevance or the complexity of wildfire behavior, which might require spatial, temporal, or vegetation data not included here.",
                    style=paragraph_style
                )
            ]),

            html.H3("Model Visualizations", style=section_title_style),
            html.Div([
                html.H4("Top Feature Importances", style=subheader_style),
                html.Img(src=f"data:image/png;base64,{feature_plot}", style={
                    "width": "100%", 
                    "maxWidth": "800px", 
                    "margin": "auto", 
                    "display": "block",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)"
                })
            ]),
            html.Div([
                html.H4("Actual vs Predicted Fire Size", style=subheader_style),
                html.Img(src=f"data:image/png;base64,{prediction_plot}", style={
                    "width": "100%", 
                    "maxWidth": "800px", 
                    "margin": "auto", 
                    "display": "block",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)"
                })
            ]),

            html.H3("Understanding the Metrics", style=section_title_style),
            html.Div(style={
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
            }, children=[
                html.Ul([
                    html.Li([
                        html.Strong("Log-space MSE:"), " Penalizes proportional errors. A lower value = better stability."
                    ]),
                    html.Li([
                        html.Strong("Log-space R² Score:"), " Measures variance explained in log(acres_burned)."
                    ]),
                    html.Li([
                        html.Strong("Raw-space MSE:"), " Measures squared error in original acre scale (sensitive to outliers)."
                    ]),
                    html.Li([
                        html.Strong("Raw-space MAE:"), " Mean error in predicted size, more interpretable."
                    ]),
                    html.Li([
                        html.Strong("Raw-space R² Score:"), " Indicates how well predictions match actual fire sizes."
                    ])
                ], style={"lineHeight": "1.8", "fontSize": "20px", "color": "#555", "paddingLeft": "20px"})
            ])
        ])
    ])
