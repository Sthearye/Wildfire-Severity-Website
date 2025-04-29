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
    # Style definitions
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
    
    section_title_style = {
        'color': '#2c3e50',
        'marginTop': '25px',
        'marginBottom': '15px',
        'fontWeight': '500',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px'
    }
    
    # Data loading and model training (unchanged)
    df = pd.read_csv('datasets/new_merged_df.csv')

    # Feature Engineering
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)
    df["log_Precip"] = np.log1p(df["Precip (in)"])

    # Drop Unused Columns
    drop_cols = ["County", "incident_longitude", "incident_latitude", "Date"]
    df = df.drop(columns=drop_cols)

    # Handle Missing Values
    df = df.fillna(df.mean(numeric_only=True))

    # Prepare Features and Target
    y = np.log1p(df["incident_acres_burned"])
    X = df.drop(columns=["incident_acres_burned"])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Regressor
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

    # Predictions and Evaluation
    y_pred_log = model.predict(X_test)
    mse_log = mean_squared_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)

    y_pred_raw = np.expm1(y_pred_log)
    y_test_raw = np.expm1(y_test)

    mse = mean_squared_error(y_test_raw, y_pred_raw)
    mae = mean_absolute_error(y_test_raw, y_pred_raw)
    r2 = r2_score(y_test_raw, y_pred_raw)

    # Plot Feature Importances with improved styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, ax=ax, color='#3498db', grid=False)
    ax.set_title("Top 10 Feature Importances - XGBoost", fontsize=14, fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ddd')
    ax.spines['bottom'].set_color('#ddd')
    ax.tick_params(colors='#666')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    buf.seek(0)
    feature_plot = base64.b64encode(buf.read()).decode("utf-8")

    # Create actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_raw, y_pred_raw, alpha=0.5, color='#3498db')
    
    # Add perfect prediction line
    max_val = max(max(y_test_raw), max(y_pred_raw))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.title('Actual vs Predicted Fire Size', fontsize=14, fontweight='bold', color='#2c3e50')
    plt.xlabel('Actual Acres Burned', fontsize=12, color='#555')
    plt.ylabel('Predicted Acres Burned', fontsize=12, color='#555')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    buf2.seek(0)
    prediction_plot = base64.b64encode(buf2.read()).decode("utf-8")

    # Return as Dash HTML with enhanced styling
    return html.Div([
        html.H1("🔥 Wildfire Severity Prediction Dashboard", style=dashboard_title_style),
        

        
        # Main content container
        html.Div(style=viz_container_style, children=[
            html.H3("Model Performance Metrics", style=section_title_style),
            
            # Log-space metrics
            html.Div([
                html.H4("Log-Space Metrics", style={'color': '#34495e', 'marginBottom': '10px', 'fontSize': '16px'}),
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
            
            # Raw-space metrics
            html.Div([
                html.H4("Raw-Space Metrics", style={'color': '#34495e', 'marginBottom': '10px', 'fontSize': '16px'}),
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
            
            # Model Performance Summary
            html.H3("Model Performance Analysis", style=section_title_style),
            html.Div(style={
                'backgroundColor': '#fff8f8', 
                'borderLeft': '4px solid #C0392B',
                'padding': '15px',
                'borderRadius': '0 8px 8px 0',
                'marginBottom': '25px'
            }, children=[
                html.P(
                    "Although the XGBoost model leverages powerful boosting techniques, the evaluation metrics indicate relatively poor performance. "
                    "The R² scores — particularly in raw space — are low, suggesting the model struggles to explain variance in wildfire sizes. "
                    "This could be due to insufficient feature relevance or the complexity of wildfire behavior, which might require spatial, temporal, or vegetation data not included here.",
                    style={"fontSize": "15px", "lineHeight": "1.6", "color": "#555"}
                )
            ]),
            
            # Visualization section
            html.H3("Model Visualizations", style=section_title_style),
            
            # Feature Importance Plot
            html.Div(style={'marginBottom': '30px'}, children=[
                html.H4("Top Feature Importances", style={'color': '#34495e', 'marginBottom': '15px', 'fontSize': '16px'}),
                html.Img(src=f"data:image/png;base64,{feature_plot}", style={
                    "width": "100%", 
                    "maxWidth": "800px", 
                    "margin": "auto", 
                    "display": "block",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)"
                })
            ]),
            
            # Actual vs Predicted Plot
            html.Div(style={'marginBottom': '30px'}, children=[
                html.H4("Actual vs Predicted Fire Size", style={'color': '#34495e', 'marginBottom': '15px', 'fontSize': '16px'}),
                html.Img(src=f"data:image/png;base64,{prediction_plot}", style={
                    "width": "100%", 
                    "maxWidth": "800px", 
                    "margin": "auto", 
                    "display": "block",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)"
                })
            ]),
            
            # Metrics explanation
            html.H3("Understanding the Metrics", style=section_title_style),
            html.Div(style={
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
            }, children=[
                html.Ul([
                    html.Li([
                        html.Strong("Log-space MSE:"), " Measures the average squared difference between predicted and actual values of the log-transformed acres burned. ",
                        "Because it’s on the log scale, this value penalizes large proportional differences rather than absolute ones. A lower value means more stable predictions across the range of fire sizes."
                    ]),
                    html.Li([
                        html.Strong("Log-space R² Score:"), " Indicates the proportion of variance in log(acres_burned) that is explained by the model. ",
                        "A score close to 1 means the model fits the data well; a score near 0 means the model performs no better than simply predicting the mean. ",
                        "In this case, an R² of ~0.04 shows that the linear model captures very little of the underlying pattern, even after transformation."
                    ]),
                    html.Li([
                        html.Strong("Raw-space MSE:"), " This is the Mean Squared Error computed after reversing the log transformation (i.e., using actual acre values). ",
                        "This metric is sensitive to large fires and will be heavily skewed by outliers. It provides a direct sense of how far off predictions are in the real-world scale, but may exaggerate performance issues."
                    ]),
                    html.Li([
                        html.Strong("Raw-space MAE:"), " The Mean Absolute Error in predicted fire size (in acres), without squaring errors. ",
                        "This is more interpretable and robust to outliers than MSE, especially helpful when communicating real-world error ranges to stakeholders."
                    ]),
                    html.Li([
                        html.Strong("Raw-space R² Score:"), " Reflects how much variance in the actual acres burned (not log-transformed) is explained by the model. ",
                        "Because fire sizes follow a heavy-tailed distribution, R² in raw space often suffers when predictions fail to capture large-scale events. ",
                        "A near-zero score here implies that the model doesn't generalize well to true fire sizes."
                    ])
                ], style={"lineHeight": "1.8", "fontSize": "15px", "color": "#555", "paddingLeft": "20px"})
            ])

        ])
    ])
