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

    # Plot Feature Importances
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, ax=ax)
    ax.set_title("Top 10 Feature Importances - XGBoost")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    feature_plot = base64.b64encode(buf.read()).decode("utf-8")

    # Return as Dash HTML
    return html.Div([
        html.H3("🔥 Wildfire Severity Prediction (XGBoost Model)"),
        html.P("This model uses XGBoost to predict the logarithm of acres burned in a wildfire based on meteorological and seasonal data."),

        html.Hr(),
        html.H4("📊 Evaluation Metrics"),
        html.Ul([
            html.Li(f"Log-space MSE: {mse_log:.4f}"),
            html.Li(f"Log-space R² Score: {r2_log:.4f}"),
            html.Li(f"Raw-space MSE: {mse:.2f}"),
            html.Li(f"Raw-space MAE: {mae:.2f}"),
            html.Li(f"Raw-space R² Score: {r2:.4f}"),
        ]),

        html.Hr(),
        html.H4("📉 Model Performance Summary", style={"color": "#C0392B"}),
        html.P(
            "Although the XGBoost model leverages powerful boosting techniques, the evaluation metrics indicate relatively poor performance. "
            "The R² scores — particularly in raw space — are low, suggesting the model struggles to explain variance in wildfire sizes. "
            "This could be due to insufficient feature relevance or the complexity of wildfire behavior, which might require spatial, temporal, or vegetation data not included here.",
            style={"fontSize": "16px", "marginBottom": "25px"}
        ),

        html.H4("📊 What These Evaluation Metrics Mean"),
        html.Ul([
            html.Li([
                html.Strong("Log-space MSE:"), " Measures average squared error between predicted and actual values on the log scale. Lower is better."
            ]),
            html.Li([
                html.Strong("Log-space R² Score:"), " Indicates how much variance in log(acres_burned) is explained by the model. 1 is perfect, 0 means no predictive power."
            ]),
            html.Li([
                html.Strong("Raw-space MSE:"), " Squared error after reversing the log transformation. Larger fires disproportionately affect this value."
            ]),
            html.Li([
                html.Strong("Raw-space MAE:"), " Average absolute error in predicted fire size (in acres). Easier to interpret than MSE."
            ]),
            html.Li([
                html.Strong("Raw-space R² Score:"), " Reflects how much variance in actual burned acres is explained. A score near 0 suggests limited real-world utility."
            ])
        ], style={"lineHeight": "1.8", "fontSize": "16px"}),

        html.Hr(),
        html.H4("🔍 Top Feature Importances"),
        html.Img(src=f"data:image/png;base64,{feature_plot}", style={"width": "80%", "margin": "auto", "display": "block"})
    ])
