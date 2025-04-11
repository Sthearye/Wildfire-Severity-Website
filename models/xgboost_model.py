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

    # Drop Irrelevant Columns
    drop_cols = [col for col in ["County", "incident_longitude", "incident_latitude", "Date"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Handle Missing Values
    df = df.dropna(subset=["incident_acres_burned"])
    df = df.fillna(df.mean(numeric_only=True))

    # Remove Extreme Outliers (Top 1%)
    threshold = df["incident_acres_burned"].quantile(0.99)
    df = df[df["incident_acres_burned"] < threshold]

    # Define Features and Target
    y = np.log1p(df["incident_acres_burned"])
    X = df.drop(columns=["incident_acres_burned", "severity"]) if "severity" in df.columns else df.drop(columns=["incident_acres_burned"])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Model
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

    # Predictions (log space)
    y_pred_log = model.predict(X_test)

    # Evaluate in Log Space
    mse_log = mean_squared_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)

    # Reverse log-transform
    y_pred_acres = np.expm1(y_pred_log)
    y_test_acres = np.expm1(y_test)

    # Evaluate in Raw Acres
    mse = mean_squared_error(y_test_acres, y_pred_acres)
    mae = mean_absolute_error(y_test_acres, y_pred_acres)
    r2 = r2_score(y_test_acres, y_pred_acres)

    # Feature Importance Plot
    xgb.plot_importance(model, max_num_features=15)
    plt.title("Top 15 Feature Importances - XGBoost")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image1 = base64.b64encode(buf.read()).decode("utf-8")

    return html.Div([
        html.H3("🔥 Wildfire Severity Prediction (XGBoost Model)"),
        html.P(f"Log-space MSE: {mse_log:.2f}"),
        html.P(f"Log-space R²: {r2_log:.2f}"),
        html.Hr(),
        html.P(f"Mean Squared Error (Raw Acres): {mse:.2f}"),
        html.P(f"Mean Absolute Error (Raw Acres): {mae:.2f}"),
        html.P(f"R-squared Score (Raw Acres): {r2:.2f}"),
        html.Img(src=f"data:image/png;base64,{image1}", style={"width": "70%", "margin": "auto"})
    ])
