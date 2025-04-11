import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from dash import html
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_linear_model():
    df = pd.read_csv('datasets/new_merged_df.csv')

    # Feature Engineering
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Create FireSeason Feature (May to October)
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)

    # Log Transform of Precipitation
    df["log_Precip"] = np.log1p(df["Precip (in)"])

    # Drop irrelevant columns (only drop columns that exist)
    drop_cols = [col for col in ["County", "incident_longitude", "incident_latitude", "Date"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Handle Missing Values
    df = df.dropna(subset=["incident_acres_burned"])
    df = df.fillna(df.mean(numeric_only=True))

    # Target and Features
    y = np.log1p(df["incident_acres_burned"])
    X = df.drop(columns=["incident_acres_burned", "severity"]) if "severity" in df.columns else df.drop(columns=["incident_acres_burned"])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Log(Acres Burned)")
    plt.ylabel("Predicted Log(Acres Burned)")
    plt.title("Linear Regression: Actual vs Predicted (Log-Acres Burned)")
    plt.grid(True)

    # Convert Plot to Image for Dash
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode("utf-8")

    return html.Div([
        html.H3("🔥 Wildfire Severity Prediction (Linear Regression)"),
        html.P(f"Mean Squared Error: {mse:.2f}"),
        html.P(f"R-squared Score: {r2:.2f}"),
        html.Img(src=f"data:image/png;base64,{image}", style={"width": "70%", "margin": "auto"})
    ])
