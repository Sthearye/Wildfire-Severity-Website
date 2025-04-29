import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from dash import html
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def run_logistic_model():
    # Load Dataset
    df = pd.read_csv('datasets/new_merged_df.csv')

    # Feature Engineering
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)
    df["log_Precip"] = np.log1p(df["Precip (in)"])

    # Drop Irrelevant Columns (only if they exist)
    drop_cols = [col for col in ["County", "incident_longitude", "incident_latitude", "Date"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Handle Missing Values
    df = df.dropna(subset=["incident_acres_burned"])
    df = df.fillna(df.mean(numeric_only=True))

    # Create Binary Classification Target for Severe Fires
    quantile_threshold = df["incident_acres_burned"].quantile(0.75)
    df["severe"] = (df["incident_acres_burned"] >= quantile_threshold).astype(int)

    # Define Features & Target
    X = df.drop(columns=["incident_acres_burned", "severe"])
    y = df["severe"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression Model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    # Predict Probabilities
    y_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.3  # Adjustable threshold for classification
    y_pred_thresh = (y_probs >= threshold).astype(int)

    # Generate Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_thresh,
        display_labels=["Non-Severe", "Severe"],
        cmap="Blues"
    )
    plt.title(f"Confusion Matrix - Logistic Regression (Threshold = {threshold})")

    # Save Plot for Dash
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image1 = base64.b64encode(buf.read()).decode("utf-8")

    return html.Div([
        html.H3("🔥 Wildfire Severity Classification (Logistic Regression)"),
        html.P(f"Quantile Threshold (75th percentile): {quantile_threshold:.2f} acres"),
        html.Img(src=f"data:image/png;base64,{image1}", style={"width": "60%", "margin": "auto"}),
    ])
