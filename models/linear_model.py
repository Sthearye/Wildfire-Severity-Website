import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from dash import html
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

def run_linear_model():
    df = pd.read_csv('datasets/new_merged_df.csv')

    # Feature Engineering
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["log_Precip"] = np.log1p(df["Precip (in)"])
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)

    # Drop irrelevant columns
    drop_cols = ["County", "incident_longitude", "incident_latitude", "Date"]
    df = df.drop(columns=drop_cols)

    # Handle missing data
    df = df.dropna(subset=["incident_acres_burned"])
    df = df.fillna(df.mean(numeric_only=True))

    # Remove outliers
    q_low = df["incident_acres_burned"].quantile(0.01)
    q_hi = df["incident_acres_burned"].quantile(0.99)
    df = df[(df["incident_acres_burned"] > q_low) & (df["incident_acres_burned"] < q_hi)]

    # Log-transform target
    df["log_acres_burned"] = np.log1p(df["incident_acres_burned"])
    y = df["log_acres_burned"]
    X = df.drop(columns=["incident_acres_burned", "log_acres_burned"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --- Q-Q Plot ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot of Residuals")
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format="png")
    plt.close()
    buf1.seek(0)
    qq_img = base64.b64encode(buf1.read()).decode("utf-8")

    # --- Actual vs Predicted Plot ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(y_test, y_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Log(Acres Burned)")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Linear Regression: Actual vs Predicted")
    ax2.grid(True)
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    scatter_img = base64.b64encode(buf2.read()).decode("utf-8")

    # Return full dashboard section
    return html.Div([
        html.H3("🔥 Wildfire Severity Prediction (Linear Regression - Log Transformed)"),
        html.P(
        "This model applies multiple linear regression to predict the logarithm of acres burned in wildfire incidents. "
        "It assumes a linear relationship between wildfire size and features such as temperature, precipitation, wind, "
        "and seasonal timing. The log transformation is used to reduce skewness in the target variable and stabilize variance. "
        "Although linear regression is simple and interpretable, it may struggle to capture the complex, nonlinear dynamics "
        "of wildfire behavior without more advanced techniques or richer data.",
        style={"marginBottom": "20px"}
    ),
        html.P(f"Mean Squared Error (Log Scale): {mse:.4f}"),
        html.P(f"R-squared Score (Log Scale): {r2:.4f}"),

        html.Hr(),
        html.H4("📉 Q-Q Plot of Residuals"),
        html.Img(src=f"data:image/png;base64,{qq_img}", style={"width": "70%", "margin": "auto", "display": "block"}),
        html.P(
            "The Q-Q plot helps assess whether the residuals from the linear regression model are normally distributed — "
            "a key assumption for reliable statistical inference. In this model, the residuals mostly follow the reference line, "
            "but some deviation at the tails suggests slight non-normality. Overall, the model performs reasonably well in capturing linear trends, "
            f"with an R² of {r2:.2f}, though it likely misses complex wildfire behavior patterns."
        , style={"marginTop": "15px", "fontSize": "16px"}),

        html.Hr(),
        html.H4("📈 Actual vs Predicted (Log-Transformed)"),
        html.Img(src=f"data:image/png;base64,{scatter_img}", style={"width": "70%", "margin": "auto", "display": "block"})
    ])
