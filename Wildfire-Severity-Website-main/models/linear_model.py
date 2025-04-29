import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import io
import base64
from dash import html
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

    return html.Div([
        html.H3("🔥 Wildfire Severity Prediction (Linear Regression - Log Transformed)"),

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

        html.Hr(),
        html.H4("📈 Actual vs Predicted (Log-Transformed)"),
        html.P(
            "The scatter plot shows predicted vs. actual values in the log scale. Ideally, predictions should closely follow "
            "the red 1:1 line. However, we observe that the model consistently underpredicts for large incidents and overpredicts "
            "for smaller ones. This 'flattening' indicates limited sensitivity to extreme events — a typical limitation of linear models.",
            style={"marginBottom": "30px"}
        ),

        html.P(
            "The R² score of 0.0445 suggests that the model explains only about 4.5% of the variation in log(acres burned). "
            "Although the Mean Squared Error (3.72) is in the log scale, it still reflects considerable residual spread. "
            "In contrast, using the raw (unlogged) target variable would likely worsen performance due to extreme outliers "
            "dominating the loss function. The log transformation was essential to normalize the target and make modeling feasible, "
            "but the linear approach still struggles to capture complex wildfire dynamics.",
            style={"marginBottom": "30px"}
        ),

        html.Img(src=f"data:image/png;base64,{scatter_img}", style={"width": "70%", "margin": "auto", "display": "block"})
    ])

def get_qqplot_img():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('datasets/new_merged_df.csv')
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Month"] = df["Date"].dt.month
    df["log_Precip"] = np.log1p(df["Precip (in)"])
    df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)

    df = df.drop(columns=["County", "incident_longitude", "incident_latitude", "Date"])
    df = df.dropna(subset=["incident_acres_burned"])
    df = df.fillna(df.mean(numeric_only=True))

    q_low = df["incident_acres_burned"].quantile(0.01)
    q_hi = df["incident_acres_burned"].quantile(0.99)
    df = df[(df["incident_acres_burned"] > q_low) & (df["incident_acres_burned"] < q_hi)]

    df["log_acres_burned"] = np.log1p(df["incident_acres_burned"])
    y = df["log_acres_burned"]
    X = df.drop(columns=["incident_acres_burned", "log_acres_burned"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Generate Q-Q Plot
    fig, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")