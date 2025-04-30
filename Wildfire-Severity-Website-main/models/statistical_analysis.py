import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import io
import base64
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress_grid
from statsmodels.stats.diagnostic import linear_rainbow
from scipy.stats import spearmanr

# ---------- CONFIGURATION ----------
CIMIS_DATA_PATH = "datasets/cimis_merged.csv"
MERGED_DATA_PATH = "datasets/new_merged_df.csv"


# ---------- SCATTER PLOTS ----------
def generate_scatter_plots(df, features=None):
    if features is None or len(features) == 0:
        return []

    visuals = []
    for feature in features:
        if feature not in df.columns:
            continue  # Skip if invalid
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[feature], y=df['incident_acres_burned'])
        plt.title(f'{feature} vs Incident Acres Burned')
        plt.xlabel(feature)
        plt.ylabel('Acres Burned')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        visuals.append((feature, img_base64))

    return visuals


# ---------- LINEAR REGRESSION + PARTIAL PLOTS ----------
def linear_regression_with_diagnostics(df, features):
    if features is None or len(features) == 0:
        return {
            "summary": "No features selected.",
            "partial_regression_img": None,
            "rainbow_p_value": None
        }

    df = df.dropna(subset=['incident_acres_burned'])
    X = df[features]
    X = sm.add_constant(X)
    y = df['incident_acres_burned']

    model = sm.OLS(y, X).fit()

    # Partial regression plot
    fig = plt.figure(figsize=(12, 8))
    plot_partregress_grid(model, fig=fig)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    partreg_plot = base64.b64encode(buf.read()).decode("utf-8")

    # Rainbow test
    rainbow_statistic, rainbow_p_value = linear_rainbow(model)

    return {
        "summary": model.summary().as_text(),
        "partial_regression_img": partreg_plot,
        "rainbow_p_value": rainbow_p_value
    }


# ---------- SPEARMAN CORRELATION ----------
def get_spearman_results(df, variables=None):
    if variables is None:
        variables = ['Max Air Temp (F)', 'Avg Wind Speed (mph)', 'D4', 'Precip (in)']

    results = []
    for var in variables:
        if var in df.columns:
            corr, p = spearmanr(df['incident_acres_burned'], df[var])
            results.append({
                "variable": var,
                "rho": round(corr, 3),
                "p_value": round(p, 4),
                "significant": p < 0.05
            })
    return results


# ---------- SPEARMAN VISUALS ----------
def generate_spearman_visuals(df, variables=['Max Air Temp (F)', 'D4']):
    visuals = []
    for var in variables:
        if var in df.columns:
            plt.figure(figsize=(6, 4))
            sns.regplot(x=df[var], y=df['incident_acres_burned'], lowess=True, line_kws={"color": "red"})
            plt.title(f"{var} vs Acres Burned (Spearman Sig.)")
            plt.xlabel(var)
            plt.ylabel("Acres Burned")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            visuals.append((var, base64.b64encode(buf.read()).decode("utf-8")))
    return visuals


# ---------- Q-Q PLOT ----------
def get_qqplot_img(data_path=MERGED_DATA_PATH):
    df = pd.read_csv(data_path)
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

    # Q-Q Plot
    fig, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------- ACF / PACF PLOTS ----------
def generate_acf_pacf_plot(county, plot_type, feature):
    try:
        # Load and prep data
        df = pd.read_csv("datasets/cimis_merged.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        county_df = df[df['County'] == county]
        ts = county_df.set_index("Date")[feature].resample("D").mean().fillna(0)

        # Plot
        fig, ax = plt.subplots()
        if plot_type == "ACF":
            plot_acf(ts, ax=ax)
        else:
            plot_pacf(ts, ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8"), None
    except Exception as e:
        return None, f"Error generating plot: {e}"
