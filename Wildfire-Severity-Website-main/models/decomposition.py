import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress
import io
import base64
from dash import html

COLORS = {"text": "#16060C"}

def run_time_series_decomposition(county, feature):
    df = pd.read_csv("datasets/cimis_merged.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["County"] == county]
    df.set_index("Date", inplace=True)
    ts = df[feature].resample("D").mean().dropna()

    if len(ts) < 730:
        return html.Div([html.P("Not enough data for decomposition.", style={"color": "red"})])

    decomposition = seasonal_decompose(ts, model="additive", period=365)
    trend = decomposition.trend.dropna()

    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{county} - Decomposition of {feature}", fontsize=14, y=0.995)
    plt.subplots_adjust(top=0.92)

    axes[0].plot(ts, label="Original")
    axes[0].legend(loc="upper left")

    axes[1].plot(trend, label="Trend")
    x = (trend.index - trend.index[0]).days.values
    y = trend.values
    slope, intercept, *_ = linregress(x, y)
    best_fit = slope * x + intercept
    axes[1].plot(trend.index, best_fit, linestyle='--', color='red', label='Best Fit Line')
    axes[1].legend(loc="upper left")

    axes[2].plot(decomposition.seasonal, label="Seasonality")
    axes[2].legend(loc="upper left")

    axes[3].plot(decomposition.resid, label="Residuals")
    axes[3].legend(loc="upper left")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    img_encoded = base64.b64encode(buf.read()).decode("utf-8")

    # === Holistic trend summary for all numeric features ===
    trend_summary_rows = []
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        ts_col = df[col].resample("D").mean().dropna()
        if len(ts_col) < 730:
            continue  # skip short time series
        trend_col = seasonal_decompose(ts_col, model="additive", period=365).trend.dropna()
        x_col = (trend_col.index - trend_col.index[0]).days.values
        y_col = trend_col.values
        slope_col, *_ = linregress(x_col, y_col)
        direction = "Increasing" if slope_col > 0 else "Decreasing" if slope_col < 0 else "Stable"
        trend_summary_rows.append(html.Tr([
            html.Td(col),
            html.Td(f"{slope_col:.6f}"),
            html.Td(direction)
        ]))

    trend_table = html.Div([
        html.H5("Trends Across Features", style={"color": COLORS["text"], "marginTop": "40px"}),
        html.Table([
            html.Tr([html.Th("Feature"), html.Th("Slope"), html.Th("Trend Direction")])
        ] + trend_summary_rows, style={"width": "100%", "fontSize": "14px", "marginTop": "10px"})
    ])

    return html.Div([
        html.H3("📊 Seasonal Decomposition", style={"textAlign": "center", "color": COLORS["text"]}),

        html.Div([
            html.P(
                "This decomposition separates long-term trends, seasonal effects, and noise, making it easier to understand weather behavior over time.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),
            html.Ul([
                html.Li("🔹 Original: The full time series, containing trend, seasonality, and residual noise.", style={"fontSize": "15px", "color": COLORS["text"]}),
                html.Li("🔹 Trend: The overall direction of the data.", style={"fontSize": "15px", "color": COLORS["text"]}),
                html.Li("🔹 Seasonality: The repeating, regular patterns that occur at fixed intervals, such as annual weather cycles.", style={"fontSize": "15px", "color": COLORS["text"]}),
                html.Li("🔹 Residuals: The remaining irregularities or noise in the data after removing the trend and seasonal patterns.", style={"fontSize": "15px", "color": COLORS["text"]})
            ], style={"marginTop": "10px"})
        ], style={"backgroundColor": "#f0f8ff", "padding": "15px", "borderRadius": "6px", "marginBottom": "25px"}),

        html.Div([
            html.Img(src=f"data:image/png;base64,{img_encoded}", style={"width": "100%", "borderRadius": "10px", "marginTop": "20px"})
        ]),

        trend_table
    ])
