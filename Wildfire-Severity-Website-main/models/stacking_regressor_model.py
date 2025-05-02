import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib
matplotlib.use('Agg')
import base64
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import xgboost as xgb

# Define color constants
COLORS = {
    "accent": "#D05F33",
    "secondary": "#E88D67",
    "light_accent": "#FFE8D6",
    "text": "#16060C",
}

# Global cache to avoid retraining
model_cache = {}


def run_stacking_regressor_model():
    if "outputs" not in model_cache:
        df = pd.read_csv("datasets/new_merged_df.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["FireSeason"] = df["Month"].apply(lambda x: 1 if 5 <= x <= 10 else 0)
        df["log_Precip"] = np.log1p(df["Precip (in)"])
        df["Drought_Severity_Index"] = df[["D0", "D1", "D2", "D3", "D4"]].sum(axis=1)
        df.drop(columns=["D0", "D1", "D2", "D3", "D4"], inplace=True)
        df = df.dropna(subset=["incident_acres_burned"])
        q_low = df["incident_acres_burned"].quantile(0.01)
        q_hi = df["incident_acres_burned"].quantile(0.99)
        df = df[(df["incident_acres_burned"] > q_low) & (df["incident_acres_burned"] < q_hi)]

        y = np.log1p(df["incident_acres_burned"])
        X = df.drop(columns=["incident_acres_burned"])
        drop_cols = ["Date", "incident_latitude", "incident_longitude", "severity"]
        X = X.drop(columns=[col for col in drop_cols if col in X.columns])

        categorical_features = ["County"] if "County" in X.columns else []
        numeric_features = X.select_dtypes(include=[np.number]).columns.difference(categorical_features)

        preprocessor = ColumnTransformer(transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", RobustScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

        # Define and train stacking model
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=150, max_depth=6,
                                     learning_rate=0.1, random_state=42)
        stack_model = StackingRegressor(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model)],
                                        final_estimator=LinearRegression())

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", stack_model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Create and encode plots
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(np.log1p(df["incident_acres_burned"]), bins=50, kde=True, ax=ax1, color="#ff7043")
        ax1.set_title("Distribution of Incident Acres Burned (Log Transformed)")
        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf1.seek(0)
        hist_image = base64.b64encode(buf1.read()).decode("utf-8")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax2, alpha=0.6, color="#4caf50")
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax2.set_title("Predicted vs Actual (Log Transformed)")
        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf2.seek(0)
        scatter_image = base64.b64encode(buf2.read()).decode("utf-8")

        model_cache["outputs"] = (mse, r2, hist_image, scatter_image)

    mse, r2, hist_image, scatter_image = model_cache["outputs"]

    return html.Div([
        html.H3("Wildfire Acreage Prediction", style={"color": COLORS["text"], "textAlign": "center"}),
        html.Div([
            html.P(
                "Due to the complexity of wildfires, we understood that a single model could not properly capture all the patterns that may influence or affect wildfires so we decided to use a Stacking Regressor. This stacking regressor combined the models of random forest, linear, and XGBoost.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),
            html.P(
                "Compared to the other models we tested such as using just a linear model or XGBoost, it struggled to capture the results effectively. The stacking regressor gave us the best results.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),
            html.P(
                "We use a logarithmic transformation to improve the models performance since our orignal data had extreme outliers which may confuse regression model. It stabilizes variance as well whuich helps spread the data more uniformly. Lastly, it helps improve the model and accuracy which lead to poorer results.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),
        ], style={"backgroundColor": "#f0f8ff", "padding": "15px", "borderRadius": "6px", "marginBottom": "25px"}),



        html.H5("Stacking Regressor Performance Summary", style={"color": COLORS["text"], "textAlign": "center", "marginBottom": "25px"}),


        html.Div([
            html.Div(f"Mean Squared Error (Log Scale): {mse:.4f}", style={"fontSize": "15px", "color": COLORS["text"]}),
            html.Div(f"R² Score: {r2:.4f}", style={"fontSize": "15px", "color": COLORS["text"]})
        ], style={"backgroundColor": "#f0f8ff", "padding": "15px", "borderRadius": "6px", "marginBottom": "25px"}),


        html.Div([
            html.H5("Distribution of Burned Acres", style={"color": COLORS["text"]}),
            html.Img(src=f"data:image/png;base64,{hist_image}", style={"width": "100%", "borderRadius": "5px"}),
            
            html.P(
                "The log-transformed distribution illustrates the high right-skewness, indicating that most fires are typically small; "
                "larger fires occur less frequently, which is a good thing. The peak is around 4, which suggests that most fires occurred between 50–60 acres.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),

            html.P("The histogram shows how log transformation reshaped the data into a form that’s easier for the Stacking Regressor to learn from. "
                "It helps explain why we used this approach and provides context for the model’s performance — for example, why it does well on most fires but struggles with extremes.",
                style={"fontSize": "15px", "color": COLORS["text"], "marginTop": "10px"}) 

            ], style={"marginBottom": "30px"}),

        html.Div([
            html.H5("Predicted vs Actual (Log Transformed)", style={"color": COLORS["text"]}),
            html.Img(src=f"data:image/png;base64,{scatter_image}", style={"width": "100%", "borderRadius": "5px"}),
            html.P("The x-axis represents the actual values, while the y-axis shows the predicted values from the stacking regressor, both in log-transformed scale. Ideally, most points would align closely with the diagonal line, which would indicate accurate predictions. However, we can see that once the actual log-transformed values exceed 8, the model's predictions become less accurate. This suggests that the stacking regressor has difficulty predicting the size of more severe, large-scale fires.",
                   style={"fontSize": "15px", "color": COLORS["text"], "marginTop": "10px"})
        ])
    ])
