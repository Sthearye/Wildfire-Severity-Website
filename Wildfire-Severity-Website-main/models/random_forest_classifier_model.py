import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from dash import html
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score

COLORS = {"text": "#16060C"}

def run_rf_classifier_model():
    df = pd.read_csv('datasets/new_merged_df.csv')
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Drought_Severity_Index"] = df[["D0", "D1", "D2", "D3", "D4"]].sum(axis=1)

    drop_cols = ["D0", "D1", "D2", "D3", "D4", "County", "incident_longitude", 
                 "incident_latitude", "Date", "Population", "Density", "Land Area (mi)"]
    df.drop(columns=drop_cols, inplace=True)

    df = df.dropna(subset=["incident_acres_burned"])
    df = df[df["incident_acres_burned"] < df["incident_acres_burned"].quantile(0.99)]

    percentiles = df["incident_acres_burned"].quantile([0.33, 0.66])
    low_cutoff, high_cutoff = percentiles[0.33], percentiles[0.66]

    def label_severity(acres):
        if acres <= low_cutoff:
            return "Low"
        elif acres <= high_cutoff:
            return "Medium"
        else:
            return "High"

    df["severity_class"] = df["incident_acres_burned"].apply(label_severity)
    X = df.drop(columns=["incident_acres_burned", "severity_class"])
    y = df["severity_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight='balanced', random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    class_labels = ["Low", "Medium", "High"]
    precision = precision_score(y_test, y_pred, average=None, labels=class_labels)
    recall = recall_score(y_test, y_pred, average=None, labels=class_labels)
    f1 = f1_score(y_test, y_pred, average=None, labels=class_labels)

    metrics_table = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }, index=class_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_labels,
        cmap="YlOrRd",
        xticks_rotation=0,
        ax=ax
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    image_encoded = base64.b64encode(buf.read()).decode("utf-8")

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='YlOrRd')
    plt.title('Top 10 Feature Importances')
    feat_buf = io.BytesIO()
    plt.savefig(feat_buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    feat_buf.seek(0)
    feat_image_encoded = base64.b64encode(feat_buf.read()).decode("utf-8")

    return html.Div([

        html.H3("Fire Severity Classifier", style={"textAlign": "center", "color": COLORS["text"], "marginBottom": "25px"}),
        html.Div([
            html.P(
                "We use this model to predict how severe a wildfire might be, whether it may be low, medium, or high. "
                "This kind of classification is crucial for emergency response teams. It helps firefighters know "
                "what level of response is needed early on, which means they can act faster and more efficiently.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),
            html.P(
                "For example, a low-severity fire might only need a small crew, while a high-severity fire could require "
                "evacuations, aircraft, and major resource deployment. By getting this information early, we save time, "
                "protect lives, and reduce costs.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            ),
            html.P(
                "This model also helps us learn what causes the most severe fires"
                " which supports better planning and prevention in the long run.",
                style={"fontSize": "15px", "color": COLORS["text"]}
            )
        ], style={"backgroundColor": "#f0f8ff", "padding": "15px", "borderRadius": "6px", "marginBottom": "25px"}),

        
        html.Div([
            html.H5("📈 Confusion Matrix", style={"textAlign": "center", "color": COLORS["text"]}),
            html.Img(src=f"data:image/png;base64,{image_encoded}", style={"width": "100%", "marginBottom": "10px"}),
            html.P("This matrix shows how well the model distinguishes between fire severity classes. Diagonal = correct predictions.",
                   style={"fontSize": "15px", "color": COLORS["text"]})
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.H5("🔍 Feature Importance", style={"textAlign": "center", "color": COLORS["text"]}),
            html.Img(src=f"data:image/png;base64,{feat_image_encoded}", style={"width": "100%", "marginBottom": "10px"}),
            html.P("Wind and temperature appear as the top contributors to wildfire severity predictions.",
                   style={"fontSize": "15px", "color": COLORS["text"]})
        ], style={"marginBottom": "30px"}),

        html.H5("Metrics", style={"color": COLORS["text"]}),
        html.Table([
            html.Tr([html.Th("Severity"), html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score")])
        ] + [
            html.Tr([
                html.Td(index),
                html.Td(f"{precision[i]:.3f}", style={'backgroundColor': get_color_scale(precision[i])}),
                html.Td(f"{recall[i]:.3f}", style={'backgroundColor': get_color_scale(recall[i])}),
                html.Td(f"{f1[i]:.3f}", style={'backgroundColor': get_color_scale(f1[i])}),
            ]) for i, index in enumerate(metrics_table.index)
        ], style={"fontSize": "15px", "color": COLORS["text"], "width": "100%", "marginTop": "15px"}),

        html.Div([
            html.P(f"Overall Weighted F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}",
                   style={"fontWeight": "bold", "textAlign": "center", "fontSize": "16px", "color": COLORS["text"]})
        ], style={"marginTop": "20px", "backgroundColor": "#f9f9f9", "padding": "15px", "borderRadius": "6px"})
    ])

def get_color_scale(value):
    if value < 0.6: return '#ffcccc'
    elif value < 0.7: return '#fff0b3'
    elif value < 0.8: return '#ffffcc'
    elif value < 0.9: return '#e6ffcc'
    else: return '#ccffcc'
