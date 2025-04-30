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
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

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
        n_estimators=300,
        max_depth=12,
        class_weight='balanced',
        random_state=42
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

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_labels,
        cmap="YlOrRd",
        xticks_rotation=0,
        ax=ax
    )
    ax.set_title("Wildfire Severity Classification\nConfusion Matrix", fontsize=16, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, str(cm_display.confusion_matrix[i, j]),
                     ha="center", va="center", color="black" if cm_display.confusion_matrix[i, j] < 30 else "white",
                     fontsize=14)

    plt.tight_layout()
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
    plt.title('Top 10 Feature Importances', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    feat_buf = io.BytesIO()
    plt.savefig(feat_buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    feat_buf.seek(0)
    feat_image_encoded = base64.b64encode(feat_buf.read()).decode("utf-8")

    return html.Div([
        html.Div([
            html.H2("🔥 Wildfire Severity Classification", style={'textAlign': 'center', 'color': '#d62728'}),
            html.H4("Random Forest Model Analysis", style={'textAlign': 'center', 'color': '#555', 'marginBottom': '30px'})
        ]),

        html.Div([
            html.H4("Model Configuration", style={'color': '#d62728', 'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Div("🌲 Algorithm", style={
                        'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px',
                        'color': '#333', 'borderBottom': '2px solid #f0ad4e', 'paddingBottom': '5px'
                    }),
                    html.Div([
                        html.Div("Random Forest", style={'fontSize': '15px', 'fontWeight': 'bold'}),
                        html.Div([html.Span("Trees: ", style={'fontWeight': 'bold'}), html.Span("300", style={'color': '#d62728'})]),
                        html.Div([html.Span("Max Depth: ", style={'fontWeight': 'bold'}), html.Span("12", style={'color': '#d62728'})]),
                        html.Div([html.Span("Class Weights: ", style={'fontWeight': 'bold'}), html.Span("Balanced", style={'color': '#d62728'})])
                    ])
                ], style=config_card_style()),

                html.Div([
                    html.Div("🔥 Severity Thresholds", style={
                        'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px',
                        'color': '#333', 'borderBottom': '2px solid #5cb85c', 'paddingBottom': '5px'
                    }),
                    html.Div([
                        html.Div([html.Span("Low: ", style={'fontWeight': 'bold'}), html.Span(f"≤ {low_cutoff:.1f} acres", style={'color': '#5cb85c'})]),
                        html.Div([html.Span("Medium: ", style={'fontWeight': 'bold'}), html.Span(f"≤ {high_cutoff:.1f} acres", style={'color': '#f0ad4e'})]),
                        html.Div([html.Span("High: ", style={'fontWeight': 'bold'}), html.Span(f"> {high_cutoff:.1f} acres", style={'color': '#d9534f'})])
                    ])
                ], style=config_card_style(marginLeft='2%')),

                html.Div([
                    html.Div("📊 Data Split", style={
                        'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px',
                        'color': '#333', 'borderBottom': '2px solid #5bc0de', 'paddingBottom': '5px'
                    }),
                    html.Div([
                        html.Div([
                            html.Div(style={'width': '80%', 'height': '20px', 'backgroundColor': '#5bc0de', 'display': 'inline-block', 'borderRadius': '4px 0 0 4px'}),
                            html.Div(style={'width': '20%', 'height': '20px', 'backgroundColor': '#d9534f', 'display': 'inline-block', 'borderRadius': '0 4px 4px 0'})
                        ]),
                        html.Div([
                            html.Div("Training: 80%", style={'width': '50%', 'display': 'inline-block', 'textAlign': 'left', 'marginTop': '5px', 'fontWeight': 'bold', 'color': '#5bc0de'}),
                            html.Div("Testing: 20%", style={'width': '50%', 'display': 'inline-block', 'textAlign': 'right', 'marginTop': '5px', 'fontWeight': 'bold', 'color': '#d9534f'})
                        ]),
                        html.Div("Stratified by severity class", style={'marginTop': '10px', 'fontStyle': 'italic', 'fontSize': '14px'})
                    ])
                ], style=config_card_style(marginLeft='2%'))
            ])
        ], style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '25px', 'border': '1px solid #eee'}),

        html.Div([
            html.Div([
                html.H4("Confusion Matrix", style={'textAlign': 'center'}),
                html.Img(src=f"data:image/png;base64,{image_encoded}", style={"width": "100%", "marginBottom": "10px"}),
                html.P("This matrix visualizes how often the model correctly classified wildfire severity. "
                       "Correct predictions lie on the diagonal. Off-diagonal cells show misclassifications — for instance, predicting 'Low' when the fire was actually 'High'. "
                       "This is critical in fire management, as underestimating severe fires could delay response and resource allocation.", 
                       style={"fontSize": "20px", "color": "#555"})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H4("Feature Importance", style={'textAlign': 'center'}),
                html.Img(src=f"data:image/png;base64,{feat_image_encoded}", style={"width": "100%", "marginBottom": "10px"}),
                html.P("These are the top features influencing the model’s predictions. Wind and temperature variables dominate, "
                       "indicating that the classifier heavily relies on environmental conditions to assess severity. "
                       "Understanding these drivers helps prioritize which features matter most for accurate prediction.", 
                       style={"fontSize": "20px", "color": "#555"})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
        ]),

        html.Div([
            html.H4("Classification Metrics", style={'color': '#555'}),
            html.P("These metrics offer insight into model performance by severity class:\n"
                   "- Precision tells us how reliable predictions for each class are.\n"
                   "- Recall shows how many actual fires of each class were successfully detected.\n"
                   "- F1-score balances the two.\n"
                   "In this model, the 'Medium' class is hardest to classify, with the lowest scores. "
                   "This suggests the model struggles to differentiate medium severity fires from low or high ones — potentially due to overlapping feature patterns.",
                   style={"fontSize": "20px", "whiteSpace": "pre-wrap"}),
            html.Table([
                html.Tr([html.Th("Severity Class"), html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score")])
            ] + [
                html.Tr([
                    html.Td(index),
                    html.Td(f"{precision[i]:.3f}", style={'backgroundColor': get_color_scale(precision[i])}),
                    html.Td(f"{recall[i]:.3f}", style={'backgroundColor': get_color_scale(recall[i])}),
                    html.Td(f"{f1[i]:.3f}", style={'backgroundColor': get_color_scale(f1[i])})
                ]) for i, index in enumerate(metrics_table.index)
            ], style={'width': '100%', 'marginTop': '15px'})
        ], style={'marginTop': '30px'}),

        html.Div([
            html.P(f"Overall weighted F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}", 
                   style={'fontWeight': 'bold', 'textAlign': 'center'})
        ], style={'marginTop': '20px', 'backgroundColor': '#f9f9f9', 'padding': '15px', 'borderRadius': '5px'})
    ])

def get_color_scale(value):
    if value < 0.6: return '#ffcccc'
    elif value < 0.7: return '#ffeecc'
    elif value < 0.8: return '#ffffcc'
    elif value < 0.9: return '#e6ffcc'
    else: return '#ccffcc'

def config_card_style(marginLeft='0%'):
    return {
        'width': '32%',
        'display': 'inline-block',
        'backgroundColor': '#fff',
        'padding': '15px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'verticalAlign': 'top',
        'marginLeft': marginLeft
    }
