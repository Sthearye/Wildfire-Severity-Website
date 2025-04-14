import pandas as pd
import plotly.express as px
from dash import dcc

def create_correlation_heatmap():
    df = pd.read_csv('datasets/new_merged_df.csv')
    
    numeric_df = df.select_dtypes(include=['float64'])

    corr_matrix = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    fig.update_traces(
        text=corr_matrix.values,
        texttemplate="%{text}",
        hovertemplate="<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Correlation:</b> %{z:.2f}"
    )
    
    fig.update_layout(
        title={
            'text': "<b>Feature Correlation Matrix</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center'
        },
        height=700,
        margin=dict(l=100, r=100, t=100, b=100),
        coloraxis_colorbar=dict(
            title="Correlation",
            thickness=15,
            len=0.75,
            yanchor="middle",
            y=0.5
        ),
        xaxis=dict(tickangle=45)
    )
    
    return dcc.Graph(
        figure=fig,
        id='correlation-heatmap',
        style={'border-radius': '8px'}
    )