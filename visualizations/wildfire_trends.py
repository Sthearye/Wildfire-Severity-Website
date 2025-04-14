import json
import pandas as pd
import plotly.express as px
from dash import dcc


def create_county_trends_plot():
    df = pd.read_csv('datasets/new_merged_df.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    county_monthly_trend = (
        df.groupby([df['Date'].dt.to_period("M"), 'County'])['incident_acres_burned']
        .sum()
        .reset_index()
    )
    county_monthly_trend['Date'] = county_monthly_trend['Date'].dt.to_timestamp()

    fig = px.line(
        county_monthly_trend,
        x='Date',
        y='incident_acres_burned',
        color='County',
        title='Monthly Wildfire Acres Burned by County',
        labels={'incident_acres_burned': 'Total Acres Burned', 'Date': 'Date'},
        template='plotly_white'
    )

    counties = county_monthly_trend['County'].unique()
    visibility = lambda selected: [c == selected for c in counties]

    fig.update_traces(visible=False)
    for i, county in enumerate(counties):
        if i == 0:
            fig.data[i].visible = True

    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'label': county,
                    'method': 'update',
                    'args': [
                        {'visible': visibility(county)},
                        {'title': f'Monthly Wildfire Acres Burned - {county.capitalize()}'}
                    ]
                } for county in counties
            ],
            'direction': 'down',
            'x': 1.2,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top',
            'showactive': True
        }],
        hovermode='x unified'
    )
    
    return dcc.Graph(
        figure=fig,
        id='county-trends-plot',
        style={'height': '600px'}
    )