import pandas as pd
import plotly.express as px
from dash import dcc

def create_county_trends_plot():
    # Load and prepare data
    df = pd.read_csv('datasets/new_merged_df.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Create monthly summary by county
    county_monthly_trend = (
        df.groupby([df['Date'].dt.to_period("M"), 'County'])
        ['incident_acres_burned']
        .sum()
        .reset_index()
        .rename(columns={'incident_acres_burned': 'Total Acres Burned'})
    )
    county_monthly_trend['Date'] = county_monthly_trend['Date'].dt.to_timestamp()

    # Create the initial figure with all counties
    fig = px.line(
        county_monthly_trend,
        x='Date',
        y='Total Acres Burned',
        color='County',
        title='Monthly Wildfire Acres Burned by County',
        labels={'Total Acres Burned': 'Total Acres Burned', 'Date': 'Date'},
        template='plotly_white'
    )

    # Add dropdown menu functionality
    counties = county_monthly_trend['County'].unique()

    # Create buttons for dropdown
    buttons = [
        {
            'label': 'All Counties',
            'method': 'update',
            'args': [
                {'visible': [True] * len(counties)},
                {'title': 'Monthly Wildfire Acres Burned - All Counties'}
            ]
        }
    ] + [
        {
            'label': county,
            'method': 'update',
            'args': [
                {'visible': [c == county for c in counties]},
                {'title': f'Monthly Wildfire Acres Burned - {county}'}
            ]
        } for county in counties
    ]

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'x': 1.1,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top',
                'showactive': True,
                'active': 0  # Default to "All Counties"
            }
        ],
        hovermode='x unified',
        legend_title_text='County',
        margin=dict(t=100)
    )

    # Default to showing all counties
    for trace in fig.data:
        trace.visible = True

    return dcc.Graph(
        figure=fig,
        id='county-trends-plot',
        style={'height': '600px'}
    )