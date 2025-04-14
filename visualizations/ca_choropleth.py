import json
import pandas as pd
import plotly.express as px
from dash import dcc

def create_wildfire_choropleth():
    df = pd.read_csv('datasets/new_merged_df.csv')
    df['County'] = df['County'].str.title().str.strip()
    
    wildfire_counts = df.groupby("County").size().reset_index(name="count")
    
    try:
        with open("datasets/California_County_Boundaries.geojson") as f:
            ca_geojson = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"GeoJSON file not found or invalid: {str(e)}")

    fig = px.choropleth(
        wildfire_counts,
        geojson=ca_geojson,
        locations='County',
        featureidkey="properties.CountyName",
        color='count',
        color_continuous_scale="OrRd",
        range_color=(0, wildfire_counts["count"].quantile(0.95)),  # Avoid outlier distortion
        scope="usa",
        labels={'count': 'Wildfire Count'},
        title='<b>California Wildfire Frequency by County</b>',
        hover_data={'County': True, 'count': ':.0f'}
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Wildfires",
            thickness=20,
            len=0.75
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        )
    )

    return dcc.Graph(
        figure=fig,
        id='ca-wildfire-choropleth',
        style={'height': '600px', 'border-radius': '10px'}
    )