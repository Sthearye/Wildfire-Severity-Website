import pandas as pd
import plotly.express as px
from dash import dcc

def create_acres_bar_chart():

    df = pd.read_csv('datasets/new_merged_df.csv')

    county_acres = df.groupby("County")["incident_acres_burned"].sum().reset_index()
    county_acres = county_acres.sort_values(by="incident_acres_burned", ascending=False)
    county_acres["County"] = county_acres["County"].str.title()

    fig = px.bar(
        county_acres,
        x="incident_acres_burned",
        y="County",
        orientation='h',
        title="<b>Total Acres Burned by County</b>",
        labels={"incident_acres_burned": "Acres Burned", "County": ""},
        height=800,
        color="incident_acres_burned",
        color_continuous_scale="OrRd"
    )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=20, t=60, b=20),
        coloraxis_showscale=False
    )
    
    return dcc.Graph(
        figure=fig,
        id='acres-by-county',
        style={'border-radius': '8px'}
    )