import pandas as pd
import plotly.express as px

def create_ca_wildfire_map(df):
    # Filter for California coordinates
    ca_df = df[
        (df['incident_latitude'].between(32.5, 42.0)) &
        (df['incident_longitude'].between(-124.5, -114.0))
    ].copy()

    # Extract year from the Date
    ca_df['Year'] = pd.to_datetime(ca_df['Date']).dt.year

    # Create the animated map
    fig = px.scatter_geo(
        ca_df,
        lat='incident_latitude',
        lon='incident_longitude',
        size='incident_acres_burned',
        color='incident_acres_burned',
        animation_frame='Year',
        hover_name='County',
        hover_data={
            'Date': '|%B %d, %Y',
            'Avg Temperature (F)': ':.1f',
            'incident_latitude': False,
            'incident_longitude': False,
            'incident_acres_burned': False
        },
        scope='north america',
        title='California Wildfire Distribution and Severity Over Time',
        color_continuous_scale='OrRd',
        projection='albers usa',
        center={'lat': 36.7783, 'lon': -119.4179},
        width=1000,
        height=800,
        labels={'incident_acres_burned': 'Acres Burned'}
    )

    # Geographic config
    fig.update_geos(
        visible=True,
        resolution=50,
        lataxis_range=[32, 42],
        lonaxis_range=[-124, -114],
        showsubunits=True,
        subunitcolor='rgba(0,0,0,0.2)',
        landcolor='lightgray',
        oceancolor='lightblue',
        coastlinewidth=1.5
    )

    # Marker scaling
    max_size = ca_df['incident_acres_burned'].max()
    fig.update_traces(
        marker=dict(
            sizeref=2. * max_size / (40.**2),
            sizemin=4,
            sizemode='area',
            line=dict(width=0.2, color='DarkSlateGrey'),
            opacity=0.7
        ),
        selector=dict(type='scattergeo')
    )

    # Update layout for smooth animation and readable year
    fig.update_layout(
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='blue',
            coastlinecolor='black',
            showland=True,
            showcountries=False,
            showocean=True
        ),
        coloraxis_colorbar=dict(
            title='Acres Burned',
            thickness=20,
            len=0.75
        ),
        title_x=0.5,
        title_font_size=20,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {
                        "frame": {"duration": 1200, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 800, "easing": "linear"}
                    }],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "currentvalue": {
                "visible": True,
                "prefix": "Year: ",
                "xanchor": "right",
                "font": {"size": 20, "color": "#333"}
            },
            "transition": {"duration": 500},
            "pad": {"b": 10},
        }]
    )

    return fig
