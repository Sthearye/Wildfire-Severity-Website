import plotly.express as px

def create_ca_wildfire_map(df):
    # Filter for California coordinates
    ca_df = df[
        (df['incident_latitude'].between(32.5, 42.0)) &
        (df['incident_longitude'].between(-124.5, -114.0))
    ].copy()

    # Create the base map
    fig = px.scatter_geo(
        ca_df,
        lat='incident_latitude',
        lon='incident_longitude',
        size='incident_acres_burned',
        color='incident_acres_burned',
        hover_name='County',
        hover_data={
            'Date': '|%B %d, %Y',
            'Avg Temperature (F)': ':.1f',
            'incident_latitude': False,
            'incident_longitude': False,
            'incident_acres_burned': False
        },
        scope='north america',
        title='California Wildfire Distribution and Severity',
        color_continuous_scale='OrRd',
        projection='albers usa',
        center={'lat': 36.7783, 'lon': -119.4179},  # Center on California
        width=1000,
        height=800,
        labels={'incident_acres_burned': 'Acres Burned'}
    )

    # Configure geographic boundaries
    fig.update_geos(
        visible=True,
        resolution=50,
        lataxis_range=[32, 42],  # California latitude range
        lonaxis_range=[-124, -114],  # California longitude range
        showsubunits=True,  # Show county borders
        subunitcolor='rgba(0,0,0,0.2)',  # Light borders
        landcolor='lightgray',
        oceancolor='lightblue',
        coastlinewidth=1.5
    )

    # Configure marker appearance
    max_size = ca_df['incident_acres_burned'].max()
    fig.update_traces(
        marker=dict(
            sizeref=2. * max_size / (40.**2),  # Scale marker sizes
            sizemin=4,  # Minimum marker size
            sizemode='area',  # Scale by area
            line=dict(width=0.2, color='DarkSlateGrey'),
            opacity=0.7  # Slightly transparent for better visibility
        ),
        selector=dict(type='scattergeo')
    )

    # Update layout for better presentation
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
        title_x=0.5,  # Center the title
        title_font_size=20
    )

    return fig