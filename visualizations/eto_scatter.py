import plotly.express as px

def eto_scatter_plot(df):
    fig = px.scatter(
        df,
        x='ETo (in)',
        y='incident_acres_burned',
        color='Avg Temperature (F)',
        size='Population',
        hover_data={
            'County': True,
            'Date': '|%B %d, %Y',  # Format date nicely
            'ETo (in)': ':.2f',     # Format to 2 decimal places
            'incident_acres_burned': True,
            'Avg Temperature (F)': ':.1f',
            'Population': True
        },
        title='<b>Wildfire Size vs Evapotranspiration</b><br><sup>Size represents population, color represents temperature</sup>',
        log_y=True,
        color_continuous_scale='thermal',  # More appropriate color scale for temperature
        labels={
            'ETo (in)': 'Reference Evapotranspiration (inches)',
            'incident_acres_burned': 'Acres Burned',
            'Avg Temperature (F)': 'Avg Temp (°F)',
            'Population': 'County Population'
        }
    )

    # Update layout for better readability
    fig.update_layout(
        yaxis_title="Acres Burned (log scale)",
        xaxis_title="Reference Evapotranspiration (inches)",
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.1)',
        title_x=0.5,  # Center title
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        coloraxis_colorbar=dict(
            title='Temp (°F)',
            thickness=20
        )
    )

    # Update marker appearance
    fig.update_traces(
        marker=dict(
            sizemode='area',
            sizeref=2.*max(df['Population'])/(40.**2),
            line=dict(width=0.5, color='DarkSlateGrey'),
            opacity=0.7
        ),
        selector=dict(mode='markers')
    )

    return fig