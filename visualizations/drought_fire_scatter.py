import plotly.express as px

def drought_fire_scatter_plot(df):
    fig = px.scatter(
        df,
        x='Date',
        y='incident_acres_burned',
        color='D4',  # Exceptional drought
        size='incident_acres_burned',
        hover_data={
            'County': True,
            'Date': '|%B %d, %Y',  # Format date nicely
            'D4': ':.1f',          # Format drought severity
            'incident_acres_burned': True
        },
        title='<b>Fire Size Over Time Colored by Drought Severity</b>',
        log_y=True,
        color_continuous_scale='orrd',  # Appropriate color scale for drought
        labels={
            'Date': 'Date',
            'incident_acres_burned': 'Acres Burned',
            'D4': 'Drought Severity (D4)'
        }
    )

    # Update layout for better readability
    fig.update_layout(
        yaxis_title="Acres Burned (log scale)",
        xaxis_title="Date",
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.1)',
        title_x=0.5,  # Center title
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        coloraxis_colorbar=dict(
            title='Drought Severity',
            thickness=20
        )
    )

    # Update marker appearance
    fig.update_traces(
        marker=dict(
            sizemode='area',
            sizeref=2.*max(df['incident_acres_burned'])/(40.**2),
            line=dict(width=0.5, color='DarkSlateGrey'),
            opacity=0.7
        ),
        selector=dict(mode='markers')
    )

    return fig