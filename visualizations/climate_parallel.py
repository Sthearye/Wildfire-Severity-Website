
import plotly.express as px

def create_climate_parallel_plot(df):
    climate_vars = ['ETo (in)', 'Precip (in)', 'Sol Rad (Ly/day)',
                   'Max Air Temp (F)', 'Avg Wind Speed (mph)',
                   'incident_acres_burned']
    
    fig = px.parallel_coordinates(
        df[climate_vars],
        color='incident_acres_burned',
        color_continuous_scale=px.colors.sequential.Inferno,
        title='<b>Climate Variables Relationship to Fire Severity</b>',
        labels={
            'ETo (in)': 'Evapotranspiration (in)',
            'Precip (in)': 'Precipitation (in)',
            'Sol Rad (Ly/day)': 'Solar Radiation (Ly/day)',
            'Max Air Temp (F)': 'Max Temp (°F)',
            'Avg Wind Speed (mph)': 'Wind Speed (mph)',
            'incident_acres_burned': 'Acres Burned'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(240,240,240,0.1)',
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=50),
        coloraxis_colorbar=dict(
            title='Acres Burned',
            thickness=20
        )
    )
    
    return fig