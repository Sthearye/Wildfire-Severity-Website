from dash import html, register_page, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from visualizations.wildfire_trends import create_county_trends_plot
from visualizations.ca_choropleth import create_wildfire_choropleth
from visualizations.acres_by_county import create_acres_bar_chart
from visualizations.correlation_heatmap import create_correlation_heatmap
from visualizations.california_wildfire_dist import create_ca_wildfire_map
from visualizations.eto_scatter import eto_scatter_plot
from visualizations.climate_parallel import create_climate_parallel_plot 
from visualizations.monthly_patterns import MonthlyPatternsPlot
from visualizations.drought_fire_scatter import drought_fire_scatter_plot

# Load your data
df = pd.read_csv('datasets/new_merged_df.csv')

register_page(__name__, path="/findings")

COLORS = {
    "bg": "#16060C05",
    "card": "#FFFFFF",
    "header": "#D05F33",
    "text": "#16060C",
    "highlight": "#FF7621",
    "card_border": "#FF7621",
    "insight_bg": "#FFF5EF"
}

INSIGHTS = {
    "county_choropleth": """
    - Wildfire frequency is especially high in southern counties (like Los Angeles and San Diego) and central counties (like Fresno and Kern), suggesting these regions are critical targets for severity prediction models.
    - Many coastal counties and highly urbanized areas (like San Francisco County) show lower wildfire counts, reflecting a natural barrier effect (coastline humidity) or stronger fire management infrastructure.
    - A few inland counties in the north (such as Butte and Shasta) show unexpectedly high wildfire counts compared to their neighbors, highlighting specific inland areas where fire severity could escalate quickly despite lower overall population density.
    """,
    "ca_wildfire_map": """
    - As the animation progresses year by year, larger and darker points (representing more acres burned) become more common — especially after the mid-2010s — suggesting that not only are wildfires becoming more frequent, but individual fires are burning significantly more land.
    - The animation highlights that major wildfire incidents, particularly those burning over hundreds of thousands of acres, are heavily concentrated in Northern California — around areas like Sonoma, Napa, and Butte counties — rather than the southern desert regions.
    - Some years show sudden spikes in wildfire activity, where multiple large fires ignite around the same time, indicating potential links to extreme weather events like heatwaves or drought years. This clustering effect emphasizes the importance of early detection and seasonal preparedness.
    """,
    "acres_burned": """
    - Tehama County alone experienced far more acres burned than any other county, highlighting it as the most wildfire-prone region in California by land impact.
    - The top five counties with the highest acres burned — Tehama, Trinity, Shasta, Colusa, and Lassen — are all located in Northern California, reinforcing that wildfire intensity is geographically skewed toward the north.
    - Counties like San Diego, Orange, and Santa Cruz show much lower total acres burned compared to interior counties, suggesting that urban development and coastal climate may reduce large-scale wildfire spread.
    """,
    "monthly_trends": """
    - From 2020 to 2022, there are noticeable spikes in the number of acres burned in specific counties, some reaching over 1.4 million acres in a single month. For example, Madera, San Bernardino, and Kern show some of the highest peaks. These extreme values suggest catastrophic fire events, such as the Creek Fire (Madera/Fresno, 2020) or Bobcat Fire (Los Angeles, 2020). The data suggests that wildfire activity is not evenly distributed over time or location—certain counties experience intense fires during specific years, rather than consistently high values over time.
    - The most dramatic cluster of spikes occurs during the 2020–2021 period, with several counties experiencing record-breaking wildfire activity. This aligns with known extreme fire seasons in California due to drought, record-high temperatures, and high fuel loads. The concentration of tall peaks during these years highlights the role of climate variability and possibly policy shifts in fire management, ignitions, and suppression.
    - The jagged, spike-like nature of many of the county lines suggests that wildfires are not only seasonal but episodic—large events can erupt and vanish from month to month, influenced by variables like wind patterns (e.g., Santa Ana winds), fuel moisture, and lightning storms or human activity. Counties like Butte and Tehama have repeated smaller spikes across years, indicating chronic exposure to wildfire threats, whereas others have more isolated peaks tied to specific incidents.
    """,
    "seasonal_patterns":  """
    - Wildfire acres burned sharply spike during summer months (July–September), while average temperatures steadily increase from winter through summer, showing that higher temperatures align with greater wildfire activity.
    """,
    "correlations": """
    - Wildfire size (acres burned) is weakly positively correlated with higher air temperatures and solar radiation, suggesting hotter, sunnier conditions slightly increase fire severity.
    - Higher relative humidity shows a slight negative correlation with fire size, indicating that moister air may help limit wildfire spread.
    - Wind-related variables and soil temperature have almost no strong correlation with fire size, implying they are less direct drivers compared to temperature and dryness.
    """,
    "climate_patterns": """
    - Large wildfire events occur under very low precipitation and moderate to high evapotranspiration (ETo), highlighting the critical role of water scarcity in increasing wildfire risk.
    - Higher solar radiation and maximum air temperatures are consistently linked to larger burned areas, emphasizing how extreme heat and sunlight dry out vegetation and fuel fires.
    - Surprisingly, many large fires happen with only moderate wind speeds, suggesting that dryness and heat, rather than extreme wind, are the primary drivers behind large-scale burns in these cases.
    """,
    "drought_impact": """
    - While many large fires occur during high drought severity (dark red dots), some of the biggest fires (largest circles) happen even when drought severity is low, suggesting that factors beyond drought, like heat and wind, also play key roles.
    - Post-2020, the plot shows a cluster of very large fires, indicating an increase in extreme fire events in recent years, regardless of drought severity.
    - Higher drought severity (darker colors) tends to correlate with larger fires overall, but the relationship is not absolute — meaning drought makes big fires more likely, but isn’t the only trigger.
    """,
    "evapotranspiration": """
    - Fires tend to get bigger as evapotranspiration increases, meaning drier conditions (more water loss) create a bigger risk for large fires.
    - Warmer temperatures (yellow dots) are more common among the larger fires, suggesting heat intensifies wildfire size even when evapotranspiration varies.
    - Bubble sizes (population) vary widely and don’t show a clear pattern, implying population density isn’t a major driver of fire size in this context.
    """
}

def create_insight_block(insight_key, title="Key Insights"):
    return dbc.Card(
        [
            dbc.CardHeader(title, style={"background": COLORS["highlight"], "color": "white"}),
            dbc.CardBody(
                dcc.Markdown(INSIGHTS[insight_key]),
                style={"background": COLORS["insight_bg"]}
            )
        ],
        className="mt-3 shadow-sm"
    )

layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader(
            "Wildfire Findings Dashboard",
            className="h4",
            style={"background": COLORS["header"], "color": "white"}
        ),
        
        dbc.CardBody([

            # 🔥 Replacing methodology section with summary findings
            dbc.Card(
                dbc.CardBody([
                    html.H4("Key Findings Summary", style={"color": COLORS["highlight"]}),
                    dcc.Markdown("""
                    Our analysis reveals that larger wildfires in California are strongly associated with higher temperatures and elevated evapotranspiration rates, both of which are symptoms of hotter and drier weather conditions. As evapotranspiration increases — meaning more moisture is lost from the land — vegetation becomes drier and more flammable, creating conditions that fuel larger fires. Temperature also plays a critical role, with hotter regions consistently experiencing larger burned areas regardless of local population density. Interestingly, while population influences exposure and damage potential, it is environmental factors, not human presence, that most strongly correlate with fire size.

                    These findings are crucial for improving wildfire severity predictions. By monitoring temperature trends and evapotranspiration levels, agencies can better forecast periods of heightened fire risk and allocate resources accordingly. In a warming climate, predictive models that integrate environmental stress indicators like these will be vital for early warning systems, risk mapping, and community preparedness. Overall, the data underscore the growing importance of proactive, climate-focused wildfire management strategies to reduce future impacts.
                    """, style={"backgroundColor": COLORS["insight_bg"], "padding": "15px", "borderRadius": "10px"})
                ]),
                className="mb-4 shadow-sm"
            ),

            # Geographic Visualizations with Insights
            html.H4("Geographic Distribution", style={"color": COLORS["highlight"]}),
            dbc.Tabs([
                dbc.Tab(
                    label="County Choropleth",
                    children=[
                        dbc.Row(dbc.Col(create_wildfire_choropleth())),
                        create_insight_block("county_choropleth")
                    ]
                ),
                dbc.Tab(
                    label="California Wildfire Map",
                    children=[
                        dbc.Row(dbc.Col(
                            dcc.Graph(
                                figure=create_ca_wildfire_map(df),
                                id='ca-wildfire-map',
                                style={'height': '600px'}
                            )
                        )),
                        html.Div(style={"height": "230px"}),  # 👈 Adds vertical space after the graph
                        dbc.Row(
                            dbc.Col(create_insight_block("ca_wildfire_map"))
                        )
                    ]
                ),
                dbc.Tab(
                    label="Acres Burned by County",
                    children=[
                        dbc.Row(dbc.Col(create_acres_bar_chart())),
                        create_insight_block("acres_burned")
                    ]
                )
            ], className="mb-4"),

            # Temporal Visualizations with Insights
            html.H4("Temporal Patterns", style={"color": COLORS["highlight"]}),
            dbc.Tabs([
                dbc.Tab(
                    label="Monthly Trends",
                    children=[
                        dbc.Row(dbc.Col(create_county_trends_plot())),
                        create_insight_block("monthly_trends")
                    ]
                ),
                dbc.Tab(
                    label="Seasonal Patterns",
                    children=[
                        dbc.Row(dbc.Col(
                            dcc.Graph(
                                figure=MonthlyPatternsPlot(df).get_figure(),
                                id='monthly-patterns-plot',
                                style={'height': '500px'}
                            )
                        )),
                        create_insight_block("seasonal_patterns")
                    ]
                )
            ], className="mb-4"),

            # Environmental Factors with Insights
            html.H4("Environmental Factors", style={"color": COLORS["highlight"]}),
            dbc.Tabs([
                dbc.Tab(
                    label="Feature Correlations",
                    children=[
                        dbc.Row(dbc.Col(create_correlation_heatmap())),
                        create_insight_block("correlations")
                    ]
                ),
                dbc.Tab(
                    label="Climate Patterns",
                    children=[
                        dbc.Row(dbc.Col(
                            dcc.Graph(
                                figure=create_climate_parallel_plot(df),
                                id='climate-parallel-plot',
                                style={'height': '500px'}
                            )
                        )),
                        create_insight_block("climate_patterns")
                    ]
                ),
                dbc.Tab(
                    label="Drought Impact",
                    children=[
                        dbc.Row(dbc.Col(
                            dcc.Graph(
                                figure=drought_fire_scatter_plot(df),
                                id='drought-scatter-plot',
                                style={'height': '500px'}
                            )
                        )),
                        create_insight_block("drought_impact")
                    ]
                ),
                dbc.Tab(
                    label="Evapotranspiration",
                    children=[
                        dbc.Row(dbc.Col(
                            dcc.Graph(
                                figure=eto_scatter_plot(df),
                                id='eto-scatter-plot',
                                style={'height': '500px'}
                            )
                        )),
                        create_insight_block("evapotranspiration")
                    ]
                )
            ])
        ], style={"background": COLORS["card"], "padding": "20px"})
    ], className="mt-4 shadow", style={"borderTop": f"3px solid {COLORS['card_border']}"} )
], fluid=True, style={"background": COLORS["bg"]})
