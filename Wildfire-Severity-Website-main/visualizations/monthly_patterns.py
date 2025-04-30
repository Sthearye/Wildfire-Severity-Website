# visualizations/monthly_patterns.py
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class MonthlyPatternsPlot:
    def __init__(self, df):
        self.df = df
        self.create_plot()

    def create_plot(self):
        # Your original code exactly as provided
        monthly_agg = self.df.groupby('Month').agg({
            'incident_acres_burned': 'sum',
            'Avg Temperature (F)': 'mean',
            'Precip (in)': 'mean'
        }).reset_index()

        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fig.add_trace(
            go.Bar(x=monthly_agg['Month'],
                   y=monthly_agg['incident_acres_burned'],
                   name="Acres Burned"),
            secondary_y=False
        )
        self.fig.add_trace(
            go.Scatter(x=monthly_agg['Month'],
                       y=monthly_agg['Avg Temperature (F)'],
                       name="Avg Temp (F)"),
            secondary_y=True
        )
        self.fig.update_layout(
            title_text="Monthly Fire Patterns and Temperature"
        )
        self.fig.update_yaxes(title_text="Acres Burned", secondary_y=False)
        self.fig.update_yaxes(title_text="Temperature (F)", secondary_y=True)

    def get_figure(self):
        return self.fig