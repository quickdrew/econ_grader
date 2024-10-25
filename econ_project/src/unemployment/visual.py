import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np

# Load the unemployment data
file_path = '../../data/unemployment/UNRATE.csv'
unemployment_data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')

# Calculate percentiles and normal distribution percentiles
unemployment_data['Percentile'] = unemployment_data['UNRATE'].rank(pct=True)
mean, std = unemployment_data['UNRATE'].mean(), unemployment_data['UNRATE'].std()
unemployment_data['Normal_Distribution_Percentile'] = stats.norm.cdf(unemployment_data['UNRATE'], mean, std)

# Create a time plot
fig = go.Figure()

# Plot raw unemployment data
fig.add_trace(go.Scatter(
    x=unemployment_data.index, 
    y=unemployment_data['UNRATE'],
    mode='lines',
    name='Unemployment Rate'
))

# Plot percentiles
fig.add_trace(go.Scatter(
    x=unemployment_data.index,
    y=unemployment_data['Percentile'],
    mode='lines',
    name='Percentile'
))

# Plot normal distribution percentiles
fig.add_trace(go.Scatter(
    x=unemployment_data.index,
    y=unemployment_data['Normal_Distribution_Percentile'],
    mode='lines',
    name='Normal Distribution Percentile'
))

# Customize the layout
fig.update_layout(
    title='Unemployment Rate, Percentile, and Normal Distribution Percentile Over Time',
    xaxis_title='Time',
    yaxis_title='Values',
    legend_title='Legend'
)

# Show the figure
fig.show()
