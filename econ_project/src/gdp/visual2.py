import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np

# Load the GDP data
file_path = '../../data/gdp/GDP_Growth_Analysis_Post80.csv'
gdp_data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')

# Extract relevant columns for plotting
gdp_data['YoY_GDP_Growth'] = pd.to_numeric(gdp_data['YoY_GDP_Growth'], errors='coerce')

# Drop NaN values
gdp_data.dropna(subset=['YoY_GDP_Growth'], inplace=True)

# Calculate normal distribution percentiles
mean, std = gdp_data['YoY_GDP_Growth'].mean(), gdp_data['YoY_GDP_Growth'].std()
gdp_data['Normal_Distribution_Percentile'] = stats.norm.cdf(gdp_data['YoY_GDP_Growth'], mean, std)

# Calculate growth percentiles
gdp_data['Growth_Percentile'] = gdp_data['YoY_GDP_Growth'].rank(pct=True)

# Create a time plot
fig = go.Figure()

# Plot raw YoY GDP growth data
fig.add_trace(go.Scatter(
    x=gdp_data.index,
    y=gdp_data['YoY_GDP_Growth'],
    mode='lines+markers',
    name='YoY GDP Growth (%)',
    text=gdp_data.apply(lambda row: f"YoY Growth: {row['YoY_GDP_Growth']}%<br>Growth Percentile: {row['Growth_Percentile']}<br>Normal Dist Percentile: {row['Normal_Distribution_Percentile']}", axis=1),
    hoverinfo='text'
))

# Plot growth percentiles
fig.add_trace(go.Scatter(
    x=gdp_data.index,
    y=gdp_data['Growth_Percentile'],
    mode='lines+markers',
    name='Growth Percentile',
    text=gdp_data.apply(lambda row: f"YoY Growth: {row['YoY_GDP_Growth']}%<br>Growth Percentile: {row['Growth_Percentile']}<br>Normal Dist Percentile: {row['Normal_Distribution_Percentile']}", axis=1),
    hoverinfo='text'
))

# Plot normal distribution percentiles
fig.add_trace(go.Scatter(
    x=gdp_data.index,
    y=gdp_data['Normal_Distribution_Percentile'],
    mode='lines+markers',
    name='Normal Distribution Percentile',
    text=gdp_data.apply(lambda row: f"YoY Growth: {row['YoY_GDP_Growth']}%<br>Growth Percentile: {row['Growth_Percentile']}<br>Normal Dist Percentile: {row['Normal_Distribution_Percentile']}", axis=1),
    hoverinfo='text'
))

# Customize the layout
fig.update_layout(
    title='YoY GDP Growth, Growth Percentile, and Normal Distribution Percentile Over Time',
    xaxis_title='Time',
    yaxis_title='Values',
    legend_title='Legend'
)

# Show the figure
fig.show()
