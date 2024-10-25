import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np

# Load the GDP data
file_path = '../../data/gdp/GDP_Growth_Analysis_Post80.csv'
gdp_data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')

# Extract relevant columns for plotting
gdp_data = gdp_data[gdp_data.index >= '1980-01-01']
gdp_data['YoY_GDP_Growth'] = pd.to_numeric(gdp_data['YoY_GDP_Growth'], errors='coerce')

# Drop NaN values
gdp_data.dropna(subset=['YoY_GDP_Growth'], inplace=True)

# Calculate growth percentiles
gdp_data['Growth_Percentile'] = gdp_data['YoY_GDP_Growth'].rank(pct=True)

# Load the unemployment data
unemployment_file_path = '../../data/unemployment/UNRATE.csv'
unemployment_data = pd.read_csv(unemployment_file_path, parse_dates=['DATE'], index_col='DATE')

# Filter data after 1980
unemployment_data = unemployment_data[unemployment_data.index >= '1980-01-01']

# Calculate employment rate and percentiles for unemployment data
unemployment_data['Employment_Rate'] = 100 - unemployment_data['UNRATE']
unemployment_data['Employment_Percentile'] = unemployment_data['Employment_Rate'].rank(pct=True)

# Create a time plot
fig = go.Figure()

# Plot GDP growth percentiles
fig.add_trace(go.Scatter(
    x=gdp_data.index,
    y=gdp_data['Growth_Percentile'],
    mode='lines',
    name='GDP Growth Percentile',
    text=gdp_data.apply(lambda row: f"YoY Growth: {row['YoY_GDP_Growth']:.2f}%<br>Growth Percentile: {row['Growth_Percentile']:.2f}", axis=1),
    hoverinfo='text'
))

# Plot employment percentiles
fig.add_trace(go.Scatter(
    x=unemployment_data.index,
    y=unemployment_data['Employment_Percentile'],
    mode='lines',
    name='Employment Percentile',
    text=unemployment_data.apply(lambda row: f"Employment Rate: {row['Employment_Rate']:.2f}%<br>Employment Percentile: {row['Employment_Percentile']:.2f}", axis=1),
    hoverinfo='text'
))

# Customize the layout
fig.update_layout(
    title='GDP Growth Percentile and Employment Percentile Over Time',
    xaxis_title='Time',
    yaxis_title='Percentiles',
    legend_title='Legend',
    xaxis=dict(showspikes=True, spikemode='across+toaxis', spikethickness=1, spikecolor='grey', spikedash='dot'),
    hovermode='x unified'
)

# Show the figure
fig.show()
