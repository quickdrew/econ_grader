import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Load the CSV file
# Assume the CSV file contains columns: 'DATE', 'Real_GDP', 'GDP_Growth', 'YoY_GDP_Growth', 'Growth_Percentile', 'YoY_Growth_Percentile'
df = pd.read_csv('../data/GDP_Growth_Analysis.csv')

# Convert 'DATE' to datetime for better handling
df['DATE'] = pd.to_datetime(df['DATE'])

# Calculate the normal distribution for YoY GDP growth
mean_yoy_growth = df['YoY_GDP_Growth'].mean()
std_yoy_growth = df['YoY_GDP_Growth'].std()
df['YoY_Normal_Distribution'] = norm.pdf(df['YoY_GDP_Growth'], mean_yoy_growth, std_yoy_growth)

# Scale and center the normal distribution for better visualization
df['YoY_Normal_Distribution'] = (df['YoY_Normal_Distribution'] - df['YoY_Normal_Distribution'].mean()) / df['YoY_Normal_Distribution'].std()

# Creating an interactive plot with Plotly
fig = go.Figure()

# Add Year-over-Year GDP Growth trace
fig.add_trace(go.Scatter(
    x=df['DATE'], y=df['YoY_GDP_Growth'],
    mode='lines',
    name='YoY GDP Growth (%)',
    line=dict(color='blue')
))

# Add YoY Growth Percentile trace, centered around 50
fig.add_trace(go.Scatter(
    x=df['DATE'], y=df['YoY_Growth_Percentile'] - 50,
    mode='lines',
    name='Centered YoY Growth Percentile (%)',
    line=dict(color='green')
))

# # Add Annualized Quarterly Growth Rate trace
# fig.add_trace(go.Scatter(
#     x=df['DATE'], y=df['Annualized_GDP_Growth'],
#     mode='lines',
#     name='Annualized GDP Growth Rate (%)',
#     line=dict(color='red')
# ))

# Add YoY Normal Distribution trace
fig.add_trace(go.Scatter(
    x=df['DATE'], y=df['YoY_Normal_Distribution'],
    mode='lines',
    name='Centered YoY Growth Normal Distribution',
    line=dict(color='purple')
))

# Adding titles and labels
fig.update_layout(
    title='Real YoY GDP Growth, Centered Percentile Rank, Annualized Growth, and Centered YoY Growth Normal Distribution Over Time',
    xaxis_title='Date',
    yaxis_title='Percentage / Density',
    legend_title='Metrics',
    hovermode='x unified'
)

# Show the interactive plot
fig.show()
