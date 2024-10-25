import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np

# Load the CSV file
df = pd.read_csv('../../data/gdp/GDP_Growth_Analysis_Post80.csv')

# Convert 'DATE' to datetime for better handling
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d', errors='coerce')


# Calculate the mean and standard deviation for YoY GDP Growth
mean_yoy = df['YoY_GDP_Growth'].mean()
std_yoy = df['YoY_GDP_Growth'].std()

# Calculate Z-scores for YoY GDP Growth
df['YoY_GDP_Growth_Z'] = (df['YoY_GDP_Growth'] - mean_yoy) / std_yoy

# Convert Z-scores to percentiles using the CDF of the normal distribution
df['YoY_Growth_Normalized_Percentile'] = stats.norm.cdf(df['YoY_GDP_Growth_Z']) * 100

# Generate values for plotting the normal distribution of YoY GDP Growth
x_values = np.linspace(mean_yoy - 4*std_yoy, mean_yoy + 4*std_yoy, 1000)
y_values = stats.norm.pdf(x_values, mean_yoy, std_yoy)

# Creating an interactive plot with Plotly
fig = go.Figure()

# Add Year-over-Year GDP Growth trace
fig.add_trace(go.Scatter(x=df['DATE'], y=df['YoY_GDP_Growth'],
                         mode='lines',
                         name='YoY GDP Growth (%)',
                         line=dict(color='blue')))

# Add YoY Growth Percentile trace, assuming normal distribution and centered around 50
fig.add_trace(go.Scatter(x=df['DATE'], y=df['YoY_Growth_Normalized_Percentile'] - 50,
                         mode='lines',
                         name='Centered YoY Growth Normalized Percentile (%)',
                         line=dict(color='green')))

# Add Annualized Quarterly Growth Rate trace
fig.add_trace(go.Scatter(x=df['DATE'], y=df['Annualized_GDP_Growth'],
                         mode='lines',
                         name='Annualized GDP Growth Rate (%)',
                         line=dict(color='red')))

# Add Normal Distribution of YoY GDP Growth
fig.add_trace(go.Scatter(x=x_values, y=y_values,
                         mode='lines',
                         name='Normal Distribution of YoY GDP Growth',
                         line=dict(color='purple', dash='dot')))

# Adding titles and labels
fig.update_layout(
    title='Real YoY GDP Growth, Normalized Percentile Rank, Annualized Growth, and Normal Distribution (1980s Onward)',
    xaxis_title='Date',
    yaxis_title='Percentage / Probability Density',
    legend_title='Metrics',
    hovermode='x unified'
)

# Show the interactive plot
fig.show()
