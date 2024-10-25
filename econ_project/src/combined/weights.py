import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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

# Merge GDP and unemployment data for averaging
combined_data = pd.merge(gdp_data[['Growth_Percentile']], unemployment_data[['Employment_Percentile']], left_index=True, right_index=True, how='inner')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("GDP Growth and Employment Percentile Analysis"),
    dcc.Graph(id='percentile-plot', style={'height': '75vh'}),
    html.Div([
        html.Label('GDP Weight:'),
        dcc.Slider(
            id='gdp-weight-slider',
            min=0,
            max=1,
            step=0.01,
            value=0.5,
            marks={i / 10: f'{i / 10}' for i in range(0, 11)}
        ),
        html.Label('Employment Weight:'),
        dcc.Slider(
            id='employment-weight-slider',
            min=0,
            max=1,
            step=0.01,
            value=0.5,
            marks={i / 10: f'{i / 10}' for i in range(0, 11)}
        )
    ], style={'display': 'inline-block', 'width': '70%'}),
    html.Div([
        html.Div(id='legend-div')
    ], style={'display': 'inline-block', 'width': '28%', 'vertical-align': 'top'})
])

# Callback to update the plot based on slider values
@app.callback(
    Output('percentile-plot', 'figure'),
    [Input('gdp-weight-slider', 'value'),
     Input('employment-weight-slider', 'value')]
)
def update_plot(gdp_weight, employment_weight):
    combined_data['Weighted_Average_Percentile'] = (
        gdp_weight * combined_data['Growth_Percentile'] + employment_weight * combined_data['Employment_Percentile']
    )

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

    # Plot weighted average percentiles
    fig.add_trace(go.Scatter(
        x=combined_data.index,
        y=combined_data['Weighted_Average_Percentile'],
        mode='lines',
        name='Weighted Average Percentile',
        text=combined_data.apply(lambda row: f"Weighted Average Percentile: {row['Weighted_Average_Percentile']:.2f}", axis=1),
        hoverinfo='text'
    ))

    # Customize the layout
    fig.update_layout(
        title='GDP Growth Percentile, Employment Percentile, and Weighted Average Percentile Over Time',
        xaxis_title='Time',
        yaxis_title='Percentiles',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center'),
        xaxis=dict(showspikes=True, spikemode='across+toaxis', spikethickness=1, spikecolor='grey', spikedash='dot'),
        hovermode='x unified'
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
