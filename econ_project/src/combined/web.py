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
combined_data = pd.merge(gdp_data[['YoY_GDP_Growth', 'Growth_Percentile']], unemployment_data[['UNRATE', 'Employment_Percentile']], left_index=True, right_index=True, how='inner')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("GDP Growth and Employment Percentile Analysis", style={'color': 'grey'}),
    dcc.Graph(id='percentile-plot', style={'height': '75vh'}),
    html.Div([
        html.Label('GDP Weight:', style={'color': 'grey'}),
        dcc.Slider(
            id='gdp-weight-slider',
            min=0,
            max=1,
            step=0.01,
            value=0.5,
            marks={i / 10: f'{i / 10}' for i in range(0, 11)}
        ),
        html.Label('Employment Weight:', style={'color': 'grey'}),
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
        html.Div(id='data-info', style={'position': 'fixed', 'bottom': '10px', 'right': '150px', 'backgroundColor': '#333', 'color': 'grey', 'padding': '10px', 'border': '1px solid #444'})
    ], style={'display': 'inline-block', 'width': '28%', 'vertical-align': 'top'})
], style={'backgroundColor': '#121212', 'padding': '20px'})

# Callback to update the plot and display data information
@app.callback(
    [Output('percentile-plot', 'figure'),
     Output('data-info', 'children')],
    [Input('gdp-weight-slider', 'value'),
     Input('employment-weight-slider', 'value'),
     Input('percentile-plot', 'hoverData')]
)
def update_plot(gdp_weight, employment_weight, hover_data):
    combined_data['Weighted_Average_Percentile'] = (
        gdp_weight * combined_data['Growth_Percentile'] + employment_weight * combined_data['Employment_Percentile']
    )

    # Create a time plot
    fig = go.Figure()

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
        title='Weighted Average Percentile Over Time',
        xaxis_title='Time',
        yaxis_title='Percentiles',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center'),
        xaxis=dict(showspikes=True, spikemode='across+toaxis', spikethickness=1, spikecolor='grey', spikedash='dot'),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=40, b=100),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font=dict(color='grey')
    )

    # Prepare data info for display based on hover
    if hover_data is not None:
        hover_point_index = hover_data['points'][0]['pointIndex']
        hover_date = combined_data.index[hover_point_index]
        hover_data_info = combined_data.iloc[hover_point_index]
        data_info = [
            html.P(f"Date: {hover_date.date()}"),
            html.P(f"GDP Growth Percentile: {hover_data_info['Growth_Percentile']:.2f}"),
            html.P(f"Employment Percentile: {hover_data_info['Employment_Percentile']:.2f}"),
            html.P(f"Weighted Average Percentile: {hover_data_info['Weighted_Average_Percentile']:.2f}"),
            html.P(f"Raw YoY GDP Growth: {hover_data_info['YoY_GDP_Growth']:.2f}%"),
            html.P(f"Raw Unemployment Rate: {hover_data_info['UNRATE']:.2f}%")
        ]
    else:
        # Default to the latest data if no hover
        latest_data = combined_data.iloc[-1]
        data_info = [
            html.P(f"Date: {combined_data.index[-1].date()}"),
            html.P(f"GDP Growth Percentile: {latest_data['Growth_Percentile']:.2f}"),
            html.P(f"Employment Percentile: {latest_data['Employment_Percentile']:.2f}"),
            html.P(f"Weighted Average Percentile: {latest_data['Weighted_Average_Percentile']:.2f}"),
            html.P(f"Raw YoY GDP Growth: {latest_data['YoY_GDP_Growth']:.2f}%"),
            html.P(f"Raw Unemployment Rate: {latest_data['UNRATE']:.2f}%")
        ]

    return fig, data_info

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
