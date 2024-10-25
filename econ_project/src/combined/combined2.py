import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

# Load the GDP data
file_path = '../../data/gdp/GDP_Growth_Analysis_Post60.csv'
gdp_data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')

# Extract relevant columns for plotting
gdp_data = gdp_data[(gdp_data.index >= '1960-01-01') & (gdp_data.index < '2024-07-01')]
gdp_data['YoY_GDP_Growth'] = pd.to_numeric(gdp_data['YoY_GDP_Growth'], errors='coerce')

# Drop NaN values
gdp_data.dropna(subset=['YoY_GDP_Growth'], inplace=True)

# Calculate growth percentiles
gdp_data['Growth_Percentile'] = gdp_data['YoY_GDP_Growth'].rank(pct=True)

# Load the unemployment data
unemployment_file_path = '../../data/unemployment/UNRATE.csv'
unemployment_data = pd.read_csv(unemployment_file_path, parse_dates=['DATE'], index_col='DATE')

# Filter data after 1960
unemployment_data = unemployment_data[(unemployment_data.index >= '1960-01-01') & (unemployment_data.index < '2024-07-01')]

# Calculate employment rate and percentiles for unemployment data
unemployment_data['Employment_Rate'] = 100 - unemployment_data['UNRATE']
unemployment_data['Employment_Percentile'] = unemployment_data['Employment_Rate'].rank(pct=True)

# Load the inflation data
inflation_file_path = '../../data/inflation/inflation_updated.csv'
inflation_data = pd.read_csv(inflation_file_path, parse_dates=['DATE'], index_col='DATE')

# Filter data after 1960
inflation_data = inflation_data[(inflation_data.index >= '1960-01-01') & (inflation_data.index < '2024-07-01')]

# Rename "Quarter Average" to "Inflation" if necessary
if 'Quarter Average' in inflation_data.columns:
    inflation_data.rename(columns={'Quarter Average': 'Inflation'}, inplace=True)

# Calculate 100 - inflation rate and percentiles for inflation data
inflation_data['Adjusted_Inflation'] = 100 - inflation_data['Inflation']
inflation_data['Inflation_Percentile'] = inflation_data['Adjusted_Inflation'].rank(pct=True)

# Resample data to quarterly frequency and calculate mean values
gdp_quarterly = gdp_data.resample('QE-DEC').mean()
unemployment_quarterly = unemployment_data.resample('QE-DEC').mean()
inflation_quarterly = inflation_data.resample('QE-DEC').mean()

# Merge GDP, unemployment, and inflation data for averaging
combined_data = pd.merge(gdp_quarterly[['YoY_GDP_Growth', 'Growth_Percentile']], unemployment_quarterly[['UNRATE', 'Employment_Percentile']], left_index=True, right_index=True, how='outer')
combined_data = pd.merge(combined_data, inflation_quarterly[['Inflation', 'Inflation_Percentile', 'Adjusted_Inflation']], left_index=True, right_index=True, how='outer')

# Fill missing values after merging
combined_data.ffill(inplace=True)  # Forward fill to propagate previous values
combined_data.fillna(0, inplace=True)  # Fill remaining NaN with 0 if no previous value exists

# President data with party affiliation
presidents = [
    {'start': '1961-01-20', 'end': '1963-11-22', 'name': 'John F. Kennedy'},
    {'start': '1963-11-22', 'end': '1969-01-20', 'name': 'Lyndon B. Johnson'},
    {'start': '1969-01-20', 'end': '1974-08-09', 'name': 'Richard Nixon'},
    {'start': '1974-08-09', 'end': '1977-01-20', 'name': 'Gerald Ford'},
    {'start': '1977-01-20', 'end': '1981-01-20', 'name': 'Jimmy Carter'},
    {'start': '1981-01-20', 'end': '1989-01-20', 'name': 'Ronald Reagan'},
    {'start': '1989-01-20', 'end': '1993-01-20', 'name': 'George H. W. Bush'},
    {'start': '1993-01-20', 'end': '2001-01-20', 'name': 'Bill Clinton'},
    {'start': '2001-01-20', 'end': '2009-01-20', 'name': 'George W. Bush'},
    {'start': '2009-01-20', 'end': '2017-01-20', 'name': 'Barack Obama'},
    {'start': '2017-01-20', 'end': '2021-01-20', 'name': 'Donald Trump'},
    {'start': '2021-01-20', 'end': '2025-01-20', 'name': 'Joe Biden'}
]

# Function to calculate area under the curve (AUC) for each president
def calculate_auc(data, start, end):
    president_data = data.loc[start:end]
    auc = np.trapezoid(president_data['Weighted_Average_Percentile'], dx=1)
    quarters_in_office = len(data.loc[start:end].resample('QE-DEC').mean())
    normalized_auc = auc / max(quarters_in_office, 1)
    return normalized_auc

# Function to calculate the total change in weighted percentile for each president
def calculate_total_change(data, start, end):
    president_data = data.loc[start:end]['Weighted_Average_Percentile']
    total_change = president_data.iloc[-1] - president_data.iloc[0]
    quarters_in_office = len(president_data.resample('QE-DEC').mean())
    normalized_total_change = (total_change / max(quarters_in_office, 1)) * 20
    return normalized_total_change


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("GDP Growth, Employment, and Inflation Percentile Analysis", style={'color': 'grey'}),
    html.Div([
        html.Div([
            html.Label('GDP Weight:', style={'color': 'grey', 'margin-right': '10px'}),
            dcc.Slider(
                id='gdp-weight-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.33,
                marks={i / 10: f'{i / 10}' for i in range(0, 11)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.Label('Employment Weight:', style={'color': 'grey', 'margin-right': '10px', 'margin-top': '20px'}),
            dcc.Slider(
                id='employment-weight-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.33,
                marks={i / 10: f'{i / 10}' for i in range(0, 11)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.Label('Inflation Weight:', style={'color': 'grey', 'margin-right': '10px', 'margin-top': '20px'}),
            dcc.Slider(
                id='inflation-weight-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.33,
                marks={i / 10: f'{i / 10}' for i in range(0, 11)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            )
        ], style={'display': 'inline-block', 'width': '30%', 'vertical-align': 'top', 'padding': '10px'}),
        html.Div(id='president-info', style={'display': 'inline-block', 'backgroundColor': '#333', 'color': 'grey', 'padding': '10px', 'border': '1px solid #444', 'width': '30%', 'margin-left': '10px', 'vertical-align': 'top'}),
        html.Div(id='data-info', style={'display': 'inline-block', 'backgroundColor': '#333', 'color': 'grey', 'padding': '10px', 'border': '1px solid #444', 'width': '30%', 'margin-left': '10px', 'vertical-align': 'top'})
    ], style={'width': '100%', 'padding': '20px 0'}),
    dcc.Graph(id='percentile-plot', style={'height': '75vh'}),
    html.Label('Total Change vs AUC Weight:', style={'color': 'grey', 'margin-top': '20px'}),
    dcc.Slider(
        id='auc-total-change-weight-slider',
        min=0,
        max=1,
        step=0.01,
        value=0.5,
        marks={i / 10: f'{i / 10}' for i in range(0, 11)},
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),
    html.H2("Normalized Area Under Curve for Each President", style={'color': 'grey', 'margin-top': '40px'}),
    html.Table(id='auc-table', style={'width': '100%', 'border': '1px solid #444', 'margin-top': '20px', 'backgroundColor': '#333', 'color': 'grey', 'border-collapse': 'collapse'})
], style={'backgroundColor': '#121212', 'padding': '20px'})

# Callback to update the plot, display data information, and update AUC table
@app.callback(
    [Output('percentile-plot', 'figure'),
     Output('data-info', 'children'),
     Output('president-info', 'children'),
     Output('auc-table', 'children')],
    [Input('gdp-weight-slider', 'value'),
     Input('employment-weight-slider', 'value'),
     Input('inflation-weight-slider', 'value'),
     Input('auc-total-change-weight-slider', 'value'),
     Input('percentile-plot', 'hoverData')]
)
def update_plot(gdp_weight, employment_weight, inflation_weight, auc_total_change_weight, hover_data):
    # Normalize weights if they don't sum to 1
    total_weight = gdp_weight + employment_weight + inflation_weight
    if total_weight > 0:
        gdp_weight /= total_weight
        employment_weight /= total_weight
        inflation_weight /= total_weight

    combined_data['Weighted_Average_Percentile'] = (
        gdp_weight * combined_data['Growth_Percentile'] + 
        employment_weight * combined_data['Employment_Percentile'] + 
        inflation_weight * combined_data['Inflation_Percentile']
    )

    # Create a time plot
    fig = go.Figure()

    # Add president background shading
    for president in presidents:
        fig.add_vrect(
            x0=president['start'], x1=president['end'],
            fillcolor='blue' if president['name'] in ['Jimmy Carter', 'Bill Clinton', 'Barack Obama', 'Joe Biden', 'John F. Kennedy', 'Lyndon B. Johnson'] else 'red',
            opacity=0.1, layer='below', line_width=0
        )

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
    if hover_data is not None and 'points' in hover_data and len(hover_data['points']) > 0:
        hover_point_index = hover_data['points'][0]['pointIndex']
        hover_date = combined_data.index[hover_point_index]
        hover_data_info = combined_data.iloc[hover_point_index]
    else:
        # Default to the latest data if no hover
        hover_data_info = combined_data.iloc[-1]
        hover_date = combined_data.index[-1]

    data_info = [
        html.P(f"Date: {hover_date.date()}"),
        html.P(f"GDP Growth Percentile: {hover_data_info['Growth_Percentile']:.2f}"),
        html.P(f"Employment Percentile: {hover_data_info['Employment_Percentile']:.2f}"),
        html.P(f"Inflation Percentile: {hover_data_info['Inflation_Percentile']:.2f}"),
        html.P(f"Weighted Average Percentile: {hover_data_info['Weighted_Average_Percentile']:.2f}"),
        html.P(f"Raw YoY GDP Growth: {hover_data_info['YoY_GDP_Growth']:.2f}%"),
        html.P(f"Raw Unemployment Rate: {hover_data_info['UNRATE']:.2f}%"),
        html.P(f"Raw Inflation Rate: {hover_data_info['Inflation']:.2f}%")
    ]

    # Determine which president was in office at the hover date
    president_info = "Unknown"
    for president in presidents:
        if pd.to_datetime(president['start']) <= hover_date <= pd.to_datetime(president['end']):
            president_info = f"President: {president['name']}"
            break
    president_info_box = [html.P(president_info)]

    # Calculate AUC and total change for each president based on updated weights
    president_aucs = []
    for president in presidents:
        start = pd.to_datetime(president['start'])
        end = pd.to_datetime(president['end'])
        auc_value = calculate_auc(combined_data, start, end)
        total_change = calculate_total_change(combined_data, start, end)
        combined_value = (auc_total_change_weight * auc_value) + ((1 - auc_total_change_weight) * total_change)
        president_aucs.append(
            {'name': president['name'], 'auc': auc_value, 'total_change': total_change, 'combined_value': combined_value}
        )

    # Sort presidents by combined value
    president_aucs = sorted(president_aucs, key=lambda x: x['combined_value'], reverse=True)

    # Create table rows for each president
    table_header = html.Tr([html.Th("President"), html.Th("AUC"), html.Th("Total Change"), html.Th("Combined Value")])
    table_rows = [
        html.Tr([html.Td(president['name']),
                 html.Td(f"{president['auc']:.2f}"),
                 html.Td(f"{president['total_change']:.2f}"),
                 html.Td(f"{president['combined_value']:.2f}")])
        for president in president_aucs
    ]

    auc_table = [table_header] + table_rows

    return fig, data_info, president_info_box, auc_table

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
