import pandas as pd

# Load the CSV file
df = pd.read_csv('../data/GDPC1.csv')

# Convert 'DATE' to datetime for better handling
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter data to only include entries starting from the 1980s
df = df[df['DATE'] >= '1960-01-01']

# Calculate the quarterly GDP growth rate
df['GDP_Growth'] = df['Real_GDP'].pct_change() * 100

# Calculate Year-over-Year Growth Rate (same quarter last year)
df['YoY_GDP_Growth'] = df['Real_GDP'].pct_change(periods=4) * 100

# Drop rows with NaN values due to pct_change calculations
df.dropna(subset=['GDP_Growth'], inplace=True)

# Convert the growth rate to percentiles, including only data from the 1980s onward
df['Growth_Percentile'] = df['GDP_Growth'].rank(pct=True) * 100
df['YoY_Growth_Percentile'] = df['YoY_GDP_Growth'].rank(pct=True) * 100

# Calculate Annualized Quarterly Growth Rate
df['Annualized_GDP_Growth'] = ((1 + (df['GDP_Growth'] / 100)) ** 4 - 1) * 100

# Display the first few rows of the DataFrame
print(df.head())

# Optionally, save the result to a new CSV file for further inspection
df.to_csv('../data/GDP_Growth_Analysis_Post60.csv', index=False)
