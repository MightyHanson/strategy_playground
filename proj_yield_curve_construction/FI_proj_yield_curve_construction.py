import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick
# FRED API endpoint and your API key
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = "3512f99492b0e9022667d242256548cb"
output_dir = 'F:/FIC_project/output/'
# Define the series IDs for different maturities
series_ids = {
    '0.0833': 'DGS1MO',
    '0.25': 'DGS3MO',      # 3-Month Treasury Constant Maturity Rate
    '0.5': 'DGS6MO',
    '1': 'DGS1',           # 1-Year Treasury Constant Maturity Rate
    '2': 'DGS2',
    '3': 'DGS3',
    '5': 'DGS5',           # 5-Year Treasury Constant Maturity Rate
    '7': 'DGS7',
    '10': 'DGS10',         # 10-Year Treasury Constant Maturity Rate
    '20': 'DGS20',
    '30': 'DGS30'          # 30-Year Treasury Constant Maturity Rate
}

series_key = list(series_ids.keys())
series_key_list = [float(x) if float(x) < 1 else int(x) for x in series_key  ]
# Function to fetch data from FRED
def fetch_data(series_id, api_key):
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json'
    }
    response = requests.get(FRED_API_URL, params=params)
    data = response.json()['observations']
    return pd.DataFrame(data)[['date', 'value']].rename(columns={'value': series_id})

# Fetch data for all series and merge into a single DataFrame
dfs = [fetch_data(series_id, API_KEY) for series_id in series_ids.values()]
yield_data = dfs[0]
for df in dfs[1:]:
    yield_data = yield_data.merge(df, on='date')

yield_data['date'] = pd.to_datetime(yield_data['date'])
yield_data.set_index('date', inplace=True)

# Replace dots with NaN and convert to float
yield_data.replace('.', np.nan, inplace=True)
yield_data = yield_data.astype(float)

# Fill missing values using forward fill method
yield_data.fillna(method='ffill', inplace=True)
yield_data.to_excel(output_dir + 'yield_data.xlsx', index = False)
# print(yield_data.head())

# print(zero_curve.head())
# print(zero_curve.tail())


def interpolate_yields(yield_data, maturity_labels, maturities, all_maturities):
    interpolated_yields = pd.DataFrame(index=yield_data.index, columns=all_maturities)

    for date in yield_data.index:
        values = yield_data.loc[date, maturity_labels].values
        pchip = PchipInterpolator(maturities, values)
        interpolated_yields.loc[date] = pchip(all_maturities)

    return interpolated_yields


original_maturities = series_key_list
additional_maturities = [4,6,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,28]
all_maturities = sorted(original_maturities + additional_maturities)

# Interpolate the bootstrapped zero curve
# interpolated_zero_curve = interpolate_yields(zero_curve, original_maturities, original_maturities, all_maturities)

# Interpolate the original FRED yield curve using the numeric maturities
interpolated_fred_curve = interpolate_yields(yield_data, list(series_ids.values()), original_maturities, all_maturities)
interpolated_fred_curve.to_excel(output_dir + 'interpolate_yield_data.xlsx', index = False)
# print(interpolated_zero_curve.head())
print(interpolated_fred_curve.head())

# Plotting the interpolated yield curve for the first date
# Plotting the yield curves for the first date
# Convert the interpolated FRED curve to the same units as the bootstrapped curve (percentages)
interpolated_fred_curve = interpolated_fred_curve / 100

# Plotting the yield curves for the first date
# Create interactive plots and save to HTML
fig = make_subplots(rows=1, cols=1)

# output via html
# for date in interpolated_zero_curve.index:
#     fig.add_trace(go.Scatter(x=all_maturities, y=interpolated_zero_curve.loc[date]*100, mode='lines+markers', name=f'Bootstrapped Yield Curve {date.date()}'))
#     fig.add_trace(go.Scatter(x=all_maturities, y=interpolated_fred_curve.loc[date]*100, mode='lines+markers', name=f'Interpolated FRED Yield Curve {date.date()}', line=dict(dash='dash')))
#
# fig.update_layout(title='Yield Curves', xaxis_title='Maturity (years)', yaxis_title='Yield (in percent)', yaxis_tickformat='.1%')
# fig.write_html(output_dir + 'yield_curves.html')

# Save plots to PDF
with PdfPages(output_dir + 'yield_curves.pdf') as pdf:
    for date in interpolated_fred_curve.index:
        plt.figure(figsize=(12, 8))
        # plt.plot(all_maturities, interpolated_fred_curve.loc[date], marker='o', label='Bootstrapped Yield Curve')
        plt.plot(all_maturities, interpolated_fred_curve.loc[date], marker='x', linestyle='--', label='Interpolated FRED Yield Curve')
        plt.title(f'Yield Curves on {date.date()}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield (in percent)')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()


