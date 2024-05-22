import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# FRED API endpoint and your API key
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = "3512f99492b0e9022667d242256548cb"

# Define the series IDs for different maturities
series_ids = {
    '0.25': 'DGS3MO',
    '0.5': 'DGS6MO',
    '1': 'DGS1',
    '5': 'DGS5',
    '10': 'DGS10',
    '30': 'DGS30'
}

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

print(yield_data.head())

def bootstrap_yield_curve_true(yield_data):
    # Assuming yield_data is a DataFrame with date as index and maturities as columns
    maturities = np.array([0.25, 0.5, 1, 5, 10, 30])
    zero_rates = pd.DataFrame(index=yield_data.index, columns=maturities)

    for date in yield_data.index:
        zero_rate_list = []
        for maturity in maturities:
            yield_rate = yield_data.loc[date, f'DGS{int(maturity * 12)}MO' if maturity < 1 else f'DGS{int(maturity)}']
            zero_rate = yield_rate / 100
            zero_rate_list.append(zero_rate)
        zero_rates.loc[date] = zero_rate_list

    return zero_rates

zero_curve_true = bootstrap_yield_curve_true(yield_data)
print(zero_curve_true.head())
print(zero_curve_true.tail())


def bootstrap_yield_curve(yield_data):
    maturities = [0.25, 0.5, 1, 5, 10, 30]
    zero_rates = pd.DataFrame(index=yield_data.index, columns=maturities)

    for date in yield_data.index:
        zero_rate_list = []
        for i, maturity in enumerate(maturities):
            if i == 0:
                zero_rate = yield_data.loc[date, series_ids[str(maturity)]] / 100
            else:
                bond_yield = yield_data.loc[date, series_ids[str(maturity)]] / 100
                bond_price = 100 / (1 + bond_yield * maturity)

                # Calculate the sum of previous cash flows discounted by zero rates
                sum_previous_cashflows = sum([
                    1 / ((1 + zero_rate_list[j]) ** maturities[j])
                    for j in range(i)
                ])

                # Adjust bond price by subtracting sum of previous cash flows
                adjusted_bond_price = bond_price - sum_previous_cashflows

                # Calculate the zero rate for the current maturity
                zero_rate = (adjusted_bond_price / 100) ** (-1 / maturity) - 1

            zero_rate_list.append(zero_rate)
        zero_rates.loc[date] = zero_rate_list

    return zero_rates

zero_curve = bootstrap_yield_curve(yield_data)
print(zero_curve.head())


print(zero_curve.tail())


def interpolate_yields(yield_data, maturity_labels, maturities, all_maturities):
    interpolated_yields = pd.DataFrame(index=yield_data.index, columns=all_maturities)

    for date in yield_data.index:
        values = yield_data.loc[date, maturity_labels].values
        pchip = PchipInterpolator(maturities, values)
        interpolated_yields.loc[date] = pchip(all_maturities)

    return interpolated_yields


original_maturities = [0.25, 0.5, 1, 5, 10, 30]
additional_maturities = [2, 3, 4, 15, 20]
all_maturities = sorted(original_maturities + additional_maturities)

# Interpolate the bootstrapped zero curve
interpolated_zero_curve = interpolate_yields(zero_curve, original_maturities, original_maturities, all_maturities)

# Interpolate the original FRED yield curve using the numeric maturities
interpolated_fred_curve = interpolate_yields(yield_data, list(series_ids.values()), original_maturities, all_maturities)

print(interpolated_zero_curve.head())
print(interpolated_fred_curve.head())

# Plotting the interpolated yield curve for the first date
# Plotting the yield curves for the first date
# Convert the interpolated FRED curve to the same units as the bootstrapped curve (percentages)
interpolated_fred_curve = interpolated_fred_curve / 100

# Plotting the yield curves for the first date
# Create interactive plots and save to HTML
fig = make_subplots(rows=1, cols=1)

for date in interpolated_zero_curve.index:
    fig.add_trace(go.Scatter(x=all_maturities, y=interpolated_zero_curve.loc[date], mode='lines+markers', name=f'Bootstrapped Yield Curve {date.date()}'))
    fig.add_trace(go.Scatter(x=all_maturities, y=interpolated_fred_curve.loc[date], mode='lines+markers', name=f'Interpolated FRED Yield Curve {date.date()}', line=dict(dash='dash')))

fig.update_layout(title='Yield Curves', xaxis_title='Maturity (years)', yaxis_title='Yield (in percent)')
fig.write_html('yield_curves.html')


