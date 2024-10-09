import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick
import os
from scipy.optimize import minimize
from tqdm import tqdm

# Set up output directory
output_dir = 'F:/strategy_playground/proj_yield_curve_construction/output/'
os.makedirs(output_dir, exist_ok=True)

# FRED API endpoint and API key (consider using environment variables for better security)
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = os.getenv('FRED_API_KEY', "3512f99492b0e9022667d242256548cb")

# Define the series IDs for different maturities
series_ids = {
    '0.0833': 'DGS1MO',
    '0.25': 'DGS3MO',
    '0.5': 'DGS6MO',
    '1': 'DGS1',
    '2': 'DGS2',
    '3': 'DGS3',
    '5': 'DGS5',
    '7': 'DGS7',
    '10': 'DGS10',
    '20': 'DGS20',
    '30': 'DGS30'
}
series_key_list = [float(x) if float(x) < 1 else int(x) for x in series_ids.keys()]
# Function to fetch data from FRED
def fetch_data(series_id, api_key):
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json'
    }
    response = requests.get(FRED_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()['observations']
        return pd.DataFrame(data)[['date', 'value']].rename(columns={'value': series_id})
    else:
        print(f"Error fetching data for series {series_id}: {response.status_code}")
        return pd.DataFrame(columns=['date', series_id])


# Fetch data for all series and merge into a single DataFrame
dfs = [fetch_data(series_id, API_KEY) for series_id in series_ids.values()]
yield_data = dfs[0]
for df in dfs[1:]:
    yield_data = yield_data.merge(df, on='date')

yield_data['date'] = pd.to_datetime(yield_data['date'])
yield_data.set_index('date', inplace=True)

# Replace dots with NaN and convert to float
yield_data = yield_data.apply(pd.to_numeric, errors='coerce')

# Fill missing values using forward fill method
yield_data.fillna(method='ffill', inplace=True)
yield_data.to_excel(output_dir + 'yield_data.xlsx', index=False)


# Function for bootstrapping zero rates with continuous compounding
def bootstrap_zero_curve_continuous(yield_data, maturities):
    import numpy as np
    from scipy.optimize import brentq
    zero_rates = pd.DataFrame(index=yield_data.index, columns=maturities)

    m = 2  # Semi-annual coupon payments

    for date in yield_data.index:
        spot_rates = {}
        for i, maturity in enumerate(maturities):
            c_semi = yield_data.loc[date, series_ids[str(maturity)]] / 100  # Par yield in decimal

            if maturity <= 0.5:
                # For maturities <= 6 months, treat as zero-coupon bond
                rate = (1 / maturity) * np.log(1 + c_semi * maturity)
                spot_rates[maturity] = rate
            else:
                # Number of coupon payments, using ceil to ensure at least one payment
                n_payments = int(np.ceil(maturity * m))
                # Times of cash flows
                payment_times = np.array([(j + 1) / m for j in range(n_payments)])
                # Clip payment times to maturity
                payment_times = np.clip(payment_times, None, maturity)
                # Cash flows
                coupon_payment = c_semi / m
                cash_flows = np.full(n_payments, coupon_payment)
                cash_flows[-1] += 1  # Add principal to the last payment

                # Define function to find the zero rate
                def bond_price_zero_rate(r):
                    # For continuous compounding
                    discount_factors = np.exp(-r * payment_times)
                    pv = np.sum(cash_flows * discount_factors)
                    return pv - 1  # Bond price is par (1)

                # Use previous rate as initial guess
                previous_rate = spot_rates.get(maturities[i - 1], 0.05)
                initial_guess = previous_rate

                # Solve for zero rate
                try:
                    zero_rate = brentq(bond_price_zero_rate, -0.1, 0.5)
                    spot_rates[maturity] = zero_rate
                except ValueError:
                    spot_rates[maturity] = np.nan

        # Store the zero rates for this date
        zero_rates.loc[date] = [spot_rates.get(m, np.nan) for m in maturities]

    return zero_rates

# Setup the maturities
original_maturities = series_key_list
additional_maturities = [4, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 28]
all_maturities = sorted(original_maturities + additional_maturities)

# Bootstrap the zero curve with continuous compounding
zero_curve_continuous = bootstrap_zero_curve_continuous(yield_data, original_maturities)
zero_curve_continuous.to_excel(output_dir + 'zero_curve_data_continuous.xlsx', index=False)

# Function for interpolation using PCHIP
def interpolate_yields(yield_data, maturity_labels, maturities, all_maturities):
    interpolated_yields = pd.DataFrame(index=yield_data.index, columns=all_maturities)

    for date in yield_data.index:
        try:
            values = yield_data.loc[date, maturity_labels].values
            pchip = PchipInterpolator(maturities, values)
            interpolated_yields.loc[date] = pchip(all_maturities)
        except ValueError as e:
            print(f"Error interpolating yields for date {date}: {e}")

    return interpolated_yields

# Interpolate the original FRED yield curve using the numeric maturities
interpolated_fred_curve = interpolate_yields(yield_data, list(series_ids.values()), original_maturities, all_maturities)
interpolated_fred_curve.to_excel(output_dir + 'interpolate_yield_data.xlsx', index=False)

# Convert the interpolated FRED curve to the same units as the bootstrapped curve (percentages)
interpolated_fred_curve = interpolated_fred_curve / 100

# Calibration Function
def cost_function(params, market_data, maturities):
    # Use PCHIP to create an interpolated curve based on current parameters
    pchip = PchipInterpolator(maturities, params)
    model_curve = pchip(maturities)
    # Calculate the sum of squared differences
    error = np.sum((market_data - model_curve) ** 2)
    return error

def calibrate_curve(yield_data, original_maturities):
    optimized_curves = pd.DataFrame(index=yield_data.index, columns=original_maturities)

    for date_idx in range(1, len(yield_data)):
        initial_guess = yield_data.iloc[date_idx - 1].values  # Use the previous day's curve as the starting point
        market_data_next_day = yield_data.iloc[date_idx].values

        # Optimize the curve parameters to minimize the cost function
        result = minimize(
            cost_function,
            initial_guess,
            args=(market_data_next_day, original_maturities),
            method='L-BFGS-B'
        )

        if result.success:
            optimized_curves.iloc[date_idx] = result.x
        else:
            print(f"Optimization failed for date {yield_data.index[date_idx]}: {result.message}")
            optimized_curves.iloc[date_idx] = initial_guess

    return optimized_curves

# Implement
optimized_curves = calibrate_curve(zero_curve_continuous, original_maturities)
optimized_curves.to_excel(output_dir + 'optimized_curves.xlsx', index=False)

# Save plots to PDF with progress bar
with PdfPages(output_dir + 'yield_curves.pdf') as pdf:
    for date in tqdm(interpolated_fred_curve.index, desc="Generating yield curve plots"):
        plt.figure(figsize=(12, 8))
        plt.plot(all_maturities, interpolated_fred_curve.loc[date], marker='x', linestyle='--',
                 label='Interpolated FRED Yield Curve')
        if date in optimized_curves.index:
            plt.plot(original_maturities, optimized_curves.loc[date], marker='o', linestyle='-',
                     label='Optimized Yield Curve')
        plt.title(f'Yield Curves on {date.date()}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield (in percent)')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()