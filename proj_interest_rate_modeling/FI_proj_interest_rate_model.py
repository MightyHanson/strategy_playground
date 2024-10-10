import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages
from dotenv import load_dotenv
import os
load_dotenv()  # Loads variables from .env into the environment
# FRED API endpoint and your API key
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = os.getenv("FRED_API_KEY")

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

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

# Function to estimate Vasicek model parameters
def estimate_vasicek_parameters(r):
    dt = 1 / 252  # Assume daily data with 252 trading days in a year
    dr = np.diff(r)

    def vasicek_log_likelihood(params):
        alpha, beta, sigma = params
        r_mean = r[:-1] + alpha * (beta - r[:-1]) * dt
        variance = sigma ** 2 * dt
        regularization = 1e-4 * (alpha**2 + beta**2 + sigma**2)
        log_likelihood = -0.5 * np.sum((dr - r_mean) ** 2 / variance + np.log(variance)) + regularization
        return -log_likelihood

    initial_params = [0.01, np.mean(r), 0.01]
    bounds = [(0, 1), (0, 1), (0, 1)]
    result = minimize(vasicek_log_likelihood, initial_params, bounds=bounds)
    return result.x

# Function to estimate CIR model parameters
def estimate_cir_parameters(r):
    dt = 1 / 252  # Assume daily data with 252 trading days in a year
    dr = np.diff(r)

    def cir_log_likelihood(params):
        alpha, beta, sigma = params
        r_mean = r[:-1] + alpha * (beta - r[:-1]) * dt
        variance = sigma ** 2 * r[:-1] * dt
        regularization = 1e-4 * (alpha**2 + beta**2 + sigma**2)
        log_likelihood = -0.5 * np.sum((dr - r_mean) ** 2 / variance + np.log(variance)) + regularization
        return -log_likelihood

    initial_params = [0.01, np.mean(r), 0.01]
    bounds = [(0, 1), (0, 1), (0, 1)]
    result = minimize(cir_log_likelihood, initial_params, bounds=bounds)
    return result.x

# Function to estimate Hull-White model parameters
def estimate_hull_white_parameters(r):
    dt = 1 / 252  # Assume daily data with 252 trading days in a year
    dr = np.diff(r)

    def hull_white_log_likelihood(params):
        alpha, sigma = params
        r_mean = r[:-1] + alpha * (np.mean(r) - r[:-1]) * dt
        variance = sigma ** 2 * dt
        regularization = 1e-4 * (alpha**2 + sigma**2)
        log_likelihood = -0.5 * np.sum((dr - r_mean) ** 2 / variance + np.log(variance)) + regularization
        return -log_likelihood

    initial_params = [0.01, 0.01]
    bounds = [(0, 1), (0, 1)]
    result = minimize(hull_white_log_likelihood, initial_params, bounds=bounds)
    return result.x

# Function to simulate interest rates using Vasicek model
def vasicek_simulation(alpha, beta, sigma, r0, T, dt):
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = alpha * (beta - rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr
    return rates

# Function to simulate interest rates using CIR model
def cir_simulation(alpha, beta, sigma, r0, T, dt):
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = alpha * (beta - rates[t-1]) * dt + sigma * np.sqrt(rates[t-1] * dt) * np.random.normal()
        rates[t] = max(rates[t-1] + dr, 0)  # Ensure non-negativity
    return rates

# Function to simulate interest rates using Hull-White model
def hull_white_simulation(alpha, sigma, r0, mean_r, T, dt):
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = alpha * (mean_r - rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr
    return rates

# Create a PDF file for charts
pdf_path = output_dir + 'simulated_interest_rates.pdf'
with PdfPages(pdf_path) as pdf:
    for tenor, series_id in series_ids.items():
        # Select the data for the current tenor
        rates = yield_data[series_id].dropna()
        if rates.empty:
            continue

        # Estimate parameters for Vasicek, CIR, and Hull-White models
        alpha_vasicek, beta_vasicek, sigma_vasicek = estimate_vasicek_parameters(rates.values)
        alpha_cir, beta_cir, sigma_cir = estimate_cir_parameters(rates.values)
        alpha_hw, sigma_hw = estimate_hull_white_parameters(rates.values)

        # Simulate interest rates using the calibrated models
        dt = 1 / 252
        T = len(rates) * dt
        mean_r = np.mean(rates.values)
        simulated_rates_vasicek = vasicek_simulation(alpha_vasicek, beta_vasicek, sigma_vasicek, rates.values[0], T, dt)
        simulated_rates_cir = cir_simulation(alpha_cir, beta_cir, sigma_cir, rates.values[0], T, dt)
        simulated_rates_hw = hull_white_simulation(alpha_hw, sigma_hw, rates.values[0], mean_r, T, dt)

        # Plot the actual vs simulated interest rates
        plt.figure(figsize=(14, 7))
        plt.plot(rates.index, rates.values, label='Actual Rates')
        plt.plot(rates.index, simulated_rates_vasicek, label='Simulated Rates (Vasicek)', linestyle='--')
        plt.plot(rates.index, simulated_rates_cir, label='Simulated Rates (CIR)', linestyle='--')
        plt.plot(rates.index, simulated_rates_hw, label='Simulated Rates (Hull-White)', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Interest Rate (%)')
        plt.title(f'Actual vs Simulated Interest Rates - {tenor} Years')
        plt.legend()
        pdf.savefig()
        plt.close()

print(f"Charts have been saved to {pdf_path}")
