import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# FRED API endpoint and your API key
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = "3512f99492b0e9022667d242256548cb"
output_dir = 'F:/FIC_project/output/'
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

# Select the 10-year Treasury yield data
rates = yield_data['DGS10'].dropna()

# Calculate the changes in interest rates
dt = 1 / 252  # Assume daily data with 252 trading days in a year
r = rates.values.flatten()
dr = np.diff(r)

# Log-likelihood function for the Vasicek model with regularization
def vasicek_log_likelihood(params):
    alpha, beta, sigma = params
    r_mean = r[:-1] + alpha * (beta - r[:-1]) * dt
    variance = sigma ** 2 * dt
    regularization = 1e-4 * (alpha**2 + beta**2 + sigma**2)
    log_likelihood = -0.5 * np.sum((dr - r_mean) ** 2 / variance + np.log(variance)) + regularization
    return -log_likelihood

# Initial parameter guesses 
initial_params = [0.01, np.mean(r), 0.01]
bounds = [(0, 1), (0, 1), (0, 1)]

# Minimize the negative log-likelihood
result = minimize(vasicek_log_likelihood, initial_params, bounds=bounds)
alpha_vasicek, beta_vasicek, sigma_vasicek = result.x

print(f"Estimated parameters for Vasicek model: alpha={alpha_vasicek}, beta={beta_vasicek}, sigma={sigma_vasicek}")

# Log-likelihood function for the CIR model with regularization
# Improved log-likelihood function for the CIR model with regularization
def cir_log_likelihood(params):
    alpha, beta, sigma = params
    r_mean = r[:-1] + alpha * (beta - r[:-1]) * dt
    variance = sigma ** 2 * r[:-1] * dt
    regularization = 1e-4 * (alpha**2 + beta**2 + sigma**2)
    log_likelihood = -0.5 * np.sum((dr - r_mean) ** 2 / variance + np.log(variance)) + regularization
    return -log_likelihood

# Initial parameter guesses
initial_params = [0.01, np.mean(r), 0.01]

# Minimize the negative log-likelihood
result = minimize(cir_log_likelihood, initial_params, bounds=bounds)
alpha_cir, beta_cir, sigma_cir = result.x

print(f"Estimated parameters for CIR model: alpha={alpha_cir}, beta={beta_cir}, sigma={sigma_cir}")

# Log-likelihood function for the Hull-White model with regularization
def hull_white_log_likelihood(params):
    alpha, sigma = params
    r_mean = r[:-1] + alpha * (np.mean(r) - r[:-1]) * dt
    variance = sigma ** 2 * dt
    regularization = 1e-4 * (alpha**2 + sigma**2)
    log_likelihood = -0.5 * np.sum((dr - r_mean) ** 2 / variance + np.log(variance)) + regularization
    return -log_likelihood

# Improved initial parameter guesses
initial_params = [0.01, 0.01]

# Minimize the negative log-likelihood
result = minimize(hull_white_log_likelihood, initial_params, bounds=[(0, 1), (0, 1)])
alpha_hw, sigma_hw = result.x

print(f"Estimated parameters for Hull-White model: alpha={alpha_hw}, sigma={sigma_hw}")

# Vasicek Model Simulation
def vasicek_simulation(alpha, beta, sigma, r0, T, dt):
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = alpha * (beta - rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr
    return rates

# CIR Model Simulation
def cir_simulation(alpha, beta, sigma, r0, T, dt):
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = alpha * (beta - rates[t-1]) * dt + sigma * np.sqrt(rates[t-1] * dt) * np.random.normal()
        rates[t] = max(rates[t-1] + dr, 0)  # Ensure non-negativity
    return rates

# Hull-White Model Simulation
def hull_white_simulation(alpha, sigma, r0, T, dt):
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = alpha * (np.mean(r) - rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr
    return rates

# Simulate interest rates using the calibrated models
simulated_rates_vasicek = vasicek_simulation(alpha_vasicek, beta_vasicek, sigma_vasicek, r[0], len(r) * dt, dt)
simulated_rates_cir = cir_simulation(alpha_cir, beta_cir, sigma_cir, r[0], len(r) * dt, dt)
simulated_rates_hw = hull_white_simulation(alpha_hw, sigma_hw, r[0], len(r) * dt, dt)

# Convert to DataFrame for saving to Excel
simulation_df = pd.DataFrame({
    'Actual Rates': r,
    'Simulated Rates (Vasicek)': simulated_rates_vasicek,
    'Simulated Rates (CIR)': simulated_rates_cir,
    'Simulated Rates (Hull-White)': simulated_rates_hw
}, index=rates.index)

# Save to Excel
simulation_df.to_excel(output_dir + 'simulated_interest_rates.xlsx')

# Plot the actual vs simulated interest rates
plt.figure(figsize=(14, 7))

plt.plot(rates.index, r, label='Actual Rates')
plt.plot(rates.index, simulated_rates_vasicek, label='Simulated Rates (Vasicek)', linestyle='--')
plt.plot(rates.index, simulated_rates_cir, label='Simulated Rates (CIR)', linestyle='--')
plt.plot(rates.index, simulated_rates_hw, label='Simulated Rates (Hull-White)', linestyle='--')

plt.xlabel('Year')
plt.ylabel('Interest Rate (%)')
plt.title('Actual vs Simulated Interest Rates')
plt.legend()
plt.show()