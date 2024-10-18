# preprocess.py

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader
import pickle
import logging
from pypfopt import expected_returns, risk_models
from sklearn.covariance import LedoitWolf

SEQ_LENGTH = 30  # Sequence length for transformer input

def create_sequences(data, seq_length):
    """
    Create sequences of data for transformer input.

    Args:
        data (np.ndarray): Array of feature data.
        seq_length (int): Length of each sequence.

    Returns:
        tuple: (sequences, targets)
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def load_and_prepare_data(data_loader, symbols):
    """
    Load data for all symbols and prepare for model training.

    Args:
        data_loader (DataLoader): Instance of DataLoader.
        symbols (list): List of symbols to load data for.

    Returns:
        pd.DataFrame: Combined DataFrame containing all symbols' data.
    """
    all_data = []
    for symbol in tqdm(symbols, desc="Loading data for all symbols"):
        df = data_loader.load_data(symbol)
        if df is not None:
            all_data.append(df)
    if not all_data:
        print("No data loaded. Exiting.")
        sys.exit(1)
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def preprocess_data(combined_df, data_loader):
    """
    Preprocess the combined data for model training.

    Args:
        combined_df (pd.DataFrame): Combined DataFrame containing all symbols' data.
        data_loader (DataLoader): Instance of DataLoader.

    Returns:
        tuple: (X, y, scaler, expected_returns, cov_matrix)
            - X (np.ndarray): Feature sequences.
            - y (np.ndarray): Targets.
            - scaler (StandardScaler): Fitted scaler object.
            - expected_returns (pd.Series): Expected returns for each asset.
            - cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    """


    # Ensure correct data types
    # Filter returns_clean to your dataset's date range
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
    combined_df['symbol'] = combined_df['symbol'].str.strip()
    # Filter data to your date range

    # Convert numeric columns to numeric types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        # Drop rows with NaN in numeric columns
    combined_df.dropna(subset=numeric_columns, inplace=True)

    # Remove rows with non-positive 'close' prices
    combined_df = combined_df[combined_df['close'] > 0]

    # Replace infinite values with NaN and drop them
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(subset=numeric_columns, inplace=True)


    # Check and remove duplicates
    duplicate_count = combined_df.duplicated(subset=['symbol', 'date']).sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Removing duplicates.")
        logging.info(f"Found {duplicate_count} duplicate rows. Removing duplicates.")
        combined_df = combined_df.drop_duplicates(subset=['symbol', 'date'])

    # Pivot the data to have symbols as separate columns
    try:
        pivot_df = combined_df.pivot(index='date', columns='symbol', values='close').sort_index()
    except ValueError as ve:
        print("Error during pivoting:", ve)
        logging.error(f"Error during pivoting: {ve}")
        sys.exit(1)

    # Handle the FutureWarning by specifying fill_method=None
    # Optionally, fill missing values if necessary
    pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill
    returns_df = pivot_df.pct_change(fill_method=None).dropna()

    # Calculate expected returns and covariance matrix
    # expected_returns = returns_df.mean()
    # cov_matrix = returns_df.cov()

    returns = pivot_df.pct_change().dropna()
    # Drop assets (columns) that contain any NaN values
    returns_clean = returns.dropna(axis=1, how='any')

    # Log the number of assets before and after cleaning
    num_assets_before = returns.shape[1]
    num_assets_after = returns_clean.shape[1]
    logging.info(f"Number of assets before cleaning: {num_assets_before}")
    logging.info(f"Number of assets after removing assets with NaN returns: {num_assets_after}")

    # Remove columns with NaN values
    cols_with_nan = returns_clean.columns[returns_clean.isnull().any()].tolist()
    if cols_with_nan:
        logging.info(f"Removing assets with NaN returns: {cols_with_nan}")
        returns_clean.drop(columns=cols_with_nan, inplace=True)

    # Remove columns with infinite values
    cols_with_inf = returns_clean.columns[np.isinf(returns_clean).any()].tolist()
    if cols_with_inf:
        logging.info(f"Removing assets with infinite returns: {cols_with_inf}")
        returns_clean.drop(columns=cols_with_inf, inplace=True)

    # Verify again that returns_clean has no NaN or infinite values
    if returns_clean.isnull().values.any():
        logging.error("NaN values still present after removing problematic assets.")
        raise ValueError("Data cleaning failed to remove all NaN values.")

    if np.isinf(returns_clean.values).any():
        logging.error("Infinite values still present after removing problematic assets.")
        raise ValueError("Data cleaning failed to remove all infinite values.")

    # Update expected_returns and cov_matrix calculations
    expected_return = returns_clean.mean()
    # cov_matrix = risk_models.CovarianceShrinkage(returns_clean).ledoit_wolf()
    scaler = StandardScaler()
    returns_scaled = pd.DataFrame(scaler.fit_transform(returns_clean), index=returns_clean.index,
                                  columns=returns_clean.columns)

    lw = LedoitWolf()
    try:
        lw.fit(returns_scaled)
        cov_matrix = pd.DataFrame(lw.covariance_, index=returns_clean.columns, columns=returns_clean.columns)
    except Exception as e:
        print(f"Ledoit-Wolf estimator failed: {e}")
        # Fallback to sample covariance
        cov_matrix = returns_clean.cov()

    # Check for NaN or infinite values in cov_matrix
    if cov_matrix.isnull().values.any() or np.isinf(cov_matrix.values).any():
        print("Covariance matrix contains NaN or infinite values.")
        # Handle the issue: e.g., remove problematic assets or regularize the matrix
        # For example, remove assets causing issues
        cols_with_nan = cov_matrix.columns[cov_matrix.isnull().any()].tolist()
        cols_with_inf = cov_matrix.columns[np.isinf(cov_matrix).any()].tolist()
        cols_to_remove = list(set(cols_with_nan + cols_with_inf))
        cov_matrix.drop(index=cols_to_remove, columns=cols_to_remove, inplace=True)
        returns_clean.drop(columns=cols_to_remove, inplace=True)
        # Recompute covariance matrix
        cov_matrix = returns_clean.cov()

    # Prepare data for each symbol
    feature_dfs = []
    target_dfs = []
    scalers = {}

    for symbol in tqdm(pivot_df.columns, desc="Preprocessing data for each symbol"):
        symbol_data = combined_df[combined_df['symbol'] == symbol].sort_values('date')
        symbol_data = symbol_data[['date', 'open', 'high', 'low', 'close', 'volume']]
        symbol_data = symbol_data.reset_index(drop=True)

        # Feature Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(symbol_data[['open', 'high', 'low', 'close', 'volume']])
        symbol_data[['open', 'high', 'low', 'close', 'volume']] = scaled_features
        scalers[symbol] = scaler

        # Create sequences
        X, y = create_sequences(symbol_data[['open', 'high', 'low', 'close', 'volume']].values, SEQ_LENGTH)
        if len(X) == 0:
            logging.warning(f"No sequences created for symbol: {symbol}. Skipping.")
            continue
        feature_dfs.append(X)
        target_dfs.append(y[:, 3])  # Assuming 'close' price is the target

    # Stack all symbols' data
    if not feature_dfs or not target_dfs:
        print("No sequences created. Exiting.")
        logging.error("No sequences created. Exiting.")
        sys.exit(1)

    X = np.vstack(feature_dfs)
    y = np.hstack(target_dfs)

    # Save a single scaler for simplicity (if needed)
    scaler = StandardScaler()
    scaler.fit(combined_df[['open', 'high', 'low', 'close', 'volume']])
    scaler_path = os.path.join(data_loader.dataset_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Saved scaler to {scaler_path}")

    return X, y, scaler, expected_return, cov_matrix
