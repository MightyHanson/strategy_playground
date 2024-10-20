# backtest.py

import pandas as pd
import numpy as np
import torch
from model import TransformerTimeSeries, train_model
from optimizer import optimize_portfolio
from predict import predict_future
from preprocess import preprocess_data, create_sequences
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
from data_loader import DataLoader

def backtest_portfolio(
    data_loader,
    symbols,
    risk_free_rate_series,
    start_date,
    end_date,
    rebalance_frequency,
    model_params,
    initial_portfolio_value=1000000,
    seq_length=30,
    num_epochs=5,
    device='cpu'
):
    """
    Backtest the optimized portfolio over a specified period.

    Args:
        data_loader (DataLoader): Instance of DataLoader to fetch data.
        symbols (list): List of symbols in the portfolio.
        risk_free_rate_series (pd.Series): Time series of daily risk-free rates.
        start_date (pd.Timestamp): Start date of the backtest.
        end_date (pd.Timestamp): End date of the backtest.
        rebalance_frequency (str): Frequency of rebalancing (e.g., 'M' for monthly).
        model_params (dict): Parameters for initializing the model.
        initial_portfolio_value (float): Starting portfolio value.
        seq_length (int): Sequence length for the Transformer model.
        num_epochs (int): Number of epochs for training the model.
        device (str): Device to use for computations ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: DataFrame containing portfolio values and performance metrics.
    """
    device = torch.device(device)
    logging.info(f"Using device: {device}")

    # Initialize variables
    portfolio_values = []
    dates = []
    positions = {}
    portfolio_value = initial_portfolio_value

    # To store dynamic weights
    weights_history = []

    # Initialize last known prices dictionary
    last_prices = {}

    # Create a date range for rebalancing
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_frequency)

    # Loop through each rebalancing date
    for i, current_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        logging.info(f"Rebalancing on {current_date.strftime('%Y-%m-%d')}")

        # Step 1: Load historical data up to current_date
        historical_data = data_loader.get_historical_data(symbols, end_date=current_date)
        if historical_data is None or len(historical_data) == 0:
            logging.warning(f"No historical data available up to {current_date}. Skipping this period.")
            continue

        # Step 2: Preprocess data
        try:
            X_train, y_train, scaler, expected_returns, cov_matrix = preprocess_data(historical_data, data_loader)
        except Exception as e:
            logging.error(f"Data preprocessing failed at {current_date}: {e}")
            continue

        # Ensure that expected_returns and cov_matrix are aligned
        aligned_symbols = expected_returns.index.intersection(cov_matrix.index)
        expected_returns = expected_returns.loc[aligned_symbols]
        cov_matrix = cov_matrix.loc[aligned_symbols, aligned_symbols]

        # Step 3: Train the Transformer model
        # Create TensorDatasets and DataLoaders
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Initialize and train the model
        model = TransformerTimeSeries(**model_params).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logging.info(f"Training model up to {current_date.strftime('%Y-%m-%d')}")
        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

        # Step 4: Generate predictions for each asset
        predictions_dict = {}
        for symbol in tqdm(aligned_symbols, desc=f"Generating predictions on {current_date.strftime('%Y-%m-%d')}"):
            symbol_data = historical_data[historical_data['symbol'] == symbol]
            symbol_data = symbol_data.sort_values('date')
            if len(symbol_data) < seq_length:
                logging.warning(f"Not enough data for symbol {symbol} to create sequences.")
                continue
            recent_data = symbol_data[['open', 'high', 'low', 'close', 'volume']].values[-seq_length:]
            try:
                predicted_close = predict_future(
                    model=model,
                    recent_data=recent_data,
                    scaler=scaler,
                    seq_length=seq_length,
                    device=device
                )
                last_close = symbol_data['close'].values[-1]
                predicted_return = (predicted_close - last_close) / last_close
                predictions_dict[symbol] = predicted_return
            except Exception as e:
                logging.error(f"Prediction failed for symbol {symbol} at {current_date}: {e}")
                continue

        # Convert predictions to a Series
        predicted_returns_series = pd.Series(predictions_dict)

        # Step 5: Optimize the portfolio
        # Ensure that predicted_returns_series aligns with cov_matrix
        common_assets = predicted_returns_series.index.intersection(cov_matrix.index)
        predicted_returns_series = predicted_returns_series.loc[common_assets]
        cov_matrix_aligned = cov_matrix.loc[common_assets, common_assets]

        try:
            optimized_weights = optimize_portfolio(
                expected_returns=predicted_returns_series,
                cov_matrix=cov_matrix_aligned,
                risk_aversion=0.5
            )
        except Exception as e:
            logging.error(f"Portfolio optimization failed at {current_date}: {e}")
            continue

        # Step 6: Simulate portfolio performance until next rebalancing date
        price_data = data_loader.get_price_data(
            symbols=optimized_weights.index.tolist(),
            start_date=current_date,
            end_date=next_date
        )

        if price_data is None or price_data.empty:
            logging.warning(f"No price data available between {current_date} and {next_date}.")
            continue

        price_data = price_data.ffill().bfill()

        # Fill missing prices using last known prices
        filled_price_data = fill_missing_prices(price_data, last_prices)

        # Calculate daily returns
        daily_returns = filled_price_data.pct_change().dropna()

        # # Calculate daily returns
        # daily_returns = price_data.pct_change().dropna()

        # Ensure that daily_returns columns align with optimized_weights
        common_assets = daily_returns.columns.intersection(optimized_weights.index)
        daily_returns = daily_returns[common_assets]
        optimized_weights = optimized_weights.loc[common_assets]

        # Store the weights with the current_date
        weights_history.append(optimized_weights)

        # Calculate portfolio daily returns
        daily_portfolio_returns = daily_returns.dot(optimized_weights)

        # Update portfolio value over the period
        cumulative_returns = (1 + daily_portfolio_returns).cumprod()
        period_portfolio_values = portfolio_value * cumulative_returns

        # Update portfolio value for next period
        portfolio_value = period_portfolio_values.iloc[-1]

        # Store results
        portfolio_values.append(period_portfolio_values)
        dates.extend(period_portfolio_values.index)

        logging.info(f"Completed the optmization for {current_date.strftime('%Y-%m-%d')}")

    # Combine all portfolio values
    if not portfolio_values:
        logging.error("No portfolio values generated during backtesting.")
        return None

    all_portfolio_values = pd.concat(portfolio_values)
    all_portfolio_values = all_portfolio_values[~all_portfolio_values.index.duplicated(keep='first')]
    all_portfolio_values.sort_index(inplace=True)

    # Calculate portfolio returns
    portfolio_returns = all_portfolio_values.pct_change().dropna()

    # Step 7: Calculate performance metrics
    performance_metrics = calculate_performance_metrics(portfolio_returns, risk_free_rate_series)

    # Compile results
    results = pd.DataFrame({
        'Portfolio Value': all_portfolio_values,
        'Portfolio Returns': portfolio_returns,
    })

    # Add performance metrics (as constants for simplicity)
    results['Sharpe Ratio'] = performance_metrics['Sharpe Ratio']
    results['Max Drawdown'] = performance_metrics['Max Drawdown']

    # Compile weights history into a DataFrame
    weights_df = pd.DataFrame(weights_history, index=[date.strftime('%Y-%m-%d') for date in rebalance_dates[:-1]])

    # Ensure that all columns (symbols) are present, fill missing with 0
    weights_df = weights_df.fillna(0)

    return results, weights_df

def calculate_performance_metrics(portfolio_returns, risk_free_rate_series):
    """
    Calculate performance metrics for the portfolio.

    Args:
        portfolio_returns (pd.Series): Series of portfolio returns.
        risk_free_rate_series (pd.Series): Series of daily risk-free rates aligned with portfolio_returns.

    Returns:
        dict: Dictionary containing performance metrics.
    """
    # Align risk-free rate series with portfolio returns
    df_rf_daily_aligned = risk_free_rate_series.reindex(portfolio_returns.index).fillna(method='ffill')

    # Calculate excess returns
    excess_returns = portfolio_returns - df_rf_daily_aligned

    # Calculate Sharpe Ratio
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)

    # Calculate Max Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
    }

def fill_missing_prices(price_data, last_prices):
    """
    Replace NaN values in price_data with the last known prices stored in last_prices.
    This function is to solve a scenario where the stock price in the current backtest cycle is not available
    And this function will replace the nan to be the last record value
    Args:
        price_data (pd.DataFrame): DataFrame containing price data with potential NaNs.
        last_prices (dict): Dictionary storing the last known price for each symbol.

    Returns:
        pd.DataFrame: Price data with NaNs filled using last known prices.
    """
    filled_price_data = price_data.copy()

    for symbol in filled_price_data.columns:
        if symbol in last_prices:
            # Replace NaNs with the last known price
            filled_price_data[symbol].fillna(last_prices[symbol], inplace=True)

        # Update last_prices with the latest non-NaN value
        non_nan_series = filled_price_data[symbol].dropna()
        if not non_nan_series.empty:
            last_prices[symbol] = non_nan_series.iloc[-1]

    return filled_price_data