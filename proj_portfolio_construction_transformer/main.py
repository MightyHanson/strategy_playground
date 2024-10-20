# main.py

import os
import sys
import pandas as pd
from data_loader import DataLoader
from data_fetching import main_data_fetching
from preprocess import load_and_prepare_data, preprocess_data
from model import TransformerTimeSeries, train_model, evaluate_model
from predict import predict_future
from optimizer import optimize_portfolio
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import setup_logging
import logging
from backtest import backtest_portfolio, calculate_performance_metrics
import matplotlib.pyplot as plt
import seaborn as sns

torch.cuda.set_per_process_memory_fraction(0.8, device=0)

def save_evaluation_results(output_dir, evaluation_results):
    """
    Save evaluation metrics and predictions to CSV files and generate plots.

    Args:
        output_dir (str): Directory where the results will be saved.
        evaluation_results (dict): Dictionary containing 'mse', 'r2', and 'predictions'.
    """
    # Create a subdirectory for evaluation results
    eval_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'MSE': [evaluation_results['mse']],
        'R2_Score': [evaluation_results['r2']]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(eval_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved evaluation metrics to {metrics_csv_path}")

    # Save predictions
    predictions_csv_path = os.path.join(eval_dir, 'predictions.csv')
    evaluation_results['predictions'].to_csv(predictions_csv_path, index=False)
    logging.info(f"Saved predictions to {predictions_csv_path}")

    # Optionally, save metrics in JSON format
    metrics_json_path = os.path.join(eval_dir, 'evaluation_metrics.json')
    metrics_df.to_json(metrics_json_path, orient='records', lines=True)
    logging.info(f"Saved evaluation metrics to {metrics_json_path}")

    # Generate and save a scatter plot of Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=evaluation_results['predictions'], alpha=0.5)
    plt.title('Actual vs. Predicted Close Prices')
    plt.xlabel('Actual Close Price')
    plt.ylabel('Predicted Close Price')
    plt.plot([evaluation_results['predictions']['Actual'].min(), evaluation_results['predictions']['Actual'].max()],
             [evaluation_results['predictions']['Actual'].min(), evaluation_results['predictions']['Actual'].max()],
             color='red', linestyle='--')
    scatter_plot_path = os.path.join(eval_dir, 'actual_vs_predicted.png')
    plt.savefig(scatter_plot_path)
    plt.close()
    logging.info(f"Saved Actual vs. Predicted plot to {scatter_plot_path}")


def main():
    """
    Coordinate the entire data fetching, loading, model training, prediction, and portfolio optimization process.
    """
    # Step 1: Define Output Directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")

    # Step 2: Set up Logging
    setup_logging(output_dir, log_file='portfolio_management.log')
    logging.info("Starting the portfolio management process.")

    # Step 3: Data Fetching and Storage
    # main_data_fetching(output_dir)

    # Step 4: Initialize DataLoader
    data_loader = DataLoader(output_dir)

    # Step 5: Load and Prepare Data
    symbols = data_loader.load_symbols()
    if not symbols:
        logging.error("No symbols loaded. Exiting the process.")
        sys.exit(1)
    combined_df = load_and_prepare_data(data_loader, symbols)
    logging.info(f"Loaded and combined data for {len(symbols)} symbols.")

    # filter the Symbols with certain time limit
    min_start_date = pd.to_datetime('2006-01-03')
    symbol_start_dates = pd.to_datetime(combined_df.groupby('symbol')['date'].min())
    valid_symbols = symbol_start_dates[symbol_start_dates <= min_start_date].index.tolist()
    list_to_exclude = ['FERG', '^VIX','CCEP.AS']
    valid_symbols = [symbol for symbol in valid_symbols if symbol not in list_to_exclude]
    combined_df = combined_df[combined_df['symbol'].isin(valid_symbols)]
    combined_df = combined_df[pd.to_datetime(combined_df['date']) >= min_start_date]
    # Create the filename using the min_start_date
    date_str = min_start_date.strftime('%Y%m%d')
    filename = f'symbols_{date_str}.xlsx'
    # Create a DataFrame from the list of symbols
    symbols_df = pd.DataFrame(valid_symbols, columns=['symbol'])
    # Save to Excel
    symbols_df.to_excel(output_dir + '/' + filename, index=False, sheet_name=f'symbols_{date_str}')
    logging.info(f"Saved filtered symbols to {output_dir}")

    # risk free data loaded
    df_rf = data_loader.load_risk_free_rate()
    df_rf_3M = df_rf['DGS3MO']
    logging.info(f"Risk Free Data Loaded")

    # Step 6: Preprocess Data
    X, y, scaler, expected_returns, cov_matrix = preprocess_data(combined_df, data_loader)
    logging.info("Data preprocessing completed.")

    # Step 7: Split into Train and Test Sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    logging.info(f"Data split into {split} training samples and {len(X_test)} testing samples.")

    # Step 8: Convert Numpy Arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Shape: (num_samples, seq_length, features)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Shape: (num_samples,)

    # Verify shapes
    print(f"X_train_tensor shape: {X_train_tensor.shape}")
    print(f"y_train_tensor shape: {y_train_tensor.shape}")
    logging.info(f"X_train_tensor shape: {X_train_tensor.shape}")
    logging.info(f"y_train_tensor shape: {y_train_tensor.shape}")

    # Ensure the first dimension matches
    assert X_train_tensor.shape[0] == y_train_tensor.shape[0], "Mismatch in number of samples between X and y."

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    logging.info("Converted data to PyTorch tensors and created DataLoader.")

    # Step 9: Initialize and Train the Transformer Model
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    logging.info(f'Using device: {device}')

    num_epoch = 1
    model = TransformerTimeSeries(feature_size=5, num_layers=2, dropout=0.1, nhead=1, dim_feedforward=128) # 128 for dim_feedforward test
    model.to(device)  # Move model to device
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info("Starting model training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epoch, device=device)
    logging.info("Model training completed.")

    # Step 10: Evaluate the Model
    torch.cuda.empty_cache()
    evaluation_results = evaluate_model(model, X_test, y_test, device=device)

    # Step 11: Save the Trained Model
    model_path = os.path.join(output_dir, 'transformer_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved the trained model to {model_path}")

    # Step 12: Save Evaluation Results
    save_evaluation_results(output_dir, evaluation_results)

    # Step 13: Generate Predictions for Portfolio Optimization
    predictions_dict = {}
    # symbols_loaded = data_loader.load_symbols()

    for symbol in tqdm(valid_symbols, desc="Generating predictions for portfolio optimization"):
        symbol_data = data_loader.load_data(symbol)
        if symbol_data is None or len(symbol_data) < 30:
            logging.warning(f"Insufficient data for symbol: {symbol}. Skipping prediction.")
            continue
        recent_data = symbol_data[['open', 'high', 'low', 'close', 'volume']].values[-30:]
        predicted_close = predict_future(model, recent_data, scaler, seq_length=30, device=device)
        last_close = symbol_data['close'].values[-1]
        predicted_return = (predicted_close - last_close) / last_close
        predictions_dict[symbol] = predicted_return
        logging.info(f"Predicted return for {symbol}: {predicted_return:.4f}")

    # Convert predictions to a Series
    predicted_returns_series = pd.Series(predictions_dict)
    logging.info(f"Generated predictions for {len(predicted_returns_series)} symbols.")

    # Select top N assets based on predicted returns
    # N = 500
    # if len(predicted_returns_series) > N:
    #     top_assets = predicted_returns_series.nlargest(N).index
    #     predicted_returns_series = predicted_returns_series.loc[top_assets]
    #     cov_matrix = cov_matrix.loc[top_assets, top_assets]

    # Step 14: Portfolio Optimization
    optimized_weights = optimize_portfolio(
        expected_returns_aligned := expected_returns.loc[predicted_returns_series.index],
        cov_matrix_aligned := cov_matrix.loc[predicted_returns_series.index, predicted_returns_series.index],
        risk_aversion=0.5
    )
    logging.info("Portfolio optimization completed.")

    # Step 15: Save the Optimized Portfolio to Excel
    portfolio_excel_path = os.path.join(output_dir, 'optimized_portfolio.xlsx')
    optimized_weights.to_excel(portfolio_excel_path, header=['Weight'])
    logging.info(f"Optimized portfolio saved to {portfolio_excel_path}")

    # Step 16: Backtesting
    logging.info("Starting backtesting of the optimized portfolio.")
    # Static Weight Backtest
    # Prepare the daily risk-free rate series
    df_rf_3M.index = pd.to_datetime(df_rf_3M.index)
    df_rf_3M.sort_index(inplace=True)
    df_rf_3M = df_rf_3M / 100  # Convert to decimal
    days_in_year = 252
    df_rf_daily = (1 + df_rf_3M) ** (1 / days_in_year) - 1

    logging.info("Starting static weight backtesting of the optimized portfolio.")

    # Define the backtest period
    latest_data_date = pd.to_datetime(combined_df['date']).max()
    optimization_date = latest_data_date - pd.DateOffset(years=1)
    backtest_start_date = optimization_date + pd.Timedelta(days=1)
    backtest_end_date = latest_data_date

    # Fetch price data for the backtest period
    price_data = data_loader.get_price_data(
        symbols=optimized_weights.index.tolist(),
        start_date=backtest_start_date,
        end_date=backtest_end_date
    )

    # Ensure price_data is available
    if price_data is None or price_data.empty:
        logging.error("No price data available for the static backtest period.")
    else:
        # Calculate daily returns
        daily_returns = price_data.pct_change().dropna()

        # Align data
        common_assets = daily_returns.columns.intersection(optimized_weights.index)
        daily_returns = daily_returns[common_assets]
        weights = optimized_weights.loc[common_assets]

        # Calculate portfolio daily returns
        daily_portfolio_returns = daily_returns.dot(weights)

        # Calculate cumulative portfolio value
        initial_portfolio_value = 1000000  # Starting portfolio value
        cumulative_returns = (1 + daily_portfolio_returns).cumprod()
        portfolio_values = initial_portfolio_value * cumulative_returns

        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(daily_portfolio_returns, df_rf_daily)

        # Save results
        static_backtest_results = pd.DataFrame({
            'Portfolio Value': portfolio_values,
            'Portfolio Returns': daily_portfolio_returns
        })
        static_backtest_results.to_csv(os.path.join(output_dir, 'backtest_results_static_weights.csv'))
        logging.info("Static weight backtesting completed and results saved.")

        # Log performance metrics
        logging.info(f"Static Backtest Sharpe Ratio: {performance_metrics['Sharpe Ratio']:.4f}")
        logging.info(f"Static Backtest Max Drawdown: {performance_metrics['Max Drawdown']:.4%}")

    # Dynamic Weight Backtest
    # Define backtesting parameters
    logging.info("Starting Dynamic weight backtesting of the optimized portfolio.")

    start_date = min_start_date + pd.DateOffset(months = 2)
    end_date = pd.to_datetime('2024-09-30')
    rebalance_frequency = 'M'  # Monthly rebalancing

    # Backtest the portfolio
    try:
        backtest_results, weights_df = backtest_portfolio(
            data_loader=data_loader,
            symbols=valid_symbols,
            risk_free_rate_series=df_rf_daily,
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency=rebalance_frequency,
            model_params={
                'feature_size': 5,
                'num_layers': 2,
                'dropout': 0.1,
                'nhead': 1,
                'dim_feedforward': 128,
            },
            initial_portfolio_value=1000000,
            seq_length=30,
            num_epochs=num_epoch,
            device=device
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during backtesting: {e}")
        sys.exit(1)

    # Save backtesting results
    if backtest_results is not None:
        backtest_results.to_csv(os.path.join(output_dir, 'backtest_results_dynamic_weights.csv'))
        logging.info("Backtesting completed and results saved.")
    else:
        logging.error("Backtesting failed. No results to save.")

    # Save dynamic weights to Excel
    if weights_df is not None and not weights_df.empty:
        weights_excel_path = os.path.join(output_dir, 'dynamic_portfolio_weights.xlsx')
        with pd.ExcelWriter(weights_excel_path, engine='openpyxl') as writer:
            weights_df.to_excel(writer, sheet_name='Weights')

            # Optionally, format the Excel sheet for better readability
            workbook = writer.book
            worksheet = writer.sheets['Weights']
            for column_cells in worksheet.columns:
                length = max(len(str(cell.value)) for cell in column_cells)
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = length + 2
                for cell in column_cells[1:]:
                    cell.number_format = '0.00%'  # Format weights as percentages

        logging.info(f"Dynamic portfolio weights saved to {weights_excel_path}")
    else:
        logging.error("No dynamic weights to save.")

    logging.info("Portfolio management process completed successfully.")

if __name__ == "__main__":
    main()
