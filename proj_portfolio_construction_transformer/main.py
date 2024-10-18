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

    # Generate and save a residual plot
    # plt.figure(figsize=(10, 6))
    # sns.residplot(x='Actual', y='Predicted', data=evaluation_results['predictions'], lowess=True, line_kws={'color': 'red'})
    # plt.title('Residuals of Predictions')
    # plt.xlabel('Actual Close Price')
    # plt.ylabel('Residuals')
    # residual_plot_path = os.path.join(eval_dir, 'residuals.png')
    # plt.savefig(residual_plot_path)
    # plt.close()
    # logging.info(f"Saved residuals plot to {residual_plot_path}")

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

    model = TransformerTimeSeries(feature_size=5, num_layers=2, dropout=0.1, nhead=1, dim_feedforward=128) # 128 for dim_feedforward test
    model.to(device)  # Move model to device
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info("Starting model training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=1, device=device)
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
    # Initialize predictions dictionary
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

    logging.info("Portfolio management process completed successfully.")

if __name__ == "__main__":
    main()
