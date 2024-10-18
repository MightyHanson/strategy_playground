# backtrader_bt_try.py

import backtrader as bt
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from optimizer import optimize_portfolio
from preprocess import preprocess_data
from model import TransformerTimeSeries, train_model
from predict import predict_future
from data_loader import DataLoader
import logging
import os

class MLPortfolioStrategy(bt.Strategy):
    params = (
        ('train_window', 252 * 2),  # 2 years of trading days
        ('test_window', 21),  # 1 month of trading days
        ('risk_free_rate', 0.04),  # 4% annual risk-free rate
        ('printlog', False),
        ('device', torch.device('cpu')),  # Default to CPU
    )

    def __init__(self):
        self.dataclose = {d: d.close for d in self.datas}
        self.order = None
        self.buyprice = {}
        self.buycomm = {}
        self.portfolio_weights = {}
        self.iteration = 0
        self.device = self.params.device  # Correctly set the device
        self.logger = logging.getLogger('MLPortfolioStrategy')
        self.portfolio_weights_over_time = []
        self.logger = logging.getLogger('MLPortfolioStrategy')
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


    def log(self, txt, dt=None):
        ''' Logging function for this strategy '''
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            self.logger.info(f'{dt.isoformat()} {txt}')

    def next(self):
        self.iteration += 1
        # Check if it's time to rebalance
        if self.iteration % self.params.test_window == 0:
            self.log('Rebalancing Portfolio')

            # Determine the start and end of the training window
            end = len(self)
            start = end - self.params.train_window
            if start < 0:
                self.log('Not enough data to train yet.')
                return

            # Extract training data
            training_data = {}
            for d in self.datas:
                symbol = d._name
                training_data[symbol] = {
                    'open': list(d.open.get(size=self.params.train_window)),
                    'high': list(d.high.get(size=self.params.train_window)),
                    'low': list(d.low.get(size=self.params.train_window)),
                    'close': list(d.close.get(size=self.params.train_window)),
                    'volume': list(d.volume.get(size=self.params.train_window)),
                }

            # Convert to DataFrame
            combined_train_df = pd.DataFrame()
            for symbol, data in training_data.items():
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                combined_train_df = pd.concat([combined_train_df, df], axis=0)

            # Preprocess training data
            self.log('Starting Preprocessing')
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'dataset')
            data_loader = DataLoader(output_dir)
            X_train, y_train, scaler, expected_returns, cov_matrix = preprocess_data(combined_train_df, data_loader)
            self.log('Preprocessing completed')

            # Train the model
            model = TransformerTimeSeries(feature_size=5, num_layers=2, dropout=0.1, nhead=1, dim_feedforward=128)
            model.to(self.device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                               torch.tensor(y_train, dtype=torch.float32)),
                batch_size=32,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            self.log('Starting Model Training')
            train_model(model, train_loader, criterion, optimizer, num_epochs=15, device=self.device)
            torch.cuda.empty_cache()
            self.log('Model training completed.')

            # Generate predictions for each asset
            predictions_dict = {}
            for symbol in expected_returns.index:
                # Extract the latest sequence for prediction
                symbol_data = training_data[symbol]
                if len(symbol_data['close']) < 30:
                    self.log(f"Insufficient data for symbol: {symbol}. Skipping prediction.")
                    continue
                recent_data = np.array([
                    symbol_data['open'][-30:],
                    symbol_data['high'][-30:],
                    symbol_data['low'][-30:],
                    symbol_data['close'][-30:],
                    symbol_data['volume'][-30:]
                ]).T  # Shape: (30, 5)

                predicted_close = predict_future(model, recent_data, scaler, seq_length=30, device=self.device)
                last_close = symbol_data['close'][-1]
                predicted_return = (predicted_close - last_close) / last_close
                predictions_dict[symbol] = predicted_return
                self.log(f'Predicted return for {symbol}: {predicted_return:.4f}')

            # Convert predictions to a Series
            predicted_returns_series = pd.Series(predictions_dict)
            self.log(f"Generated predictions for {len(predicted_returns_series)} symbols.")

            # Align expected returns and covariance matrix
            expected_returns_aligned = expected_returns.loc[predicted_returns_series.index]
            cov_matrix_aligned = cov_matrix.loc[predicted_returns_series.index, predicted_returns_series.index]

            # Portfolio Optimization
            optimized_weights = optimize_portfolio(
                expected_returns=expected_returns_aligned,
                cov_matrix=cov_matrix_aligned,
                risk_aversion=0.5
            )
            self.portfolio_weights = optimized_weights.to_dict()
            self.log("Portfolio optimization completed.")

            # Save Portfolio Weights Over Time
            self.portfolio_weights_over_time.append(self.portfolio_weights.copy())

            # Rebalance portfolio
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        ''' Rebalance the portfolio based on optimized weights '''
        for d in self.datas:
            symbol = d._name
            weight = self.portfolio_weights.get(symbol, 0)
            if weight == 0:
                # If weight is zero, close any existing position
                if self.getposition(d).size > 0:
                    self.close(data=d)
                    self.log(f'Closing position in {symbol}')
                continue
            target_value = self.broker.getvalue() * weight
            current_position = self.getposition(d).size
            current_value = current_position * d.close[0]
            diff = target_value - current_value
            if diff > 0:
                size = diff / d.close[0]
                self.buy(data=d, size=size)
                self.log(f'Buying {size:.2f} shares of {symbol}')
            elif diff < 0:
                size = -diff / d.close[0]
                self.sell(data=d, size=size)
                self.log(f'Selling {size:.2f} shares of {symbol}')
