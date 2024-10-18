# benchmark.py

import pandas as pd
import numpy as np

class Benchmark:
    def __init__(self, symbol, data_loader, start_date, end_date, method='soared_then_decay',
                 buy_threshold=None, decay_days=None, window=None, num_std=None):
        """
        Initialize the Benchmark instance.

        Parameters:
        - symbol (str): Benchmark symbol (e.g., '^VIX').
        - data_loader (DataLoader): Instance of DataLoader to fetch data.
        - start_date (str): Start date for historical data (YYYY-MM-DD).
        - end_date (str): End date for historical data (YYYY-MM-DD).
        - method (str): Signal generation method (e.g., 'soared_then_decay').
        - buy_threshold (float): Buy threshold percentage (for 'soared_then_decay').
        - decay_days (int): Number of decay days (for 'soared_then_decay').
        - window (int): Window size for moving averages or Bollinger Bands.
        - num_std (float): Number of standard deviations for Bollinger Bands.
        """
        self.symbol = symbol
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.method = method.lower().replace(' ', '_')  # Normalize method name
        self.buy_threshold = buy_threshold
        self.decay_days = decay_days
        self.window = window
        self.num_std = num_std
        self.data = self.load_data()
        self.signals = self.generate_signals()

    def load_data(self):
        """
        Load historical data for the benchmark.

        Returns:
        - pd.DataFrame: Historical data with date as index.
        """
        data = self.data_loader.load_historical_data(self.symbol, self.start_date, self.end_date)
        if data is None or data.empty:
            raise ValueError(f"No historical data found for benchmark: {self.symbol}")
        return data

    def generate_signals(self):
        """
        Generate buy and sell signals based on the specified method.

        Returns:
        - pd.DataFrame: DataFrame with Buy and Sell signals.
        """
        # Mapping of method names to functions
        methods = {
            'soared_then_decay': self.soared_then_decay,
            'ma_20_50': self.ma_20_50,
            'ma_50_100': self.ma_50_100,
            'bollinger_bands': self.bollinger_bands,
            # Add more methods here as needed
        }

        if self.method not in methods:
            raise ValueError(f"Signal generation method '{self.method}' is not defined for benchmark '{self.symbol}'.")

        # Call the appropriate method
        return methods[self.method]()

    def soared_then_decay(self):
        """
        Generate buy and sell signals based on the "Soared Then Decay" strategy.

        Buy Signal: Benchmark increases by buy_threshold% in a single day.
        Sell Signal: Benchmark decreases for decay_days consecutive days.

        Returns:
        - pd.DataFrame: DataFrame with Buy and Sell signals.
        """
        buy_threshold = self.buy_threshold if self.buy_threshold is not None else 35
        decay_days = self.decay_days if self.decay_days is not None else 14

        signals = pd.DataFrame(index=self.data.index)
        signals['Benchmark'] = self.data['close']
        signals['Benchmark_Change'] = self.data['close'].pct_change() * 100  # Percentage change

        # Buy Signal: Benchmark increases by buy_threshold% in a single day
        signals['Buy'] = signals['Benchmark_Change'] >= buy_threshold

        # Sell Signal: Benchmark decreases for decay_days consecutive days
        signals['Benchmark_Decay'] = signals['Benchmark'].rolling(window=decay_days).apply(
            lambda x: all(x[i] < x[i - 1] for i in range(1, len(x))) if len(x) == decay_days else False,
            raw=True
        )
        signals['Sell'] = signals['Benchmark_Decay'] == 1

        # After generating signals, print counts
        num_buy_signals = signals['Buy'].sum()
        num_sell_signals = signals['Sell'].sum()
        print(f"Symbol {self.symbol}: {num_buy_signals} Buy signals, {num_sell_signals} Sell signals generated.")

        return signals

    def ma_20_50(self):
        """
        Generate buy and sell signals based on the MA_20_50 strategy.

        Buy Signal: Benchmark crosses above MA_20.
        Sell Signal: Benchmark crosses below MA_50.

        Returns:
        - pd.DataFrame: DataFrame with Buy and Sell signals.
        """
        signals = pd.DataFrame(index=self.data.index)
        signals['Benchmark'] = self.data['close']
        signals['MA_20'] = self.data['close'].rolling(window=20).mean()
        signals['MA_50'] = self.data['close'].rolling(window=50).mean()

        # Buy Signal: Benchmark crosses above MA_20
        signals['Buy'] = (signals['Benchmark'].shift(1) <= signals['MA_20'].shift(1)) & (signals['Benchmark'] > signals['MA_20'])

        # Sell Signal: Benchmark crosses below MA_50
        signals['Sell'] = (signals['Benchmark'].shift(1) >= signals['MA_50'].shift(1)) & (signals['Benchmark'] < signals['MA_50'])

        return signals

    def ma_50_100(self):
        """
        Generate buy and sell signals based on the MA_50_100 strategy.

        Buy Signal: Benchmark crosses above MA_50.
        Sell Signal: Benchmark crosses below MA_100.

        Returns:
        - pd.DataFrame: DataFrame with Buy and Sell signals.
        """
        signals = pd.DataFrame(index=self.data.index)
        signals['Benchmark'] = self.data['close']
        signals['MA_50'] = self.data['close'].rolling(window=50).mean()
        signals['MA_100'] = self.data['close'].rolling(window=100).mean()

        # Buy Signal: Benchmark crosses above MA_50
        signals['Buy'] = (signals['Benchmark'].shift(1) <= signals['MA_50'].shift(1)) & (signals['Benchmark'] > signals['MA_50'])

        # Sell Signal: Benchmark crosses below MA_100
        signals['Sell'] = (signals['Benchmark'].shift(1) >= signals['MA_100'].shift(1)) & (signals['Benchmark'] < signals['MA_100'])

        return signals

    def bollinger_bands(self):
        """
        Generate buy and sell signals based on the Bollinger Bands strategy.

        Buy Signal: Benchmark crosses below the lower Bollinger Band.
        Sell Signal: Benchmark crosses above the upper Bollinger Band.

        Returns:
        - pd.DataFrame: DataFrame with Buy and Sell signals.
        """
        window = self.window if self.window is not None else 20
        num_std = self.num_std if self.num_std is not None else 2

        signals = pd.DataFrame(index=self.data.index)
        signals['Benchmark'] = self.data['close']
        signals['MA'] = self.data['close'].rolling(window=window).mean()
        signals['STD'] = self.data['close'].rolling(window=window).std()
        signals['Upper Band'] = signals['MA'] + (signals['STD'] * num_std)
        signals['Lower Band'] = signals['MA'] - (signals['STD'] * num_std)

        # Buy Signal: Benchmark crosses below the lower Bollinger Band
        signals['Buy'] = (signals['Benchmark'].shift(1) >= signals['Lower Band'].shift(1)) & (signals['Benchmark'] < signals['Lower Band'])

        # Sell Signal: Benchmark crosses above the upper Bollinger Band
        signals['Sell'] = (signals['Benchmark'].shift(1) <= signals['Upper Band'].shift(1)) & (signals['Benchmark'] > signals['Upper Band'])

        return signals

    # Future signal generation methods can be defined here
    # def another_new_strategy(self):
    #     pass
