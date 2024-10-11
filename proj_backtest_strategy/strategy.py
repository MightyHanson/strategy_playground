# strategy.py

class BuyTheDipStrategy:
    def __init__(self, benchmark, buy_threshold=None, decay_days=None, window=None, num_std=None):
        """
        Initialize the BuyTheDipStrategy.

        Parameters:
        - benchmark (Benchmark): Benchmark instance with data and signals.
        - buy_threshold (float, optional): Threshold percentage for buy signals.
        - decay_days (int, optional): Number of days for decay to trigger sell signals.
        - window (int, optional): Window size for Bollinger Bands.
        - num_std (float, optional): Number of standard deviations for Bollinger Bands.
        """
        self.benchmark = benchmark
        self.buy_threshold = buy_threshold
        self.decay_days = decay_days
        self.window = window
        self.num_std = num_std

    def generate_signals(self):
        """
        Retrieve the signals from the benchmark.

        Returns:
        - signals (pd.DataFrame): DataFrame containing Buy and Sell signals.
        """
        return self.benchmark.signals
