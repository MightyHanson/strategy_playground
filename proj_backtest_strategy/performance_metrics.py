# performance_metrics.py

import pandas as pd
import numpy as np

class PerformanceMetrics:
    def __init__(self, portfolio_df):
        """
        Initialize the PerformanceMetrics instance.

        Parameters:
        - portfolio_df (pd.DataFrame): DataFrame containing portfolio value and trades.
        """
        self.portfolio_df = portfolio_df

    def calculate_metrics(self):
        """
        Calculate performance metrics.

        Returns:
        - metrics (dict): Dictionary containing calculated metrics.
        """
        metrics = {}

        # Ensure the index is datetime
        self.portfolio_df = self.portfolio_df.copy()
        self.portfolio_df.index = pd.to_datetime(self.portfolio_df.index)

        # Calculate cumulative return
        starting_value = self.portfolio_df['total_value'].iloc[0]
        ending_value = self.portfolio_df['total_value'].iloc[-1]
        cumulative_return = (ending_value / starting_value) - 1
        metrics['Cumulative Return'] = cumulative_return

        # Calculate annualized return
        days = (self.portfolio_df.index[-1] - self.portfolio_df.index[0]).days
        years = days / 365.25
        if years > 0:
            annualized_return = (ending_value / starting_value) ** (1 / years) - 1
        else:
            annualized_return = np.nan
        metrics['Annualized Return'] = annualized_return

        # Calculate daily returns
        self.portfolio_df['daily_return'] = self.portfolio_df['total_value'].pct_change().fillna(0)

        # Calculate Sharpe Ratio (Assuming risk-free rate = 0)
        if self.portfolio_df['daily_return'].std() != 0:
            sharpe_ratio = (self.portfolio_df['daily_return'].mean() / self.portfolio_df['daily_return'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = np.nan
        metrics['Sharpe Ratio'] = sharpe_ratio

        # Calculate Max Drawdown
        cumulative = (1 + self.portfolio_df['daily_return']).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        metrics['Max Drawdown'] = max_drawdown

        # Calculate Win Rate
        trades = self.portfolio_df[self.portfolio_df['action'].isin(['buy', 'sell'])].copy()
        trades['trade_return'] = trades['total_value'].pct_change().shift(-1)  # Shift to get return after the trade

        valid_returns = trades['trade_return'].dropna()
        if not valid_returns.empty:
            wins = valid_returns > 0
            win_rate = wins.mean()
        else:
            win_rate = np.nan

        metrics['Win Rate'] = win_rate

        return metrics
