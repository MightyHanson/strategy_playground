# data_loader.py

import os
import sqlite3
import pandas as pd

class DataLoader:
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = dataset_dir
        self.symbols_file = os.path.join(self.dataset_dir, 'symbols.xlsx')
        self.benchmark_config_file = os.path.join(self.dataset_dir, 'benchmark_config.xlsx')  # New config file
        self.db_path = os.path.join(self.dataset_dir, 'historical_data.db')

    def load_symbols(self):
        """
        Load and deduplicate symbols from Stocks and ETFs sheets.

        Returns:
            unique_symbols (list): List of unique stock and ETF symbols.
        """
        stocks_df = pd.read_excel(self.symbols_file, sheet_name='Stocks')
        etfs_df = pd.read_excel(self.symbols_file, sheet_name='ETFs')

        stocks = stocks_df['Stock Symbol'].dropna().unique().tolist()
        etfs = etfs_df['ETF Symbol'].dropna().unique().tolist()

        unique_symbols = list(set(stocks + etfs))
        return unique_symbols

    def load_benchmark_symbols_and_methods(self):
        """
        Load benchmark configurations from the benchmark_config.xlsx file.

        Returns:
            benchmark_configs (list of dict): List containing benchmark symbol and its configuration.
        """
        if not os.path.exists(self.benchmark_config_file):
            print(f"Error: {self.benchmark_config_file} does not exist.")
            return []

        try:
            benchmarks_df = pd.read_excel(self.benchmark_config_file)
            # Ensure required columns exist
            required_columns = ['Benchmark Symbol', 'Method']
            for col in required_columns:
                if col not in benchmarks_df.columns:
                    raise ValueError(f"Missing column '{col}' in benchmark configuration file.")

            # Fill NaN values with default parameters if necessary
            benchmarks_df.fillna({
                'Method': 'soared_then_decay',
                'Buy Threshold (%)': 35,
                'Decay Days': 14,
                'Window': 20,
                'Num Std': 2
            }, inplace=True)

            benchmark_configs = benchmarks_df.to_dict('records')
            return benchmark_configs
        except Exception as e:
            print(f"Error loading benchmark configurations: {e}")
            return []

    def load_historical_data(self, symbol, start_date, end_date):
        """
        Load historical price data for a given symbol from the SQLite database.

        Parameters:
            symbol (str): Stock, ETF, or benchmark symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            df (pd.DataFrame or None): DataFrame with historical data or None if no data found.
        """
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT date, close
        FROM historical_data
        WHERE symbol = ?
          AND date BETWEEN ? AND ?
        ORDER BY date ASC
        """
        params = (symbol, start_date, end_date)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            print(f"No data found for {symbol} between {start_date} and {end_date}.")
            return None

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
