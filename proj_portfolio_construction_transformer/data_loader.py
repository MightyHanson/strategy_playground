# data_loader.py

import os
import pandas as pd
import sqlite3

class DataLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.db_path = os.path.join(self.dataset_dir, 'historical_data.db')

    def load_data(self, symbol):
        """
        Load historical data for a given symbol from the SQLite database.

        Parameters:
        - symbol (str): The stock or ETF symbol.

        Returns:
        - data (pd.DataFrame): DataFrame containing historical data for the symbol.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM historical_data WHERE symbol = '{symbol}' ORDER BY date"
            data = pd.read_sql_query(query, conn)
            conn.close()
            if data.empty:
                return None
            return data
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None

    def load_symbols(self):
        """
        Load the list of symbols from the symbols.xlsx file.

        Returns:
        - symbols (list): List of stock and ETF symbols.
        """
        symbols_excel_path = os.path.join(self.dataset_dir, 'symbols.xlsx')
        if not os.path.exists(symbols_excel_path):
            print(f"Error: {symbols_excel_path} does not exist.")
            return []
        try:
            xls = pd.ExcelFile(symbols_excel_path)
            stock_symbols = xls.parse('Stocks')['Stock Symbol'].dropna().tolist()
            etf_symbols = xls.parse('ETFs')['ETF Symbol'].dropna().tolist()
            benchmark_symbols = xls.parse('Benchmark Indices')['Benchmark Symbol'].dropna().tolist()
            symbols = stock_symbols + etf_symbols + benchmark_symbols
            return symbols
        except Exception as e:
            print(f"Error loading symbols from Excel: {e}")
            return []

    def load_risk_free_rate(self):
        risk_free_rate_path = os.path.join(self.dataset_dir, 'yield_data_df.xlsx')
        if not os.path.exists(risk_free_rate_path):
            print(f"Error: {risk_free_rate_path} does not exist.")
            return []
        try:
            xls = pd.ExcelFile(risk_free_rate_path)
            df_risk_free = xls.parse('Sheet1').dropna()
            df_risk_free['date'] = pd.to_datetime(df_risk_free['date'])
            df_risk_free.set_index('date',inplace = True)
            return df_risk_free
        except Exception as e:
            print(f"Error loading symbols from Excel: {e}")
            return []

    def load_benchmark_configurations(self):
        """
        Load benchmark configurations from the configuration file.

        Returns:
        - benchmark_configs (dict): Dictionary with symbol as key and configuration dict as value.
        """
        benchmark_configs_excel_path = os.path.join(self.dataset_dir, 'benchmark_configs.xlsx')
        if not os.path.exists(benchmark_configs_excel_path):
            print(f"Configuration file {benchmark_configs_excel_path} not found.")
            return {}

        try:
            df = pd.read_excel(benchmark_configs_excel_path)
            benchmark_configs = {}

            for _, row in df.iterrows():
                symbol = row['Symbol']
                config = {
                    'method': row['Method'].lower().replace(' ', '_') if not pd.isna(row['Method']) else 'tail_risk_buy',
                    'percentile': row['Percentile'] if not pd.isna(row['Percentile']) else None,
                    'cumulative_gain_threshold': row['Cumulative Gain Threshold'] if not pd.isna(row['Cumulative Gain Threshold']) else None,
                    'initial_window': row['Initial Window'] if not pd.isna(row['Initial Window']) else None,
                    'buy_threshold': row['Buy Threshold'] if not pd.isna(row['Buy Threshold']) else None,
                    'decay_days': row['Decay Days'] if not pd.isna(row['Decay Days']) else None,
                    # Add other parameters as needed
                }
                benchmark_configs[symbol] = config
            return benchmark_configs
        except Exception as e:
            print(f"Error loading benchmark configurations: {e}")
            return {}

    def get_historical_data(self, symbols, end_date):
        """
        Fetch historical data for the given symbols up to the end_date.

        Args:
            symbols (list): List of symbols.
            end_date (pd.Timestamp): End date for the data.

        Returns:
            pd.DataFrame: Combined DataFrame of historical data.
        """
        all_data = []
        for symbol in symbols:
            df = self.load_data(symbol)
            df['date'] = pd.to_datetime(df['date'])
            if df is not None:
                df = df[df['date'] <= end_date]
                all_data.append(df)
        if not all_data:
            return None
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    def get_price_data(self, symbols, start_date, end_date):
        """
        Fetch price data for the given symbols between start_date and end_date.

        Args:
            symbols (list): List of symbols.
            start_date (pd.Timestamp): Start date for the data.
            end_date (pd.Timestamp): End date for the data.

        Returns:
            pd.DataFrame: DataFrame of price data with dates as index and symbols as columns.
        """
        all_data = []
        for symbol in symbols:
            df = self.load_data(symbol)
            if df is not None:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                df = df[['date', 'close']]
                df.set_index('date', inplace=True)
                df.rename(columns={'close': symbol}, inplace=True)
                all_data.append(df)
        if not all_data:
            return None
        price_data = pd.concat(all_data, axis=1)
        price_data.sort_index(inplace=True)
        return price_data
