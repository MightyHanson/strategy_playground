# data_fetching.py

import os
import sys
import requests
import pandas as pd
import sqlite3
from tqdm import tqdm
import csv
from datetime import datetime, timedelta
import yfinance as yf
from ratelimiter import RateLimiter
from dotenv import load_dotenv
from data_loader import DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================================ Configuration and Constants ================================
load_dotenv()  # Load environment variables from .env file

FMP_API_KEY = os.getenv("FMP_API_KEY")  # Financial Modeling Prep API Key
BENCHMARK_INDICES = ['^VIX']  # List of benchmark indices to include
ETF_VOLUME_THRESHOLD = 200000  # Minimum average daily volume for ETFs
HISTORICAL_DATA_START_DATE = '1994-10-09'  # Start date for historical data (up to ~30 years)
HISTORICAL_DATA_END_DATE = datetime.today().strftime('%Y-%m-%d')  # Current date
MARKET_CAP_THRESHOLD = 500000000
HISTORICAL_MONITOR_CSV_FILE = 'historical_data_monitor.csv'  # Log file for historical data downloads
ETF_MONITOR_CSV_FILE = 'etf_filter_monitor.csv'  # Log file for ETF filtering process
STOCK_MONITOR_CSV_FILE = 'stock_latest_date_monitor.csv'
SEQ_LENGTH = 30  # Sequence length for transformer input
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
RISK_AVERSION = 0.5  # Risk aversion parameter for portfolio optimization

# ================================ Helper Functions ================================

def fetch_stocks(api_key, market_cap_threshold):
    """
    Retrieve stock symbols with a market capitalization greater than $500M.

    Args:
        api_key (str): Your Financial Modeling Prep API key.
        market_cap_threshold (int): Market cap threshold in dollars.

    Returns:
        list: A list of stock symbols.
    """
    stocks_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={market_cap_threshold}&apikey={api_key}'
    try:
        response = requests.get(stocks_url)
        response.raise_for_status()  # Check for request errors
        stocks_data = response.json()
        stock_symbols = [stock['symbol'] for stock in stocks_data]
        print(f"Successfully fetched {len(stock_symbols)} stocks with Market Cap > ${market_cap_threshold / 1000000}M.")
        return stock_symbols
    except Exception as e:
        print(f"Failed to fetch stocks: {e}")
        return []

def fetch_latest_date(symbol):
    """
    Fetch the latest trading date for a given stock symbol using Yahoo Finance.

    Args:
        symbol (str): The stock symbol to fetch data for.

    Returns:
        tuple: (symbol, latest_date, status)
               - symbol (str): The stock symbol.
               - latest_date (str or None): Latest available trading date in 'YYYY-MM-DD' format.
               - status (str): 'Success', 'No Data', or 'Error'.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")  # Fetch the most recent trading day

        if hist.empty:
            print(f"No data found for {symbol} on Yahoo Finance. Marking for removal.")
            return (symbol, None, 'No Data')

        latest_date = hist.index[-1].strftime('%Y-%m-%d')
        return (symbol, latest_date, 'Success')

    except Exception as e:
        print(f"Error fetching data for {symbol} on Yahoo Finance: {e}")
        return (symbol, None, 'Error')

def filter_stocks_by_latest_date(stock_symbols, output_dir, max_workers=5):
    """
    Filter out stock symbols that do not have the latest data date using multithreading.

    Args:
        stock_symbols (list): List of stock symbols to filter.
        output_dir (str): Directory where the log CSV will be saved.
        max_workers (int, optional): Number of threads for parallel downloads. Defaults to 5.

    Returns:
        list: Updated list of stock symbols after filtering.
    """
    symbol_latest_dates = {}
    removed_symbols = []
    monitor_csv_path = os.path.join(output_dir, STOCK_MONITOR_CSV_FILE)

    # Initialize the CSV file with headers
    with open(monitor_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Latest_Date', 'Status'])

    print("Filtering stocks based on latest data dates...")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all fetch_latest_date tasks
        futures = {executor.submit(fetch_latest_date, symbol): symbol for symbol in stock_symbols}

        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying symbol data dates"):
            symbol = futures[future]
            try:
                sym, latest_date, status = future.result()
                # Write to CSV
                with open(monitor_csv_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([sym, latest_date if latest_date else 0, status])

                if status == 'Success' and latest_date:
                    symbol_latest_dates[sym] = latest_date
                else:
                    removed_symbols.append(sym)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                with open(monitor_csv_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([symbol, 0, 'Error'])
                removed_symbols.append(symbol)

    # Determine the overall latest date across all symbols
    if not symbol_latest_dates:
        print("No valid symbols found after checking with Yahoo Finance. Exiting.")
        sys.exit(1)

    overall_latest_date = max(symbol_latest_dates.values())
    print(f"Overall latest data date across all symbols: {overall_latest_date}")

    # Remove symbols whose latest date is older than the overall latest date
    symbols_to_remove = [symbol for symbol, date in symbol_latest_dates.items() if date < overall_latest_date]
    for symbol in symbols_to_remove:
        print(f"{symbol} is outdated (Latest: {symbol_latest_dates[symbol]}, Expected: {overall_latest_date}). Removing from dataset.")
        removed_symbols.append(symbol)

    # Remove duplicates in removed_symbols
    removed_symbols = list(set(removed_symbols))

    # Remove the outdated symbols from stock_symbols
    updated_stock_symbols = [symbol for symbol in stock_symbols if symbol not in removed_symbols]

    print(f"Removed {len(removed_symbols)} outdated stock symbols. {len(updated_stock_symbols)} symbols remain.")

    return updated_stock_symbols

def fetch_etfs(api_key):
    """
    Retrieve ETF symbols using the Financial Modeling Prep API.

    Args:
        api_key (str): Your Financial Modeling Prep API key.

    Returns:
        list: A list of ETF symbols.
    """
    etfs_url = f'https://financialmodelingprep.com/api/v3/etf/list?apikey={api_key}'
    try:
        response = requests.get(etfs_url)
        response.raise_for_status()
        etfs_data = response.json()
        etf_symbols = [etf['symbol'] for etf in etfs_data]
        print(f"Successfully fetched {len(etf_symbols)} ETFs.")
        return etf_symbols
    except Exception as e:
        print(f"Failed to fetch ETFs: {e}")
        return []

def fetch_volume(symbol, volume_threshold):
    """
    Fetch the latest trading volume for a given ETF symbol using Yahoo Finance.

    Args:
        symbol (str): The ETF symbol to fetch volume for.
        volume_threshold (int): The minimum volume threshold.

    Returns:
        tuple: (symbol, volume, status)
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")  # Fetch the most recent trading day

        if hist.empty:
            print(f"No data found for ETF: {symbol}")
            return (symbol, 0, 'No Data')

        # Extract the latest trading volume
        volume = int(hist['Volume'].iloc[-1])
        status = 'Pass' if volume >= volume_threshold else 'Fail'
        return (symbol, volume, status)

    except Exception as e:
        print(f"Error fetching volume for {symbol}: {e}")
        return (symbol, 0, 'Error')

def filter_etfs_by_volume(etf_symbols, volume_threshold, output_dir, max_workers=5):
    """
    Filter ETFs based on their trading volume using Yahoo Finance with multithreading.

    Args:
        etf_symbols (list): List of ETF symbols to filter.
        volume_threshold (int): Minimum daily trading volume required.
        output_dir (str): Directory where the log CSV will be saved.
        max_workers (int, optional): Number of threads for parallel downloads. Defaults to 5.

    Returns:
        list: A list of ETFs that passed the volume filter.
    """
    filtered_etfs = []
    failed_symbols = []
    monitor_csv_path = os.path.join(output_dir, ETF_MONITOR_CSV_FILE)
    num_max_workers = max_workers  # Number of threads for parallel downloads

    # Initialize the CSV file with headers
    with open(monitor_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Volume', 'Status'])

    # Define a helper function to write results safely
    def write_result(sym, vol, status):
        with open(monitor_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([sym, vol, status])

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all fetch_volume tasks
        futures = {executor.submit(fetch_volume, symbol, volume_threshold): symbol for symbol in etf_symbols}

        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering ETFs by Volume"):
            symbol = futures[future]
            try:
                sym, vol, status = future.result()
                write_result(sym, vol, status)

                if status == 'Pass':
                    filtered_etfs.append(sym)
                elif status == 'Error':
                    failed_symbols.append(sym)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                write_result(symbol, 0, 'Error')
                failed_symbols.append(symbol)

    print(f"Filtered ETFs with Volume >= {volume_threshold}: {len(filtered_etfs)}")
    if failed_symbols:
        print(f"Encountered issues with {len(failed_symbols)} ETFs.")

    return filtered_etfs

def add_benchmark_indices(benchmark_indices):
    """
    Include benchmark indices in the list of symbols.

    Args:
        benchmark_indices (list): List of benchmark index symbols.

    Returns:
        list: The same list of benchmark indices.
    """
    print(f"Including benchmark indices: {', '.join(benchmark_indices)}")
    return benchmark_indices

def save_symbols_to_excel(stock_symbols, etf_symbols, benchmark_indices, output_dir):
    """
    Save stock, ETF, and benchmark index symbols into an Excel file with separate sheets.

    Args:
        stock_symbols (list): List of stock symbols.
        etf_symbols (list): List of ETF symbols.
        benchmark_indices (list): List of benchmark index symbols.
        output_dir (str): Directory where the Excel file will be saved.
    """
    symbols_excel_path = os.path.join(output_dir, 'symbols.xlsx')
    df_stocks = pd.DataFrame(stock_symbols, columns=['Stock Symbol'])
    df_etfs = pd.DataFrame(etf_symbols, columns=['ETF Symbol'])
    df_benchmarks = pd.DataFrame(benchmark_indices, columns=['Benchmark Symbol'])

    try:
        with pd.ExcelWriter(symbols_excel_path, engine='xlsxwriter') as writer:
            df_stocks.to_excel(writer, sheet_name='Stocks', index=False)
            df_etfs.to_excel(writer, sheet_name='ETFs', index=False)
            df_benchmarks.to_excel(writer, sheet_name='Benchmark Indices', index=False)
        print(f"Saved symbols to {symbols_excel_path}")
    except Exception as e:
        print(f"Error saving symbols to Excel: {e}")

def initialize_database(db_path):
    """
    Set up the SQLite database and create the historical_data table if it doesn't exist.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        tuple: SQLite connection and cursor objects.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create the table with a primary key to avoid duplicate entries
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
        '''
        cursor.execute(create_table_query)

        # Add indexes to speed up queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON historical_data(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON historical_data(date)')
        conn.commit()
        print(f"Initialized SQLite database and ensured table 'historical_data' exists.")
        return conn, cursor
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

def get_latest_date(conn, symbol):
    """
    Retrieve the latest date for a given symbol in the database.

    Args:
        conn (sqlite3.Connection): SQLite connection object.
        symbol (str): The symbol to query.

    Returns:
        str: The latest date in 'YYYY-MM-DD' format, or None if no records exist.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM historical_data WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else None

def insert_data_with_ignore(conn, df):
    """
    Insert data into the historical_data table with IGNORE on conflict.

    Args:
        conn (sqlite3.Connection): SQLite connection object.
        df (pandas.DataFrame): DataFrame containing historical data.
    """
    cursor = conn.cursor()
    for index, row in df.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO historical_data (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['symbol'], row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))
    conn.commit()

def download_historical_data_fmp(symbols, conn, output_dir, api_key=FMP_API_KEY):
    """
    Fetch and update historical data for a list of symbols.

    Args:
        symbols (list): List of symbols to download data for.
        conn (sqlite3.Connection): SQLite connection object.
        output_dir (str): Directory where the log CSV will be saved.
        api_key (str): Your Financial Modeling Prep API key.
    """
    monitor_csv_path = os.path.join(output_dir, HISTORICAL_MONITOR_CSV_FILE)
    rate_limiter = RateLimiter(max_calls=300, period=60)  # 300 requests per 60 seconds
    session = requests.Session()

    with open(monitor_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Status', 'Records Downloaded'])

        for symbol in tqdm(symbols, desc="Downloading Historical Data"):
            try:
                print(f"Fetching data for {symbol}...")

                latest_date = get_latest_date(conn, symbol)
                if latest_date:
                    start_date_dt = datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)
                    start_date = start_date_dt.strftime('%Y-%m-%d')
                    print(f"Latest date in DB for {symbol}: {latest_date}. Fetching data from {start_date} onwards.")
                else:
                    start_date = HISTORICAL_DATA_START_DATE
                    print(f"No existing data for {symbol}. Fetching data from {start_date}.")

                historical_url = (
                    f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}'
                    f'?from={start_date}&to={HISTORICAL_DATA_END_DATE}&apikey={api_key}'
                )

                # Rate limiting
                rate_limiter.wait()

                response = session.get(historical_url)
                response.raise_for_status()
                data = response.json()

                if 'historical' not in data:
                    status = "Error: Unexpected response format"
                    print(f"{symbol}: {status}")
                    writer.writerow([symbol, status, 0])
                    continue

                historical_data = data['historical']
                if not historical_data:
                    print(f"No new historical data available for {symbol}.")
                    status = 'No Data'
                    writer.writerow([symbol, status, 0])
                    continue

                df = pd.DataFrame(historical_data)
                df['symbol'] = symbol
                df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]

                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                records = len(df)
                insert_data_with_ignore(conn, df)
                print(f"Successfully inserted {records} new records for {symbol}.")
                status = 'Success'
                writer.writerow([symbol, status, records])

            except Exception as e:
                print(f"Failed to download data for {symbol}: {e}")
                writer.writerow([symbol, f'Error: {e}', 0])

    print("Finished downloading all historical data.")

def main_data_fetching(output_dir):
    """
    Main function to handle data fetching and updating.
    """
    # Initialize DataLoader
    data_loader = DataLoader(output_dir)

    # Initialize the SQLite database
    conn, cursor = initialize_database(data_loader.db_path)

    # Check if symbols.xlsx exists
    symbols_excel_path = os.path.join(output_dir, 'symbols.xlsx')
    if not os.path.exists(symbols_excel_path):
        print("Fetching and filtering symbols...")

        # Fetch stock symbols
        stock_symbols = fetch_stocks(FMP_API_KEY, MARKET_CAP_THRESHOLD)
        stock_symbols = filter_stocks_by_latest_date(stock_symbols, output_dir)

        # Fetch ETF symbols
        etf_symbols = fetch_etfs(FMP_API_KEY)
        etf_symbols = filter_etfs_by_volume(etf_symbols, ETF_VOLUME_THRESHOLD, output_dir)

        # Add benchmark indices
        benchmark_symbols = add_benchmark_indices(BENCHMARK_INDICES)

        # Save symbols to Excel
        save_symbols_to_excel(stock_symbols, etf_symbols, benchmark_symbols, output_dir)
    else:
        print("symbols.xlsx already exists. Skipping symbol fetching.")

    # Load symbols from Excel
    symbols = data_loader.load_symbols()

    # Download historical data
    download_historical_data_fmp(
        symbols,
        conn,
        output_dir,
        api_key=FMP_API_KEY
    )

    # Close the database connection after downloading
    cursor.close()
    conn.close()
    print("Closed the SQLite database connection.")
